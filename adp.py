# region 
import os, time, copy, tqdm, argparse, pickle, math, csv, itertools
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from torch.nn.functional import kl_div, log_softmax, softmax

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed
# endregion

def train_fold(trainloader, valloader, config):
    fold, seed = config['fold'], config['seed']

    set_random_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(config['model_name'], config['num_classes'])
    model.output_hidden_states = True
    model.to(device)

    n_classifiers = model.get_num_layers()
    middle_classifiers = {
        i: copy.deepcopy(model.model.classifier).to(device)
        for i in range(n_classifiers - 1)
    }

    # Initialize parameters for distillation
    alpha_raw = nn.Parameter(torch.zeros(n_classifiers - 1, device=device))
    T_raw = nn.Parameter(torch.ones(n_classifiers - 1, device=device))

    all_params = list(model.parameters()) + [alpha_raw, T_raw]
    for m in middle_classifiers.values():
        all_params += list(m.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.get("lr", 1e-4))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(trainloader) * config['max_epochs']),
        num_training_steps=len(trainloader) * config['max_epochs']
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    best_record = [
        {
            'fold': fold,
            'seed': seed,
            'layer': i,
            'acc': 0.0,
            'epoch': 0
        } for i in range(n_classifiers)
    ]
    checkpoints = {}
    t0 = time.time()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        for m in middle_classifiers.values():
            m.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=True):
                outputs = model(inputs)
                hidden_states, final_logits = outputs.hidden_states, outputs.logits
                final_loss = criterion(final_logits, labels)

                total_middle_loss = 0
                for i, m_classifier in middle_classifiers.items():
                    logits = m_classifier(hidden_states[i + 1][:, 0, :])
                    ce_loss = criterion(logits, labels)

                    if config.get("use_distill", True):
                        alpha_i = torch.sigmoid(alpha_raw[i])
                        T_i = F.softplus(T_raw[i]) + 1e-3
                        with torch.no_grad():
                            teacher_probs = softmax(final_logits.detach() / T_i, dim=-1)
                        student_log_probs = log_softmax(logits / T_i, dim=-1)
                        kl_loss = kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T_i ** 2)
                        loss_i = (1 - alpha_i) * ce_loss + alpha_i * kl_loss
                    else:
                        loss_i = ce_loss
                    total_middle_loss += loss_i

                total_loss = final_loss + total_middle_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        for m_classifier in middle_classifiers.values():
            m_classifier.eval()

        vcorrects, vtotal = [0] * n_classifiers, 0
        with torch.no_grad(), autocast(device_type='cuda', enabled=True):
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                hidden_states, final_logits = outputs.hidden_states, outputs.logits
                for i in middle_classifiers:
                    logits_i = middle_classifiers[i](hidden_states[i + 1][:, 0, :])
                    vcorrects[i] += (logits_i.argmax(-1) == labels).sum().item()
                vcorrects[-1] += (final_logits.argmax(-1) == labels).sum().item()
                vtotal += inputs.size(0)

        vaccs = [c / vtotal for c in vcorrects]

        for i, acc in enumerate(vaccs):
            if i not in middle_classifiers:
                continue
            if acc > best_record[i]['acc']:
                best_record[i]['acc'] = acc
                best_record[i]['epoch'] = epoch

                model_copy = copy.deepcopy(model)
                model_copy.set_layers(i + 1)
                model_copy.model.classifier.load_state_dict(copy.deepcopy(middle_classifiers[i].state_dict()))

                checkpoints[i] = {
                    "acc": acc,
                    "epoch": epoch,
                    "state_dict": model_copy.state_dict()
                }

        accs_str = [f"{100 * best_record[i]['acc']:.2f}%" for i in range(n_classifiers)]
        print(f"Epoch {epoch}/{config['max_epochs']}, "
              f"Fold {fold}, Seed {seed}, "
              f"Time: {time.time() - t0:.2f}s, "
              f"Best Accs: {accs_str}")
    torch.save(checkpoints, f"checkpoints/fold_{fold}_seed_{seed}.pt")
    return best_record

def main(args):
    dataset_name, fold = args.dataset_name, args.fold
    if args.log_path is None:
        log_path = f'results/ADP-{dataset_name}-{args.model_name}-fold_{fold}.csv'
    else:
        log_path = args.log_path

    dataset, _, _ = get_hf_dataset(dataset_name)
    trainset, valset = split_hf_dataset(dataset, fold=fold)
    trainloader, valloader = get_dataloaders(trainset, valset, 32, 2)

    config = vars(args)

    global_best_record = None
    for seed in list(range(1, 100)):
        config['seed'] = seed
        best_record = train_fold(trainloader, valloader, config)

        # === 即时写入 ===
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        pd.DataFrame(best_record).to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

        # === 更新全局最优 ===
        if global_best_record is None:
            global_best_record = copy.deepcopy(best_record)
        else:
            for i in range(len(best_record)):
                if best_record[i]['acc'] > global_best_record[i]['acc']:
                    global_best_record[i] = copy.deepcopy(best_record[i])

        # === 打印当前全局最优 ===
        print("\n=== Global Best layer-wise accuracies so far ===")
        print(f"Fold {global_best_record[0]['fold']}, Seed {global_best_record[0]['seed']}")
        for i, info in enumerate(global_best_record):
            print(f"Layer {i:2d} | Acc = {info['acc']:.2%} @ Epoch {info['epoch']}")
    return global_best_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive Depth-wise Pruning")
    parser.add_argument('--dataset_name', type=str, default='esc50')
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--model_name', type=str, default='ast')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--log_path', type=str, default=None, help='.csv')
    args = parser.parse_args()
    main(args)

    # 使用方法示例：
    # CUDA_VISIBLE_DEVICES=1 python adp.py --fold 1
    # seeds: 6, 59, 27, 96
