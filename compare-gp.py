# region libraries
# 参考torch_pruning.examples.transformers.prune_hf_vit.py
import os, sys, copy, gc, itertools, tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd

import torch_pruning as tp
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention, ASTSelfOutput

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed, validate, fine_tune_amp
# endregion

def restore_grads(model, saved_grads):
    if saved_grads is None:
        print("⚠️ [restore_grads] No gradients to restore.")
        return

    restored, skipped = 0, 0
    for name, param in model.named_parameters():
        if name in saved_grads:
            grad_tensor = saved_grads[name].to(param.device)
            if param.grad is None:
                param.grad = grad_tensor.clone()
            else:
                param.grad.copy_(grad_tensor)
            restored += 1
        else:
            skipped += 1

    # print(f"✅ [restore_grads] Restored gradients for {restored} params. Skipped {skipped}.")

def prune_ast(model, example_inputs, trainloader, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    unwrapped_parameters = [
        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
    ]
    ignored_layers, num_heads = [], {}
    for _, m in model.named_modules(): 
        if isinstance(m, ASTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
        # if isinstance(m, ASTSelfOutput):
        #     ignored_layers.append(m.dense)
    ignored_layers.append(model.model.classifier)

    def get_importance_method(pruning_type):
        if pruning_type == 'random':
            return tp.importance.RandomImportance()
        elif pruning_type == 'l1':
            return tp.importance.GroupMagnitudeImportance(p=1)
        elif pruning_type == 'l2':
            return tp.importance.GroupMagnitudeImportance(p=2)
        elif pruning_type == 'taylor':
            return tp.importance.GroupTaylorImportance()
        elif pruning_type == 'hessian':
            return tp.importance.GroupHessianImportance()
        else:
            raise NotImplementedError
    imp = get_importance_method(kwargs.get("pruning_type", "taylor"))

    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=imp,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,        
        num_heads=num_heads,
        pruning_ratio=kwargs.get("pruning_ratio", 0.5),
        global_pruning=kwargs.get("global_pruning", True),
        isomorphic=kwargs.get("isomorphic", False),
        prune_num_heads=kwargs.get("prune_num_heads", True),
        prune_head_dims=kwargs.get("prune_head_dims", False),
        head_pruning_ratio=kwargs.get("head_pruning_ratio", 0.0),
    )

    # # === Load grads ===
    # if kwargs.get('grads') is not None:
    #     restore_grads(model, kwargs["grads"])

    # # === Accumulate gradients (for Taylor) ===
    # if kwargs.pruning_type == 'taylor':
    #     model.train()
    #     for i, (x, y) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
    #         if i >= args.taylor_batchs:
    #             break
    #         x, y = x.to(device), y.to(device)
    #         logits = model(x).logits
    #         loss = F.cross_entropy(logits, y)
    #         loss.backward()

    for g in pruner.step(interactive=True):
        g.prune()

    for m in model.modules():
        if isinstance(m, ASTSelfAttention):
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
    return model

def grid_search_prune(
    base_model,
    checkpoint, 
    trainloader,
    valloader,
    example_inputs,
    acc_threshold: float,
    log_entry: dict
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    minimal_params = float('inf')
    best_entry = None
    log = []
    head_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for head_ratio, pruning_ratio in tqdm.tqdm(itertools.product(head_ratios, pruning_ratios), desc="Grid Search"):
        model = copy.deepcopy(base_model)
        model.load_state_dict({k: v.cpu() for k, v in checkpoint["state_dict"].items()})
        restore_grads(model, {k: v.cpu() for k, v in checkpoint["grads"].items()})

        # 注意：search_key 通过 kwargs 注入
        pruned_model = prune_ast(model, example_inputs, trainloader)
        pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        tuned_point = fine_tune_amp(device, trainloader, valloader, pruned_model, epochs=30)

        pruned_model.load_state_dict(tuned_point['state_dict'])  # 确保微调后的参数被加载
        _, pruned_acc = validate(valloader, pruned_model, nn.CrossEntropyLoss())

        # === 构造当前日志条目 ===
        log_entry_i = log_entry.copy()
        log_entry_i.update({
            'pruning_ratio': pruning_ratio,
            'head_pruning_ratio': head_ratio,
            'pruned_params': pruned_params,
            'pruned_acc': pruned_acc,
            'current_best': False
        })

        if pruned_acc >= acc_threshold:
            if pruned_params < minimal_params:
                minimal_params = pruned_params
                best_entry = log_entry_i.copy()
                log_entry_i['current_best'] = True

        print(f"🔍 Trying head_ratio={head_ratio}, ffn_ratio={pruning_ratio} → Params = {pruned_params}, Acc = {100 * pruned_acc:.2f}%, 合格:{log_entry_i['current_best']}")
        log.append(log_entry_i)

        del model, pruned_model
        torch.cuda.empty_cache()

    return best_entry, log

def main(
    seed=42,
    dataset_name='esc50',
    model_name='ast',
    batch_size=16,
    epochs=30
):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, num_classes, folds = get_hf_dataset(dataset_name)
    example_inputs = dataset[0]['fbank'].unsqueeze(0).to(device)

    # 分开保存文件
    for fold in range(1, folds + 1):
        log_csv = f"gp-{fold}.csv"
        if os.path.exists(log_csv):
            print(f"--Log file {log_csv} already exists. Skipping fold {fold}.")
            continue
        print(f"\n========== Fold {fold}/{folds} ==========")
        train_dataset, val_dataset = split_hf_dataset(dataset, fold=fold)
        trainloader, valloader = get_dataloaders(train_dataset, val_dataset, batch_size=batch_size)

        base_model = get_model(model_name=model_name, num_classes=num_classes)
        base_model.to(device)

        print("Fine-tuning model...")
        if os.path.exists(f"result_{fold}.pt"):
            print(f"--Loading model for fold {fold}...")
            checkpoint = torch.load(f"result_{fold}.pt", weights_only=True)
        else:
            print(f"--Fine-tuning model for fold {fold}...")
            checkpoint = fine_tune_amp(device, trainloader, valloader, base_model, epochs)
            torch.save(checkpoint, f"result_{fold}.pt")
        base_model.load_state_dict(checkpoint['state_dict'])

        base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        base_acc = checkpoint['acc']
        print(f"--Fold: {fold}, Base Params: {base_params}, Accuracy: {100 * base_acc:.2f}%")
        torch.cuda.empty_cache(); gc.collect()

        log_entry = {
            "fold": fold,
            "keep_layers": 12,
            "pruning_ratio": 0.,
            "head_pruning_ratio": 0.,
            "pruned_params": base_params,
            "pruned_acc": base_acc,
            "current_best": True  # 初始状态下，base_model是最优的
        }

        base_model.set_layers(7)
        checkpoint = fine_tune_amp(device, trainloader, valloader, base_model, epochs)
        base_model.load_state_dict(checkpoint['state_dict'])
        base_acc = checkpoint['acc']

        # Given by ADP
        best_entry1 = {
                'fold': fold,
                'keep_layers': 7,
                "pruning_ratio": 0.,
                "head_pruning_ratio": 0.,                
                'pruned_params': sum(p.numel() for p in base_model.parameters() if p.requires_grad),
                'pruned_acc': base_acc, #
                'current_best': True
            }
        log1 = [best_entry1]

        print(f"**** Minimal layers: {best_entry1['keep_layers']}, Params: {best_entry1['pruned_params']}, Acc: {100 * best_entry1['pruned_acc']:.2f}%")
        base_model.set_layers(best_entry1['keep_layers'])
        checkpoint['state_dict'] = base_model.state_dict()

        # Stage 2: Search head_pruning_ratio
        print(f"==> Stage 2: Searching head_pruning_ratio")
        log_entry.update(best_entry1)        
        best_entry2, log2 = grid_search_prune(
            base_model,
            checkpoint,
            trainloader,
            valloader,
            example_inputs,
            acc_threshold=0.93,
            log_entry=log_entry.copy(),
        )
        # Merge and save log
        log_df = pd.DataFrame(log1 + log2)
        log_df.to_csv(log_csv, index=False)

if __name__ == '__main__':
    main()
