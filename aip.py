# region libraries
# 参考torch_pruning.examples.transformers.prune_hf_vit.py
import os, time, sys, copy, gc, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd

import torch_pruning as tp
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention, ASTSelfOutput

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed, validate, fine_tune_amp, restore_grads
# endregion

def prune_ast(model, example_inputs, trainloader, valloader, **kwargs):
    # === 确保只能传一个剪枝参数 ===
    assert ('pruning_ratio' in kwargs) ^ ('head_pruning_ratio' in kwargs), \
        "❌ Must provide exactly one of pruning_ratio or head_pruning_ratio!"

    # === 忽略层设置 ===
    unwrapped_parameters = [
        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
    ]
    ignored_layers = [model.model.classifier]
    # === 提取 attention 中的 Linear 层作为 key ===
    num_heads = {}
    for name, m in model.named_modules():
        if isinstance(m, ASTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads    

    # === 配置 Pruner ===
    imp = tp.importance.GroupTaylorImportance()
    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=imp,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        num_heads=num_heads,
        global_pruning=True,
        isomorphic=False,
        prune_num_heads=True,
        prune_head_dims=False,
        **kwargs
    )

    # === 执行剪枝 ===
    for g in pruner.step(interactive=True):
        g.prune()

    # === 更新 ASTAttention 的结构参数 ===
    for name, m in model.named_modules():
        # 更新 ASTAttention 的 num_attention_heads 和 head_size
        if isinstance(m, ASTSelfAttention) and m.query in pruner.num_heads:
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print(f"[{name}] -> num_heads: {m.num_attention_heads}, head_dim: {m.attention_head_size}, all_head_size: {m.all_head_size}")

        # 标记 FFN 变化
        if isinstance(m, nn.Linear):
            if 'encoder' in name and ('fc1' in name or 'fc2' in name):
                in_dim = m.in_features
                out_dim = m.out_features
                print(f"[{name}] FFN Linear -> in: {in_dim}, out: {out_dim}")
            if isinstance(m, nn.Linear):
                print(f"[{name}] -> in: {m.in_features}, out: {m.out_features}")
    return model

def binary_search_prune(
    model,
    grads,
    example_inputs,
    trainloader,
    valloader,
    search_key,
    search_range,
    acc_threshold,
    fine_tune_epochs,
    param_epsilon=1000000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    low, high = search_range
    log = []

    seen_params = set()
    best_params = float('inf')
    best_model = None
    best_checkpoint = None

    while True:
        mid = (low + high) / 2
        # print(f"\n🔍 Trying {search_key} = {mid:.4f}")

        model_copy = copy.deepcopy(model)
        restore_grads(model_copy, grads)
        pruned_model = prune_ast(model_copy, example_inputs, trainloader, valloader, **{search_key: mid})
        # check_ast_attention_consistency(pruned_model)

        pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        checkpoint = fine_tune_amp(device, trainloader, valloader, pruned_model, epochs=fine_tune_epochs)
        val_acc = checkpoint['acc']

        log.append({
            search_key: mid,
            'params': pruned_params,
            'acc': val_acc
        })
        print(f"📉 {search_key}:{mid:.4f}, Params:{pruned_params}, Acc:{val_acc*100:.2f}%")

        # === 停止条件：参数数目重复（近似重复）
        if any(abs(pruned_params - p) < param_epsilon for p in seen_params):
            print(f"🛑 Stop: Param count {pruned_params} already seen (±{param_epsilon})")
            break

        seen_params.add(pruned_params)

        # === 成绩达标：更新最优模型 + 继续剪更狠 ===
        if val_acc >= acc_threshold:
            if pruned_params < best_params:
                best_params = pruned_params
                best_model = copy.deepcopy(pruned_model)
                best_checkpoint = {
                    search_key: mid,
                    'grads': checkpoint['grads'],
                    'state_dict': checkpoint['state_dict'],
                    'params': pruned_params,
                    'acc': val_acc
                }
            low = mid
        else:
            high = mid
        del model_copy, pruned_model  # 清理内存

    if best_model is None:
        raise ValueError("❌ No model met the accuracy threshold during search.")
    return best_model, best_checkpoint, log

def main(args):
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name, fold, seed = args.dataset_name, args.fold, args.seed
    dataset, num_classes, folds = get_hf_dataset(dataset_name)
    example_inputs = dataset[0]['fbank'].unsqueeze(0).to(device)

    set_random_seed(seed)
    train_dataset, val_dataset = split_hf_dataset(dataset, fold=fold)

    batch_size = args.batch_size
    trainloader, valloader = get_dataloaders(train_dataset, val_dataset, batch_size=batch_size)

    # region base model
    model_name = args.model_name
    model = get_model(model_name=model_name, num_classes=num_classes)
    model.to(device)

    log_csv = f"results/AIP-{dataset_name}-{model_name}-fold_{fold}-seed_{seed}.csv"
    if os.path.exists(log_csv):
        raise FileExistsError(f"Log file {log_csv} already exists. Skipping fold {fold}.")

    # 基础模型
    if os.path.exists(f"checkpoint_{fold}.pt"):
        print(f"--Loading model for fold {fold}...")
        checkpoint = torch.load(f"checkpoint_{fold}.pt", weights_only=True)
    else:
        epochs = args.epochs
        print(f"--Fine-tuning model for fold {fold}...")
        checkpoint = fine_tune_amp(device, trainloader, valloader, model, epochs)
        # torch.save(checkpoint, f"checkpoint_{fold}.pt")
    base_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_acc = checkpoint['acc']
    print(f"--seed:{seed}, {dataset_name}_{fold}, {model_name}, Base Params: {base_params}, Accuracy: {100 * base_acc:.2f}%")
    # endregion

    # 开始剪枝
    log = [{'params': base_params, 'acc': base_acc, 'layer': model.get_num_layers(), 'head_pruning_ratio': 0.0, 'pruning_ratio': 0.0}]
    # region 层剪枝 or Given by ADP
    # model, acc = adp(model, params, acc)    
    model.set_layers(7)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint = fine_tune_amp(device, trainloader, valloader, model, epochs)
    model.load_state_dict(checkpoint['state_dict'])
    acc = checkpoint['acc']
    print(f"layer: 7, params: {params}, acc: {100 * acc:.2f}%")
    log.append({'params': params, 'acc': acc, 'layer': 7, 'head_pruning_ratio': 0.0, 'pruning_ratio': 0.0})
    # endregion

    # region Binary search head剪枝
    grads = checkpoint['grads']
    model, checkpoint, log2 = binary_search_prune(
        model,
        grads,
        example_inputs,
        trainloader,
        valloader,
        search_key="head_pruning_ratio",
        search_range=(0.0, 1.0),
        acc_threshold=args.acc_threshold,
        fine_tune_epochs=args.epochs
    )

    for d in log2:
        d.update({'layer': 7, 'pruning_ratio': 0})
    # endregion

    # region Binary search FFN剪枝
    model.load_state_dict(checkpoint['state_dict'])

    grads = checkpoint['grads']
    model, checkpoint, log3 = binary_search_prune(
        model, # 有梯度的基础模型
        grads,
        example_inputs,
        trainloader,
        valloader,
        search_key="pruning_ratio",
        search_range=(0.0, 1.0),
        acc_threshold=args.acc_threshold,
        fine_tune_epochs=args.epochs
    )
    for d in log3:
        d.update({'layer': 7, 'head_pruning_ratio': 0})
    print(f"params: {params}, acc: {100 * acc:.2f}%")
    # endregion

    # 汇总结果
    pd.DataFrame(log + log2 + log3).to_csv(log_csv, index=False)
    print(f"Results saved to {log_csv}, total time: {time.time() - t0:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIP for AST")
    parser.add_argument('--dataset_name', type=str, default='esc50', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='ast', help='Model name')
    parser.add_argument('--fold', type=int, default=1, help='Fold number')
    parser.add_argument('--seed', type=int, default=6, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='不能更大了')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for fine-tuning')
    parser.add_argument('--acc_threshold', type=float, default=0.9325, help='Accuracy threshold for pruning')

    args = parser.parse_args()
    main(args)
    # python aip.py 2>&1 | tee aip.log

    # 使用方法示例：
    # python aip.py --fold 1 --seed 6 --epochs 100 --acc_threshold 0.93 2>&1 | tee aip_1_6.log
