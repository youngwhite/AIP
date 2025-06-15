# aip_clean.py - Finalized pruning script for AST with binary search head pruning

import os, sys, copy, argparse, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import torch_pruning as tp
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed, validate, fine_tune_amp, restore_grads

def deepcopy_model_with_grads(model):
    model_copy = copy.deepcopy(model)
    for (name, orig_param), (_, copy_param) in zip(model.named_parameters(), model_copy.named_parameters()):
        if orig_param.grad is not None:
            copy_param.grad = orig_param.grad.detach().clone()
    return model_copy

def print_pruning_summary(pruner, pruned_info=None, max_groups_to_show=5):
    print("=== [PRUNING SUMMARY] ===")
    if pruned_info is not None:
        print(f"[v1.5.2] Pruned {len(pruned_info)} groups.")
        for i, group in enumerate(pruned_info[:max_groups_to_show]):
            print(f"  - Group {i+1}:")
            for dep in group:
                print(f"    • {dep.name} | shape={tuple(dep.target.shape)} | index={dep.indexes}")
        return

    try:
        groups = pruner.group_manager.groups
        print(f"[v2.x] Found {len(groups)} groups.")
        for i, group in enumerate(groups[:max_groups_to_show]):
            print(f"  - Group {i+1}:")
            for dep in group:
                print(f"    • {dep.name} | shape={tuple(dep.target.shape)} | index={dep.indexes}")
    except AttributeError:
        print("[⚠️] No pruning groups found.")

def prune_ast(model, example_inputs, trainloader, **kwargs):
    print("=== [DEBUG] prune_ast called with kwargs ===")
    for k, v in kwargs.items():
        print(f"=== {k}: {v}")
    pruning_type = kwargs.get("pruning_type", "taylor")
    taylor_batchs = kwargs.get("taylor_batchs", 3)

    # === 选择重要性函数 ===
    if pruning_type == 'taylor':
        imp = tp.importance.TaylorImportance()
    elif pruning_type == 'magnitude':
        imp = tp.importance.MagnitudeImportance()
    else:
        raise ValueError(f"Unknown pruning_type: {pruning_type}")
    print(f"=== [DEBUG] Using importance method: {imp.__class__.__name__}")

    # === 屏蔽 embedding 等特殊参数 ===
    unwrapped_parameters = [
        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
    ]
    ignored_layers = [model.model.classifier]

    num_heads = {}
    for _, m in model.named_modules(): 
        if isinstance(m, ASTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

    # === 构建剪枝器 ===
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
        head_pruning_ratio=kwargs.get("head_pruning_ratio", 0.0),
        pruning_ratio=kwargs.get("pruning_ratio", 0.0)
    )

    # === 记录所有 head 和 ffn 层 ===
    for g in pruner.group_manager.groups:
        name = g.target.name
        param = g.target.module.weight if hasattr(g.target.module, "weight") else None
        if param is not None:
            grad = param.grad
            print(f"[{'✓' if grad is not None else '×'}] {name} has grad, shape: {param.shape}")

    # === 剪枝前参数量 ===
    before_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # === 执行剪枝 ===
    pruner.step()

    # === 剪枝后参数量 ===
    after_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params before: {before_params}, after: {after_params}, pruned: {before_params - after_params}")

    return model

def binary_search_prune(
    model, example_inputs, trainloader, valloader,
    search_key: str, search_range: tuple,
    acc_threshold=0.93, finetune_epochs=10,
    params_epsilon=1_000_000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 检查 grad 是否存在，否则积累新的 ===
    if not any(p.grad is not None for p in model.parameters()):
        print("[INFO] No .grad found, accumulating for Taylor pruning...")
        model.train()
        model.zero_grad()
        for i, (x, y) in enumerate(trainloader):
            if i >= 3: break
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x).logits, y)
            loss.backward()
        print(f"[INFO] Gradients ready.")

    low, high = search_range
    log = []

    prev_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_model, best_acc = None, 0.0

    while True:
        mid = (low + high) / 2
        print(f"\n=== [BINARY SEARCH] Trying {search_key} = {mid:.6f} ===")

        model_copy = deepcopy_model_with_grads(model)
        pruned_model = prune_ast(model_copy, example_inputs, trainloader, **{search_key: mid})
        pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)

        tuned_point = fine_tune_amp(device, trainloader, valloader, pruned_model, epochs=finetune_epochs)
        pruned_acc = tuned_point['acc']

        log.append({
            search_key: mid,
            'params': pruned_params,
            'acc': pruned_acc
        })
        print(f"✅ Params: {pruned_params}, Acc: {100 * pruned_acc:.2f}%")

        if pruned_acc >= acc_threshold and pruned_params < prev_params:
            best_model, best_acc = deepcopy_model_with_grads(pruned_model), pruned_acc

        if abs(pruned_params - prev_params) < params_epsilon:
            print("🛑 Stopping: param change too small.")
            break

        prev_params = pruned_params
        if pruned_acc >= acc_threshold:
            low = mid
        else:
            high = mid

    print(f"\n=== [RESULT] Best acc: {100 * best_acc:.2f}% ===")
    return (best_model or model), log

def main(args):
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset, num_classes, folds = get_hf_dataset(args.dataset_name)
    train_dataset, val_dataset = split_hf_dataset(dataset, fold=args.fold)
    trainloader, valloader = get_dataloaders(train_dataset, val_dataset, batch_size=args.batch_size)

    example_inputs = dataset[0]['fbank'].unsqueeze(0).to(device).to(torch.float32)
    model = get_model(args.model_name, num_classes).to(device)
    model = model.set_layers(7)

    checkpoint = fine_tune_amp(device, trainloader, valloader, model, epochs=5)
    print(f"Initial params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, acc: {100 * checkpoint['acc']:.2f}%")

    restore_grads(model, checkpoint['grads'])
    pruned_model, log = binary_search_prune(
        model, example_inputs, trainloader, valloader,
        search_key="head_pruning_ratio", search_range=(0.0, 1.0),
        acc_threshold=args.acc_threshold, finetune_epochs=args.epochs
    )
    print(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='esc50')
    parser.add_argument('--model_name', type=str, default='ast')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--acc_threshold', type=float, default=0.0)
    main(parser.parse_args())

    # python aip.py 2>&1 | tee aip.log

