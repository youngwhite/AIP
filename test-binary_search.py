import tqdm, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed, validate, restore_grads, fine_tune_amp

def check_ast_attention_consistency(model):
    print("🔍 Checking ASTSelfAttention consistency...\n")
    for name, m in model.named_modules():
        if isinstance(m, ASTSelfAttention):
            q_shape = tuple(m.query.weight.shape)
            k_shape = tuple(m.key.weight.shape)
            v_shape = tuple(m.value.weight.shape)
            num_heads = m.num_attention_heads
            head_dim = m.attention_head_size
            expected = num_heads * head_dim

            print(f"🧠 Layer: {name}")
            print(f"    query.weight.shape = {q_shape}")
            print(f"    key.weight.shape   = {k_shape}")
            print(f"    value.weight.shape = {v_shape}")
            print(f"    num_heads = {num_heads}, head_dim = {head_dim}, expected total dim = {expected}")

            if q_shape[0] != expected:
                print(f"❌ [ERROR] query output dim {q_shape[0]} != expected {expected}")
            else:
                print(f"✅ [OK]    query output dim matches num_heads × head_dim")

            if q_shape != k_shape or q_shape != v_shape:
                print(f"⚠️ [WARN] q/k/v shapes do not match!")

            print("-" * 60)

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
        if isinstance(m, ASTSelfAttention) and m.query in pruner.num_heads:
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            # print(f"[{name}] -> num_heads: {m.num_attention_heads}, head_dim: {m.attention_head_size}, all_head_size: {m.all_head_size}")
        # if isinstance(m, nn.Linear):
        #     if m.in_features != m.out_features:
        #         print(f"[{name}] FFN Layer: in={m.in_features}, out={m.out_features}")

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
    param_epsilon=10_000
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
        print(f"\n🔍 Trying {search_key} = {mid:.4f}")

        model_copy = copy.deepcopy(model)
        restore_grads(model_copy, grads)
        pruned_model = prune_ast(model_copy, example_inputs, trainloader, valloader, **{search_key: mid})
        # check_ast_attention_consistency(pruned_model)

        pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        checkpoint = fine_tune_amp(device, trainloader, valloader, pruned_model, epochs=5)
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
                    'grads': {
                        name: p.grad.clone().detach()
                        for name, p in best_model.named_parameters() if p.grad is not None
                    },
                    'state_dict': best_model.state_dict(),  # ✅ 修复 key 命名为 state_dict
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

if __name__ == "__main__":
    set_random_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, num_classes, folds = get_hf_dataset('esc50')
    train_dataset, val_dataset = split_hf_dataset(dataset, fold=1)
    trainloader, valloader = get_dataloaders(train_dataset, val_dataset, batch_size=16)
    model = get_model(model_name='ast', num_classes=50).set_layers(7).to(device)
    
    checkpoint = fine_tune_amp(device, trainloader, valloader, model, epochs=3)
    model.load_state_dict(checkpoint['state_dict'])
    grads = checkpoint['grads']

    example_inputs = dataset[0]['fbank'].unsqueeze(0).to(device)
    model, checkpoint = binary_search_prune(
        model,
        grads,
        example_inputs,
        trainloader,
        valloader,
        search_key="head_pruning_ratio",
        search_range=(0.0, 1.0),
        acc_threshold=0.6
    )
    
    pruned_params, pruned_acc = checkpoint['params'], checkpoint['acc']
    print(f"🎯 Best pruning ratio:{checkpoint['head_pruning_ratio']}, params:{pruned_params}, acc:{pruned_acc:.4f}")
