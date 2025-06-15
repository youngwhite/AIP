import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import fine_tune_amp, restore_grads

def prune_ast(model, example_inputs, trainloader, valloader, **kwargs):
    assert not ('pruning_ratio' in kwargs and 'head_pruning_ratio' in kwargs), \
        "❌ Cannot use both pruning_ratio and head_pruning_ratio at the same time!"

    # === 忽略层和head映射 ===
    unwrapped_parameters = [
        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
    ]
    # === 忽略的层 ===
    ignored_layers = [model.model.classifier]
    # === 提取 attention 中的 Linear 层作为 key ===
    num_heads = {}
    for name, m in model.named_modules():
        if isinstance(m, ASTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

    # === 配置剪枝器 ===
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

    # # === 梯度积累，用于Taylor重要性评估 ===
    # model.train()
    # model.zero_grad()
    # taylor_batchs = 3
    # for i, (x, y) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), desc="Accumulating Gradients"):
    #     if i >= taylor_batchs:
    #         break
    #     x, y = x.to(device), y.to(device)
    #     logits = model(x).logits
    #     loss = F.cross_entropy(logits, y)
    #     loss.backward()

    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"{name} grad norm: {param.grad.norm().item():.4f}")
    # === 执行剪枝 ===
    for g in pruner.step(interactive=True):
        g.prune()

    # === 更新attention head参数（必须手动）===
    for name, m in model.named_modules():
        if isinstance(m, ASTSelfAttention) and m.query in pruner.num_heads:
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print(f"[{name}] -> num_heads: {m.num_attention_heads}, head_dim: {m.attention_head_size}, all_head_size: {m.all_head_size}")
        if isinstance(m, nn.Linear):
            if m.in_features != m.out_features:
                print(f"[{name}] FFN Layer: in={m.in_features}, out={m.out_features}")

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, num_classes, folds = get_hf_dataset('esc50')

    train_dataset, val_dataset = split_hf_dataset(dataset, fold=1)
    trainloader, valloader = get_dataloaders(train_dataset, val_dataset, batch_size=16)

    example_inputs = next(iter(trainloader))[0].to(device)
    model = get_model(model_name='ast', num_classes=50)
    model.set_layers(7)
    model.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = fine_tune_amp(device, trainloader, valloader, model, epochs=3)
    restore_grads(model, checkpoint['grads'])    

    before_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = prune_ast(model, example_inputs, trainloader, valloader, head_pruning_ratio=0.5)
    after_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {before_params} -> {after_params}")

    output = model(example_inputs)
    print(f"Output shape: {output.logits.shape}")
    
    model.to(device)
    checkpoint = fine_tune_amp(device, trainloader, valloader, model, epochs=5)
    print(f"acc: {checkpoint['acc']:.4f}")

    # before_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model = prune_ast(model, example_inputs, trainloader, valloader, pruning_ratio=0.3)
    # after_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Parameters: {before_params} -> {after_params}")    
