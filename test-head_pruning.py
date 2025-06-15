import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import fine_tune_amp, restore_grads

# === 设置设备 ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === 加载数据集和模型 ===
dataset, num_classes, folds = get_hf_dataset('esc50')
example_inputs = dataset[0]['fbank'].unsqueeze(0).to(device)
train_dataset, val_dataset = split_hf_dataset(dataset, fold=1)
trainloader, valloader = get_dataloaders(train_dataset, val_dataset, batch_size=16)
model = get_model(model_name='ast', num_classes=50)
model.to(device)
model.set_layers(3)

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
    head_pruning_ratio=0.3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = fine_tune_amp(device, trainloader, valloader, model, epochs=2)
restore_grads(model, checkpoint['grads'])

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


# === 测试剪枝后的模型 ===
output = model(example_inputs)
print(f"Output shape: {output.logits.shape}")
