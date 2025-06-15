import random, copy
import torch, torch.nn as nn
from transformers import ASTForAudioClassification
import torchaudio.transforms as T
# pip install --upgrade transformers datasets

class WrappedAST(nn.Module):
    def __init__(self, num_classes=50, time_mask=192, freq_mask=48, p_augment=0.5):
        super().__init__()

        # === 加载预训练模型 ===
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        # === 修改分类头（保证模型输出类别数正确）===
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Linear(hidden_size, num_classes)
        self.model.config.num_labels = num_classes  # 确保 config 同步

        # === 输出 hidden_states 支持剪枝与蒸馏 ===
        self.output_hidden_states = False

        # === 时间/频率 Mask 增强 ===
        self.p_augment = p_augment
        self.augmentations = nn.Sequential()
        if time_mask > 0:
            self.augmentations.add_module("time_mask", T.TimeMasking(time_mask_param=time_mask))
        if freq_mask > 0:
            self.augmentations.add_module("freq_mask", T.FrequencyMasking(freq_mask_param=freq_mask))

    def forward(self, S: torch.Tensor):
        if S.shape[-2:] != (1024, 128):
            raise ValueError(f"Input shape must be [*, 1024, 128], but got {S.shape}")
        if self.training and self.p_augment > 0 and random.random() < self.p_augment:
            S = self.augmentations(S)
        return self.model(S, output_hidden_states=self.output_hidden_states)
        
    def count_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def count_depth_params(self):
        """
        统计每保留前 k 层 encoder 时的总参数量（排除 pre_logits 和 distillation_token 等）
        """
        model = self.model
        ast = model.audio_spectrogram_transformer

        # === 1. Embedding（去掉 distillation_token）
        embedding_params = sum(
            p.numel() for name, p in ast.embeddings.named_parameters()
            if "distillation_token" not in name
        )

        # === 2. 每层 encoder
        encoder_layers = ast.encoder.layer
        layer_params = [sum(p.numel() for p in layer.parameters()) for layer in encoder_layers]

        # === 3. 分类头
        classifier_params = sum(p.numel() for p in model.classifier.parameters())

        # === 4. 去除 pre_logits 和 pooler
        excluded = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "pre_logits" in name or "pooler" in name
        )

        # === 5. 构建累积参数列表
        depth_params = []
        encoder_accum = 0
        for i in range(len(layer_params)):
            encoder_accum += layer_params[i]
            total = embedding_params + encoder_accum + classifier_params
            depth_params.append(total)

        return depth_params

    def get_num_layers(self):
        return len(self.model.audio_spectrogram_transformer.encoder.layer)

    def set_layers(self, num_layers: int):
        if not (1 <= num_layers <= self.get_num_layers()):
            raise ValueError(f"num_layers must be in range [1, {self.get_num_layers()}]")
        self.model.audio_spectrogram_transformer.encoder.layer = \
            self.model.audio_spectrogram_transformer.encoder.layer[:num_layers]
        return self

if __name__ == '__main__':
    import torch
    x = torch.rand(4, 1024, 128)

    model = WrappedAST(num_classes=50)
    model.output_hidden_states = True

    outputs = model(x)
    print('--model outputs.shape:', outputs['logits'].shape)
    print('--length hidden_states:', len(outputs['hidden_states']))

    model.set_layers(7)
    para = model.count_params()
    print('--para:', para)
