import random, copy
import torch, torch.nn as nn
from transformers import ViTForImageClassification
import torchaudio.transforms as T

# 
class WrappedViT(nn.Module):
    def __init__(self, num_classes=1000, time_mask=56, freq_mask=56, p_augment=0.5):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.num_layers = len(self.model.vit.encoder.layer)

        self.p_augment = p_augment
        self.output_hidden_states = False

        # 修改分类头
        if num_classes != 1000:
            self.model.classifier = nn.Linear(768, num_classes)

        # 数据增强模块
        self.augmentations = nn.ModuleList()
        if time_mask > 0:
            self.augmentations.append(T.TimeMasking(time_mask_param=time_mask))
        if freq_mask > 0:
            self.augmentations.append(T.FrequencyMasking(freq_mask_param=freq_mask))

    def forward(self, S: torch.Tensor):
        # 数据增强（仅在训练模式下生效）
        if self.training and random.random() < self.p_augment:
            for aug in self.augmentations:
                S = aug(S)

        # 调用原始模型的 forward 方法
        return self.model(S, output_hidden_states=self.output_hidden_states)

    def retain_layers(self, depth: int):
        """
        截取模型的前 depth 层。
        """
        if depth > self.num_layers:
            raise ValueError(f"depth must be <= {self.num_layers}, but got {depth}.")
        self.model.vit.encoder.layer = self.model.vit.encoder.layer[:depth]
        self.num_layers = depth
        return self  # 返回修改后的模型，便于链式调用

    def get_para_list(self):
        total_param = sum(p.numel() for p in self.model.parameters())
        para_list = []
        for i in range(self.num_layers):
            pruned_param = sum(p.numel() for p in self.model.vit.encoder.layer[i+1:].parameters())
            para_list.append(total_param - pruned_param)

        return para_list


if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    
    model = WrappedViT(num_classes=10)
    model.output_hidden_states = True
    print('--num_params:', sum(p.numel() for p in model.parameters()))

    pruned_model = model.retain_layers(12)
    print('--rest num_params:', sum(p.numel() for p in model.parameters()))
    print('--pruned num_params:', sum(p.numel() for p in pruned_model.parameters()))

    outputs = model(x)
    print('--model outputs.shape:', outputs['logits'].shape)
    print('--length hidden_states:', len(outputs['hidden_states']))

    outputs = pruned_model(x)
    print('--pruned outputs.shape:', outputs['logits'].shape)

    para_list = model.get_para_list()
    print('--para_list:', para_list)
    # inputs = feature_extractor(x, sampling_rate=16000, return_tensors="pt")['input_values']
    # inputs.shape
