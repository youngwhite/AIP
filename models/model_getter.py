# imported at root
import torch.nn as nn
from models.AST.ast_ import WrappedAST
from models.ViT.vit_ import WrappedViT

def get_model(model_name: str, num_classes: int, **kwargs):
    if model_name == 'ast':
        model = WrappedAST(num_classes=num_classes)
    elif model_name == 'vit':
        model = WrappedViT(num_classes=num_classes)
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model

if __name__ == '__main__':
    import torch
    # x = torch.rand(4, 1024, 128)
    x = torch.rand(4, 3, 224, 224)

    # model = get_model(model_name='ast', num_classes=50)
    model = get_model(model_name='vit', num_classes=10)
    print('--num_layers:', model.num_layers)

    outputs = model(x)
    print('outputs.shape:', outputs['logits'].shape)
