import glob, pandas as pd

# 6 59 27 96
csv_files = [
    'results/ADP-esc50-ast-fold_3.csv',
    'results/ADP-esc50-ast-fold_4.csv',
    'results/ADP-esc50-ast-fold_5.csv',
    ]

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    id = df[df['layer']==6]['acc'].idxmax()
    row = df.loc[id]
    print(f"Layer: {row['layer']}, Acc: {row['acc']:.2f}, Seed: {row['seed']}")

# import torch
# checkpoints = torch.load('checkpoints/fold_1_seed_0.pt')
# checkpoints.keys()

# from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
# from models.model_getter import get_model

# model = get_model(model_name='ast', num_classes=50)
# model.set_layers(7)

# model.load_state_dict(checkpoints[6]['state_dict'])
# pre_acc = checkpoints[6]['acc']

# # 验证子模型的acc
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# dataset, num_classes, folds = get_hf_dataset('esc50')
# trainset, valset = split_hf_dataset(dataset, fold=1)
# trainloader, valloader = get_dataloaders(trainset, valset, batch_size=32, num_workers=2)
# from src.traintest import validate, validate_amp

# # _, pos_acc = validate_amp(valloader=valloader, model=model, criterion=torch.nn.CrossEntropyLoss())
# _, pos_acc = validate(valloader=valloader, model=model, criterion=torch.nn.CrossEntropyLoss())
# print(f"Pre acc: {100 * pre_acc:.2f}%, Post acc: {100 * pos_acc:.2f}%")
