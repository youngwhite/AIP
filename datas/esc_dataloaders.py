# hugging face dataset ESC-50
import os, tempfile, zipfile, tarfile, requests, numpy as np
from filelock import FileLock
import torch, torchaudio
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor
import datasets

def download_and_extract_esc50(cache_dir: str):
    zip_file = os.path.join(cache_dir, "ESC-50-master.zip")
    extracted_dir = os.path.join(cache_dir, "ESC-50-master")

    if not os.path.exists(extracted_dir):
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(zip_file):
            dataset_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
            response = requests.get(dataset_url, stream=True)
            with open(zip_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
    return extracted_dir

def download_and_extract_urbansound8k(cache_dir: str):
    tar_file = os.path.join(cache_dir, "UrbanSound8K.tar.gz")
    extracted_dir = os.path.join(cache_dir, "UrbanSound8K")

    if not os.path.exists(extracted_dir):
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(tar_file):
            dataset_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
            print("Downloading UrbanSound8K dataset from Zenodo...")
            response = requests.get(dataset_url, stream=True, allow_redirects=True)
            with open(tar_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print("Download complete!")

        print("Extracting UrbanSound8K dataset...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=cache_dir)
        print("Extraction complete!")

    return extracted_dir

def get_hf_dataset(dataset_name: str, num_proc: int=4):
    if dataset_name == 'cifar10':
        dataset = datasets.load_dataset(dataset_name)
        dataset['train'] = dataset['train'].rename_columns({'img': 'fbank', 'label': 'target'})
        dataset['test'] = dataset['test'].rename_columns({'img': 'fbank', 'label': 'target'})

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        def transform_batch(batch):
            batch['fbank'] = torch.stack([transform(img.convert('RGB')) for img in batch['fbank']])
            return batch

        dataset['train'].set_transform(transform_batch)
        dataset['test'].set_transform(transform_batch)
        return dataset, 10, 1

    cache_dir = os.path.join(tempfile.gettempdir(), f"{dataset_name}_cache")
    if dataset_name == 'esc50':
        num_classes, folds = 50, 5
        root = download_and_extract_esc50(cache_dir)
        csv_path = os.path.join(root, "meta", "esc50.csv")
        dataset = datasets.Dataset.from_csv(csv_path)
    elif dataset_name == 'urbansound8k':
        num_classes, folds = 10, 10
        root = download_and_extract_urbansound8k(cache_dir)
        csv_path = os.path.join(root, "metadata", "UrbanSound8K.csv")
        dataset = datasets.Dataset.from_csv(csv_path)
    else:
        raise ValueError(f"--Unknown dataset_name: {dataset_name}")

    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    if dataset_name == 'esc50':
        def preprocess(example):
            audio_path = os.path.join(root, "audio", example["filename"])
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0)
            features = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            example["fbank"] = features["input_values"].squeeze(0)
            example["target"] = int(example["target"])
            return example

    elif dataset_name == 'urbansound8k':
        def preprocess(example):
            audio_path = os.path.join(root, "audio", f"fold{example['fold']}", example["slice_file_name"])
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0)
            features = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
            assert isinstance(features["input_values"], torch.Tensor)
            example["fbank"] = features["input_values"].squeeze(0)
            example["target"] = example["classID"]
            return example

    assert preprocess is not None, "❌ Preprocess function not defined."

    with FileLock(os.path.expanduser("~/.data.lock")):
        cached_features_path = os.path.join(cache_dir, f"{dataset_name}_features.arrow")
        if os.path.exists(cached_features_path):
            dataset = datasets.Dataset.load_from_disk(cached_features_path)
        else:
            dataset = dataset.map(preprocess, num_proc=num_proc)
            dataset.save_to_disk(cached_features_path)

        dataset.set_format(type='torch', columns=['fbank', 'target', 'fold'])
        return dataset, num_classes, folds

def split_hf_dataset(dataset: datasets.arrow_dataset.Dataset, fold: int):
    if 'train' in dataset.column_names and 'test' in dataset.column_names:
        trainset, valset = dataset["train"], dataset["test"]
        return trainset, valset

    folds = np.array(dataset['fold'])
    train_indices, val_indices = np.where(folds != fold)[0], np.where(folds == fold)[0]
    trainset, valset = dataset.select(train_indices), dataset.select(val_indices)
    return trainset, valset

def get_dataloaders(
    trainset: datasets.arrow_dataset.Dataset,
    valset: datasets.arrow_dataset.Dataset,
    batch_size: int=48,   
    num_workers: int=4,
    pin_memory: bool=True,
    seed: int=42
):
    def collate_fn(batch):
        fbanks = [item["fbank"] for item in batch]
        targets = [item["target"] for item in batch]
        return torch.stack(fbanks), torch.tensor(targets)
    def seed_worker(worker_id):
        np.random.seed(worker_id + 42)  # 确保每个worker的种子唯一且可重复
    
    trainloader = DataLoader(
        trainset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory
        )
    valloader = DataLoader(valset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return trainloader, valloader

def get_target2label(dataset_name: str):
    cache_dir = os.path.join(tempfile.gettempdir(), f"{dataset_name}_cache")
    if dataset_name == 'esc50':
        num_classes, folds = 50, 5
        root = download_and_extract_esc50(cache_dir)
    
    csv_path = os.path.join(root, "meta", "esc50.csv")
    dataset = datasets.Dataset.from_csv(csv_path)

    target2label = {target: label for target, label in zip(dataset['target'], dataset['category'])}
    return target2label

if __name__ == '__main__':
    dataset, num_classes, folds = get_hf_dataset('esc50') 
    trainset, valset = split_hf_dataset(dataset, fold=1) 
    print(f"--trainset: {len(trainset)}, valset: {len(valset)}") 

    # 
    trainloader, valloader= get_dataloaders(
        trainset=trainset,
        valset=valset,
        num_workers=4, 
        batch_size=2
        )
    X, y = next(iter(trainloader)) 

    print(f"--X.shape: {X.shape}, target:{y}")
