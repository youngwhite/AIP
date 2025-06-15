import time, random, copy, tqdm
import torch, torch.nn as nn, numpy as np
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup


def set_random_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(device, trainloader, model, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs  # for models returning Tensor

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += inputs.size(0)

        del inputs, targets, outputs, loss

    avg_loss, avg_acc = total_loss / total_samples, total_correct / total_samples
    return avg_loss, avg_acc

def validate(valloader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs  # for models returning Tensor

            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)  # Adjust for batch-averaged loss
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += len(labels)
            del inputs, labels, logits  # optional

    avg_loss, accuracy = total_loss / total_samples, total_correct / total_samples
    return avg_loss, accuracy

def fine_tune(device, trainloader, valloader, model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(device, trainloader, model, optimizer, criterion)
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        val_loss, val_acc = validate(valloader, model, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            model_copy = copy.deepcopy(model)

        print(f"--Epoch {epoch+1}/{args.epochs}, "
              f"Loss T/V={train_loss:.4f}/{val_loss:.4f}, "
              f"Acc T/V={100*train_acc:.2f}/{100*val_acc:.2f}%, "
              f"Max Mem={max_mem:.2f} GB, "
              f"Time={time.time()-t0:.2f}s")

    return {
        'model': model_copy,
        'acc': best_acc
    }

from torch.nn.utils import clip_grad_norm_

def train_epoch_amp(device, trainloader, model, optimizer, criterion, scaler, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1).long()  # 分类任务

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=True):
            outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if max_grad_norm and max_grad_norm > 0:
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_samples += inputs.size(0)

        del inputs, targets, logits, loss

    return total_loss / total_samples, total_correct / total_samples

def validate_amp(valloader, model, criterion):
    device = next(model.parameters()).device
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad(), autocast(device_type=device.type, enabled=True):
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1).long()

            outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs

            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += len(labels)

            del inputs, labels, logits

    return total_loss / total_samples, total_correct / total_samples

def fine_tune_amp(device, trainloader, valloader, model, epochs):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(trainloader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = GradScaler()
    best_state, best_acc, best_grads = None, 0.0, None

    # for epoch in range(1, 1 + epochs):
    for epoch in tqdm.tqdm(range(1, 1 + epochs), desc="Fine-tuning"):
        torch.cuda.reset_peak_memory_stats(device)

        # === 单个 epoch 训练 ===
        train_loss, train_acc = train_epoch_amp(
            device, trainloader, model, optimizer, criterion, scaler, max_grad_norm=1.0
        )

        # === 保存当前 epoch 的 .grad（注意是训练后立即提取）===
        current_grads = {
            name: p.grad.clone().detach()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        # === 验证阶段 ===
        val_loss, val_acc = validate_amp(valloader, model, criterion)

        # === 保存最优精度对应的状态 ===
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_grads = {name: g.clone().detach() for name, g in current_grads.items()}

        scheduler.step()

    return {
        'acc': best_acc,
        'state_dict': best_state,
        'grads': best_grads,
        'epoch': epoch
    }

def restore_grads(model, saved_grads: dict):
    if saved_grads is None:
        raise ValueError("⚠️ No gradients to restore.")

    restored, skipped = 0, 0
    for name, param in model.named_parameters():
        if name in saved_grads:
            grad_tensor = saved_grads[name].to(param.device)
            if param.shape != grad_tensor.shape:
                raise ValueError(f"Shape mismatch in {name}: {param.shape} vs {grad_tensor.shape}")            
            if param.grad is None:
                param.grad = grad_tensor.clone()
            else:
                param.grad.copy_(grad_tensor)
            restored += 1
        else:
            skipped += 1
    # print(f"✅ [restore_grads] Restored gradients for {restored} params. Skipped {skipped}.")

