
import os
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from torch_optimizer import Ranger
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# Device & Paths (unchanged base)
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/data/dn564/sampledv11_cropped'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f'/data/dn564/experiments/simple_baseline_{timestamp}')
output_dir.mkdir(parents=True, exist_ok=True)

# New: structured subdirs (like v02)
models_dir = output_dir / 'models'
plots_dir = output_dir / 'plots'
logs_dir = output_dir / 'logs'
results_dir = output_dir / 'results'
for d in [models_dir, plots_dir, logs_dir, results_dir]:
    d.mkdir(exist_ok=True)

print(f"[INFO] All outputs will be saved to: {output_dir}")

# -------------------------------
# Logging helper
# -------------------------------
log_file = logs_dir / 'training_log.txt'
def log_message(msg: str, echo: bool = True):
    if echo:
        print(msg)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

# -------------------------------
# Model (unchanged architecture)
# -------------------------------
class SimpleCNN(nn.Module):
    """
    Simple CNN baseline based on successful AM defect detection papers:
    - Westphal et al. (2021): 5-layer CNN for SLS defects
    - Caggiano et al. (2019): 4-conv architecture for AM monitoring
    - Snow et al. (2022): Lightweight CNN for LPBF defects

    Architecture: Conv-Pool blocks with increasing filters (32-64-128-256)
    """
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to fixed size
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -------------------------------
# Transforms & Datasets (same idea)
# -------------------------------
'''
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])'''

#exp4
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    AutoAugment(AutoAugmentPolicy.IMAGENET),  # Add this
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=train_transform)
val_dataset   = datasets.ImageFolder(f'{data_dir}/val',   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(f'{data_dir}/test',  transform=val_test_transform)

# -------------------------------
# Class weights & sampler (same)
# -------------------------------
train_targets = [s[1] for s in train_dataset.samples]
class_counts = Counter(train_targets)
num_classes  = len(train_dataset.classes)
class_names  = train_dataset.classes

class_weights = torch.zeros(num_classes, dtype=torch.float32)
total_samples = len(train_targets)

for idx in range(num_classes):
    count = class_counts[idx]
    class_weights[idx] = np.sqrt(total_samples / (num_classes * max(count, 1)))

if 'under_extrusion' in class_names:
    class_weights[class_names.index('under_extrusion')] *= 2.5

print(f"Class weights: {dict(zip(class_names, [float(w) for w in class_weights]))}")

sample_weights = [class_weights[t].item() for t in train_targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights) * 2, replacement=True)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

# -------------------------------
# Model, loss, optimizer, scheduler (unchanged logic)
# -------------------------------
model = SimpleCNN(num_classes=num_classes, dropout_rate=0.3).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        if self.weight is not None:
            ce_loss = F.nll_loss(log_probs, targets, weight=self.weight, reduction='none')
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * ce_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = SmoothCrossEntropyLoss(weight=class_weights.to(device), smoothing=0.1)
#optimizer = Ranger(
 #   filter(lambda p: p.requires_grad, model.parameters()),
#    lr=0.001,
  #  weight_decay=0.01 #ep2
#)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) #exp0-1 &3
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) 
#optimizer = torch_optimizer.Ranger(filter(lambda p: p.requires_grad, model.parameters()),lr=0.001, weight_decay=0.01)​

num_epochs = 120
steps_per_epoch = len(train_loader)
#scheduler = optim.lr_scheduler.OneCycleLR(
 #   optimizer,
  #  max_lr=0.15,
   # epochs=num_epochs,
    #steps_per_epoch=steps_per_epoch,
    #pct_start=0.3,
    #anneal_strategy='cos',
    #div_factor=25,
    #final_div_factor=1000
#) #exp0-1

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) #exp2
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5) exp3
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs) #exp4

# -------------------------------
# Mixup & train/validate (unchanged logic; add logging)
# -------------------------------
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_epoch(model, loader, criterion, optimizer, scheduler, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if use_mixup and np.random.random() > 0.5:
            inputs, ta, tb, lam = mixup_data(inputs, targets, alpha=0.2)
            outputs = model(inputs)
            loss = lam * criterion(outputs, ta) + (1 - lam) * criterion(outputs, tb)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0 for _ in range(num_classes)]
    class_total   = [0 for _ in range(num_classes)]
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # SAFE per-class accounting (handles batch size 1)
            c = (predicted == targets)
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += int(c[i].item())
                class_total[label]   += 1

    class_acc = [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
                 for i in range(num_classes)]
    return total_loss / len(loader), 100.0 * correct / total, class_acc

# -------------------------------
# Training loop with structured logging & checkpoints
# -------------------------------
log_message("="*60)
log_message("Starting training with Simple Baseline CNN...")
log_message(f"Architecture: 4 conv blocks (32-64-128-256 filters)")
log_message(f"Dataset: {data_dir}")
log_message("="*60)

best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
log_data = []  # for JSON export

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
    val_loss, val_acc, class_accs = validate(model, val_loader, criterion)

    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);   val_accs.append(val_acc)

    # log epoch
    current_lr = optimizer.param_groups[0]['lr']
    log_message(
        f"Epoch [{epoch+1:3d}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
        f"LR: {current_lr:.6f}"
    )
    log_data.append({
        "epoch": epoch + 1,
        "train_loss": float(train_loss),
        "train_acc":  float(train_acc),
        "val_loss":   float(val_loss),
        "val_acc":    float(val_acc),
        "lr":         float(current_lr)
    })

    # optional: per-class every 10 epochs
    if (epoch + 1) % 10 == 0:
        pcs = ", ".join([f"{n}: {a:.1f}%" for n, a in zip(class_names, class_accs)])
        log_message(f"  Per-class Val Acc: {pcs}")

    # checkpoint best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = models_dir / 'best_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': float(val_acc),
            'val_loss': float(val_loss)
        }, ckpt_path)
        log_message(f"  --> New best model saved: {val_acc:.2f}%")

log_message(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# -------------------------------
# Testing (same validate) + predictions for confusion matrix
# -------------------------------
# load best
ckpt = torch.load(models_dir / 'best_model.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

test_loss, test_acc, test_class_accs = validate(model, test_loader, criterion)
log_message("\n" + "="*60)
log_message("TEST RESULTS")
log_message("="*60)
log_message(f"Test Accuracy: {test_acc:.2f}%")
for n, a in zip(class_names, test_class_accs):
    log_message(f"  {n}: {a:.2f}%")

# collect predictions/labels for CM
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.numpy())

cm = confusion_matrix(all_labels, all_preds)

# -------------------------------
# Plots: loss/acc curves, confusion matrix, per-class bars
# -------------------------------
fig = plt.figure(figsize=(18, 10))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(train_losses, label='Train Loss', alpha=0.9)
ax1.plot(val_losses,   label='Val Loss',   alpha=0.9)
ax1.set_title('Training & Validation Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3); ax1.legend()

ax2 = plt.subplot(2, 2, 2)
ax2.plot(train_accs, label='Train Acc', alpha=0.9)
ax2.plot(val_accs,   label='Val Acc',   alpha=0.9)
ax2.set_title('Training & Validation Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3); ax2.legend()

ax3 = plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax3)
ax3.set_title(f'Confusion Matrix (Test: {test_acc:.2f}%)')
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')

ax4 = plt.subplot(2, 2, 4)
bars = ax4.bar(class_names, test_class_accs, edgecolor='navy')
ax4.set_title('Per-Class Test Accuracy'); ax4.set_ylabel('Accuracy (%)'); ax4.set_ylim(0, 105)
for b, acc in zip(bars, test_class_accs):
    ax4.text(b.get_x() + b.get_width()/2, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=9)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle(f'SimpleCNN Results - {timestamp}', fontsize=16)
plt.tight_layout()
fig_path = plots_dir / 'comprehensive_results.png'
plt.savefig(fig_path, dpi=100, bbox_inches='tight')
plt.close(fig)

# -------------------------------
# Save JSON results + history
# -------------------------------
results = {
    "model": "SimpleCNN",
    "timestamp": timestamp,
    "dataset": {
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "test_samples":  len(test_dataset),
        "num_classes":   num_classes,
        "class_names":   class_names
    },
    "model_info": {
        "total_parameters":   int(total_params),
        "trainable_parameters": int(trainable_params),
        "architecture": "4 conv blocks (32-64-128-256) + GAP(7x7) + FC(512,128,classes)"
    },
    "hyperparameters": {
        "batch_size": batch_size,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "scheduler": "OneCycleLR",
        "max_lr": 0.1,
        "pct_start": 0.3,
        "div_factor": 25,
        "final_div_factor": 1000,
        "epochs": num_epochs,
        "loss": "SmoothCrossEntropy (smoothing=0.1, weighted by class weights)"
    },
    "training": {
        "best_val_accuracy": float(best_val_acc),
        "final_train_accuracy": float(train_accs[-1]) if train_accs else None,
        "final_val_accuracy":   float(val_accs[-1])   if val_accs   else None
    },
    "test_results": {
        "overall_accuracy": float(test_acc),
        "per_class_accuracy": {name: float(acc) for name, acc in zip(class_names, test_class_accs)}
    },
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
    "training_history": log_data
}

with open(results_dir / 'complete_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# -------------------------------
# Final console summary
# -------------------------------
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"✔ Model dir:  {models_dir}")
print(f"✔ Plots dir:  {plots_dir}")
print(f"✔ Logs dir:   {logs_dir}")
print(f"✔ Results dir:{results_dir}\n")
print("Key Results:")
print(f"- Best Val Accuracy: {best_val_acc:.2f}%")
print(f"- Test Accuracy:     {test_acc:.2f}%")
print("Per-Class Test Accuracy:")
for name, acc in zip(class_names, test_class_accs):
    status = "✓" if acc >= 85 else "✗"
    print(f"  {status} {name}: {acc:.2f}%")
print(f"\nSaved plot:    {fig_path}")
print(f"Saved JSON:    {results_dir / 'complete_results.json'}")
print(f"Saved log:     {log_file}")
