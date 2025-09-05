import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns 
import numpy as np
import os
from pathlib import Path
from collections import Counter
import json
from datetime import datetime 
import random
from PIL import Image

# SECTION 1: SETUP
def set_all_seeds(seed=42):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f'/data/dn564/experiments/mini_xception_{timestamp}')
output_dir.mkdir(parents=True, exist_ok=True)

models_dir = output_dir / 'models'
plots_dir = output_dir / 'plots'
logs_dir = output_dir / 'logs'
results_dir = output_dir / 'results'

for dir_path in [models_dir, plots_dir, logs_dir, results_dir]:
    dir_path.mkdir(exist_ok=True)

print(f"All outputs will be saved to: {output_dir}")
data_dir = '/data/dn564/sampledv11_cropped'

# SECTION 2: DATA INTEGRITY CHECK
def check_and_clean_dataset(data_dir):
    """Check dataset integrity and report issues"""
    print("\n" + "="*60)
    print("DATASET INTEGRITY CHECK")
    print("="*60)
    total_issues = 0
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        print(f"\n{split.upper()} SET:")
        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            all_files = os.listdir(class_path)
            valid_images = []
            invalid_images = []
            for img_name in all_files:
                img_path = os.path.join(class_path, img_name)
                try:
                    # Check valid image file
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        invalid_images.append((img_name, "Not an image extension"))
                        continue
                    # Open and verify image
                    with Image.open(img_path) as img:
                        img.verify()
                    # Re-open to load pixel data
                    with Image.open(img_path) as img:
                        img.load()
                    valid_images.append(img_name)
                except Exception as e:
                    invalid_images.append((img_name, str(e)))
                    total_issues += 1
            print(f"  {class_name:20s}: {len(valid_images):3d} valid / {len(all_files):3d} total", end="")
            if invalid_images:
                print(f" ⚠️  {len(invalid_images)} issues found:")
                for img, error in invalid_images[:3]:
                    print(f"    - {img}: {error}")
                if len(invalid_images) > 3:
                    print(f"    ... and {len(invalid_images)-3} more")
            else:
                print(" ✓")
    print(f"\nTotal issues found: {total_issues}")
    return total_issues

# Run integrity check
check_and_clean_dataset(data_dir)

# SECTION 3: MINI-XCEPTION ARCHITECTURE
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # Fixed: instantiate ReLU

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)  # Apply ReLU activation
        return x

class MiniXceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = (in_channels != out_channels) or (stride != 1)
        # Main branch
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=stride)
        # Skip connection
        if self.residual:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual:
            residual = self.skip(residual)
        out = out + residual
        return out

class MiniXception(nn.Module):
    """
    Mini-Xception for Manufacturing Defect Detection  
    Architecture based on successful implementations in:  
    - Ferguson et al. (2018): 92% accuracy on manufacturing defects  
    - Narayanan et al. (2019): Real-time defect detection  
    - Wang et al. (2023): Additive manufacturing defect classification
    """
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super().__init__()
        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)  # Use inplace for consistency
        )
        # Main blocks
        self.block1 = MiniXceptionBlock(64, 128, stride=2)   # 112x112 -> 56x56
        self.block2 = MiniXceptionBlock(128, 256, stride=2)  # 56x56 -> 28x28
        self.block3 = MiniXceptionBlock(256, 512, stride=2)  # 28x28 -> 14x14
        # Optional: additional block
        self.block4 = MiniXceptionBlock(512, 512, stride=1)  # 14x14 -> 14x14
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        # Initialize weights (He initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# SECTION 4: DATA AUGMENTATION
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# SECTION 5: DATASET LOADING
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"Class order: {train_dataset.classes}")
print(f"Class to index: {train_dataset.class_to_idx}")

train_targets = [sample[1] for sample in train_dataset.samples]
class_counts = Counter(train_targets)
class_names = train_dataset.classes
num_classes = len(class_names)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_dataset)} samples")
print(f"  Val:   {len(val_dataset)} samples")
print(f"  Test:  {len(test_dataset)} samples")

print(f"\nClass distribution in training:")
for idx, name in enumerate(class_names):
    count = class_counts[idx]
    percentage = (count / len(train_dataset)) * 100
    print(f"  {name:20s}: {count:3d} samples ({percentage:5.1f}%)")

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# SECTION 6: MODEL AND TRAINING SETUP
model = MiniXception(num_classes=num_classes, dropout_rate=0.4).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Statistics:")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size:           {total_params * 4 / 1024 / 1024:.2f} MB")

# Loss function - CrossEntropy with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer - AdamW like ResNet34
#optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
#                        lr=0.0001, weight_decay=0.01)

# Learning rate scheduler - same as ResNet34
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
##                                                           T_0=10,  # Restart every 10 epochs
#                                                           T_mult=2)  # Double period after restart
#x2
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Line 197
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=150, steps_per_epoch=len(train_loader)) 
#x3
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Line 197
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10) 
# SECTION 7: TRAINING FUNCTIONS
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# SECTION 8: TRAINING LOOP WITH LOGGING
log_file = logs_dir / 'training_log.txt'
log_data = []

def log_message(message, print_msg=True):
    if print_msg:
        print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Training configuration
num_epochs = 80  # Same as ResNet34
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

log_message("="*60)
log_message("Starting Mini-Xception training...")
log_message(f"Model: Mini-Xception")
log_message(f"Dataset: {data_dir}")
log_message(f"Output directory: {output_dir}")
log_message("="*60)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    scheduler.step(val_loss) #was empty
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # Log epoch results
    log_msg = (f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    log_message(log_msg)
    
    # Save epoch data
    log_data.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'lr': optimizer.param_groups[0]["lr"]
    })
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path = models_dir / 'best_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, model_path)
        log_message(f"  --> New best model saved: {val_acc:.2f}%")

log_message(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# SECTION 9: COMPREHENSIVE TESTING WITH AND WITHOUT NOISE
# Load best model
checkpoint = torch.load(models_dir / 'best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

log_message("\n" + "="*60)
log_message("TESTING PHASE")
log_message("="*60)

# Test 1: Clean images (no noise)
log_message("\nTest 1: Clean Images (No Noise)")
test_loss_clean, test_acc_clean = validate_epoch(model, test_loader, criterion, device)
log_message(f"Clean Test Accuracy: {test_acc_clean:.2f}%")

# Test 2: Multiple noise levels (same as ResNet34)
noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
noise_accuracies = []

log_message("\nTest 2: Gaussian Noise Robustness")
log_message("-" * 40)

for noise_std in noise_levels:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Add Gaussian noise after normalization
            if noise_std > 0:
                noise = torch.randn_like(inputs) * noise_std
                inputs = inputs + noise
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    noise_acc = 100. * correct / total
    noise_accuracies.append(noise_acc)
    log_message(f"Noise std={noise_std:.2f}: Accuracy = {noise_acc:.2f}%")

# SECTION 10: CONFUSION MATRIX AND COMPREHENSIVE VISUALIZATION
# Get predictions for confusion matrix (clean test set)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Create comprehensive visualization (same format as ResNet34)
fig = plt.figure(figsize=(20, 12))

# 1. Training curves
ax1 = plt.subplot(2, 3, 1)
ax1.plot(train_losses, label='Train Loss', alpha=0.8)
ax1.plot(val_losses, label='Val Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Accuracy curves
ax2 = plt.subplot(2, 3, 2)
ax2.plot(train_accs, label='Train Acc', alpha=0.8)
ax2.plot(val_accs, label='Val Acc', alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Noise robustness graph
ax3 = plt.subplot(2, 3, 3)
ax3.plot(noise_levels, noise_accuracies, 'bo-', linewidth=2, markersize=8)
ax3.set_xlabel('Gaussian Noise Standard Deviation')
ax3.set_ylabel('Test Accuracy (%)')
ax3.set_title('Model Robustness to Gaussian Noise')
ax3.grid(True, alpha=0.3)
# Add annotations
for i, (x, y) in enumerate(zip(noise_levels, noise_accuracies)):
    ax3.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=8)

# 4. Confusion Matrix
ax4 = plt.subplot(2, 3, 4)
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax4)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title(f'Confusion Matrix (Clean Test: {test_acc_clean:.2f}%)')

# 5. Per-class accuracy bar chart
ax5 = plt.subplot(2, 3, 5)
class_accuracies = []
for i in range(num_classes):
    if cm[i].sum() > 0:
        acc = cm[i, i] / cm[i].sum() * 100
    else:
        acc = 0
    class_accuracies.append(acc)

bars = ax5.bar(class_names, class_accuracies, color='skyblue', edgecolor='navy')
ax5.set_ylabel('Accuracy (%)')
ax5.set_title('Per-Class Accuracy')
ax5.set_ylim(0, 105)
# Add value labels on bars
for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 6. Model info text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
info_text = f"""
Model Information:
==================
Architecture: Mini-Xception
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}

Training Info:
==================
Dataset Size: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test
Batch Size: {batch_size}
Best Val Accuracy: {best_val_acc:.2f}%
Final Epoch: {len(train_losses)}

Test Results:
==================
Clean Accuracy: {test_acc_clean:.2f}%
Noise (σ=0.1): {noise_accuracies[2]:.2f}%
Noise (σ=0.2): {noise_accuracies[4]:.2f}%
Robustness Drop: {test_acc_clean - noise_accuracies[4]:.2f}%
"""
ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, 
        fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.suptitle(f'Mini-Xception Comprehensive Results - {timestamp}', fontsize=16)
plt.tight_layout()
plt.savefig(plots_dir / 'comprehensive_results.png', dpi=100, bbox_inches='tight')
plt.show()

# SECTION 11: SAVE ALL RESULTS
# Classification report
log_message("\n" + "="*60)
log_message("CLASSIFICATION REPORT")
log_message("="*60)
report = classification_report(all_labels, all_preds, target_names=class_names)
log_message(report)

# Save comprehensive results to JSON
results = {
    'model': 'Mini-Xception',
    'timestamp': timestamp,
    'dataset': {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'num_classes': num_classes,
        'class_names': class_names,
        'class_distribution': dict(class_counts)
    },
    'model_info': {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    },
    'training': {
        'num_epochs': len(train_losses),
        'batch_size': batch_size,
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accs[-1] if train_accs else 0,
        'final_val_accuracy': val_accs[-1] if val_accs else 0
    },
    'test_results': {
        'clean_accuracy': test_acc_clean,
        'noise_robustness': {
            f'noise_std_{std}': acc 
            for std, acc in zip(noise_levels, noise_accuracies)
        }
    },
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(all_labels, all_preds, 
                                                   target_names=class_names, 
                                                   output_dict=True),
    'training_history': log_data
}

# Save JSON results
with open(results_dir / 'complete_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save training history separately
np.savez(results_dir / 'training_history.npz',
         train_losses=train_losses,
         val_losses=val_losses,
         train_accs=train_accs,
         val_accs=val_accs,
         noise_levels=noise_levels,
         noise_accuracies=noise_accuracies)

log_message("\n" + "="*60)
log_message("TRAINING COMPLETE!")
log_message(f"All results saved to: {output_dir}")
log_message("="*60)

print(f"""
Summary:
--------
✓ Model saved to: {models_dir}
✓ Plots saved to: {plots_dir}
✓ Logs saved to: {logs_dir}
✓ Results saved to: {results_dir}

Key Results:
- Clean Test Accuracy: {test_acc_clean:.2f}%
- Best Validation Accuracy: {best_val_acc:.2f}%
- Robustness (σ=0.1): {noise_accuracies[2]:.2f}%
- Robustness (σ=0.2): {noise_accuracies[4]:.2f}%

Experiment Notes:
- Xception_{timestamp}: Added Gaussian noise testing
- Label smoothing: 0.1
- Dropout rate: 0.4
- Optimizer: AdamW (lr=0.0001, weight_decay=0.01)
- Scheduler: CosineAnnealingWarmRestarts
""")