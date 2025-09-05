import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import AutoAugment, AutoAugmentPolicy #add
from torchvision import datasets, transforms, models
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

# SECTION 1: SETUP AND FILE MANAGEMENT
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CREATE ORGANIZED OUTPUT DIRECTORY
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f'/data/dn564/experiments/resnet34_run_{timestamp}')
output_dir.mkdir(parents=True, exist_ok=True)

# Sub-directories for different outputs
models_dir = output_dir / 'models'
plots_dir = output_dir / 'plots'
logs_dir = output_dir / 'logs'
results_dir = output_dir / 'results'

for dir_path in [models_dir, plots_dir, logs_dir, results_dir]:
    dir_path.mkdir(exist_ok=True)

print(f"All outputs will be saved to: {output_dir}")

# Dataset path
data_dir = '/data/dn564/sampledv11_cropped'

# SECTION 2: DATA AUGMENTATION SETUP
class GaussianNoise(nn.Module):

    def __init__(self, mean=0.0, std=0.1, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p  # Probability of applying noise
    
    def forward(self, img):
        if self.training and random.random() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            return torch.clamp(img + noise, 0, 1)
        return img

# Training transforms with augmentation
'''
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Slightly larger for random crop
    transforms.RandomCrop(224),     # ResNet34 expects 224x224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # Less common for defects
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))  # Simulates occlusions
])'''
#exp5
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),  # Automatic augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.3))
])
#exp5-end
# Validation/test transforms (no augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# SECTION 3: DATASET LOADING AND CLASS BALANCING

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)

# Analyze class distribution
train_targets = [sample[1] for sample in train_dataset.samples]
class_counts = Counter(train_targets)
class_names = ['crack', 'no_defect', 'over_extrusion', 'under_extrusion', 'warping']
num_classes = len(class_names)
#new
print(train_dataset.classes)        # order used by ImageFolder
print(train_dataset.class_to_idx)   # mapping name -> index

###start
# === (2) Count samples per class in the *train* split ===
from collections import Counter
import torch, torch.nn as nn

targets = getattr(train_dataset, "targets", [y for _, y in train_dataset.samples])
counts = Counter(targets)
num_classes = len(train_dataset.classes)
counts_t = torch.tensor([counts[i] for i in range(num_classes)], dtype=torch.float32)
print("Train counts per class (idx order):", counts_t.tolist())

# === (3) Build class weights (inverse-frequency) and optionally boost under/over ===
class_weights = counts_t.sum() / (num_classes * counts_t)  # ~1 / count
class_weights = class_weights / class_weights.mean()       # normalize around 1.0

UNDER = train_dataset.class_to_idx.get("under_extrusion")
OVER  = train_dataset.class_to_idx.get("over_extrusion")
if UNDER is not None: class_weights[UNDER] *= 1.5   # tweak 1.3–1.8 as needed
if OVER  is not None: class_weights[OVER]  *= 1.2   # tweak 1.1–1.4 as needed

print("Class weights (idx order):", [round(float(w), 4) for w in class_weights])

# === (4) Plug into loss (with your label smoothing) ===
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),
    label_smoothing=0.05
)
###end
print(f"\nDataset Statistics:")
print(f"Classes: {class_names}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Class distribution in training:")
for idx, name in enumerate(class_names):
    count = class_counts[idx]
    percentage = (count / len(train_dataset)) * 100
    print(f"  {name}: {count} samples ({percentage:.1f}%)")

# Calculate weights for balanced sampling
total_samples = len(train_targets)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
sample_weights = [class_weights[target] for target in train_targets]
sampler = WeightedRandomSampler(weights=sample_weights, 
                               num_samples=len(sample_weights), 
                               replacement=True)

# Create data loaders
batch_size = 32  # Good balance between speed and memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                         sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                       shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                        shuffle=False, num_workers=4, pin_memory=True)


# SECTION 4: MODEL ARCHITECTURE -
class ResNet34WithDropout(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        
        # Load pretrained ResNet34 (new PyTorch syntax)
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Freeze early layers initially (transfer learning best practice)
        # Early layers detect edges/textures - already learned from ImageNet
        for name, param in self.resnet.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
        
        # Replace classifier with custom head
        num_features = self.resnet.fc.in_features  # 512 for ResNet34
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.7),  # Less dropout in deeper layer
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze_layers(self, layer_names):
        """Progressive unfreezing for fine-tuning"""
        for name, param in self.resnet.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True

# Initialize model
model = ResNet34WithDropout(num_classes=num_classes, dropout_rate=0.5).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Statistics:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {(trainable_params/total_params)*100:.1f}%")


# SECTION 5: LOSS FUNCTION AND OPTIMIZATION
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05): #was 0.1
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        # Apply smoothing
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        # Calculate loss
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

criterion = LabelSmoothingCrossEntropy(smoothing=0.05) #was 0.1

# Training configuration
num_epochs = 100
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0
#early_stopping = EarlyStopping(patience=20)

#---exp3
#optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
#                        lr=0.0002,  # 2x LR for 2x batch size
#                        weight_decay=0.01)
#exp3-end

#------baseline
#Optimizer with weight decay (L2 regularization)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=0.0001, weight_decay=0.01) #baseline


# Learning rate scheduler (cosine annealing with warm restarts)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, #baseline
                                                           T_0=10,  # Restart every 10 epochs
                                                           T_mult=2)  # Double period after restart

#------baseline-end

#exp2
'''
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                     lr=0.01,  # Starting LR (will be divided by div_factor)
                     momentum=0.9,  # Helps overcome local minima
                     weight_decay=0.01,
                     nesterov=True)  # Look-ahead momentum

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,  # Peak learning rate
    epochs= num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% time warming up
    anneal_strategy='cos',
    div_factor=25,  # Start LR = max_lr/25 = 0.004
    final_div_factor=1000  # End LR = max_lr/1000 = 0.0001
)
scheduler.step() 
#exp2 end

#exp4
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
)
#exp4-end
'''
# SECTION 6: TRAINING FUNCTIONS WITH MIXUP
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True, epoch=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progressive unfreezing schedule
    if epoch == 20:
        print("  --> Unfreezing layer2")
        model.unfreeze_layers(['layer2'])
    elif epoch == 40:
        print("  --> Unfreezing layer1")
        model.unfreeze_layers(['layer1'])
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup with 50% probability
        if use_mixup and np.random.random() > 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients
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


# SECTION 7: TRAINING LOOP WITH LOGGING
# Initialize logging
log_file = logs_dir / 'training_log.txt'
log_data = []

def log_message(message, print_msg=True):
    """Helper function to log messages to file and optionally print"""
    if print_msg:
        print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Early stopping
'''class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score 
            self.counter = 0
'''

log_message("="*60)
log_message("Starting training...")
log_message(f"Model: ResNet34")
log_message(f"Dataset: {data_dir}")
log_message(f"Output directory: {output_dir}")
log_message("="*60)
#ori

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                       optimizer, device, use_mixup=True, epoch=epoch)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    scheduler.step()
    
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
'''
#exp4
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                       optimizer, device, use_mixup=True, epoch=epoch)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    # For ReduceLROnPlateau, step AFTER validation with metric
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_acc)  # Monitor validation accuracy
    else:
        scheduler.step()  # Other schedulers
#exp4-end
'''
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
    
    # Early stopping
    #early_stopping(val_loss)
    #if early_stopping.early_stop:
       # log_message(f"Early stopping triggered at epoch {epoch+1}")
       # break

log_message(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# SECTION 8: COMPREHENSIVE TESTING WITH AND WITHOUT NOISE
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

# Test 2: Multiple noise levels
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

# SECTION 9: CONFUSION MATRIX AND VISUALIZATION
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

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Training curves(TRAIN-VAL)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(train_losses, label='Train Loss', alpha=0.8)
ax1.plot(val_losses, label='Val Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 3: Noise robustness (TEST) 
ax2 = plt.subplot(2, 3, 2)
ax2.plot(train_accs, label='Train Acc', alpha=0.8)
ax2.plot(val_accs, label='Val Acc', alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 2. Noise robustness graph
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

# 3. Confusion Matrix
ax4 = plt.subplot(2, 3, 4)
cm = confusion_matrix(all_labels, all_preds) #test data
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax4)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title(f'Confusion Matrix (Clean Test: {test_acc_clean:.2f}%)')

# 4. Per-class accuracy bar chart
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

# 5. Model info text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
info_text = f"""
Model Information:
==================
Architecture: ResNet34
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

plt.suptitle(f'ResNet34 Comprehensive Results - {timestamp}', fontsize=16)
plt.tight_layout()
plt.savefig(plots_dir / 'comprehensive_results.png', dpi=100, bbox_inches='tight')
plt.show()

# SECTION 10: SAVE ALL RESULTS
# Classification report
log_message("\n" + "="*60)
log_message("CLASSIFICATION REPORT")
log_message("="*60)
report = classification_report(all_labels, all_preds, target_names=class_names)
log_message(report)

# Save comprehensive results to JSON
results = {
    'model': 'ResNet34',
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
         val_accs=val_accs)

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
""")

#resnet182436 trial11_v01 smoothing =0.05  
#resnet184430 trial11_v02 class weight 
#2008 fix error 1420
#2008 150epochs 1433
#2008 150epoch 1528 best