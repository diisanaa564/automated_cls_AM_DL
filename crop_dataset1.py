from pathlib import Path
from PIL import Image
import random
import shutil
import warnings
from collections import defaultdict
import numpy as np

warnings.simplefilter("ignore", Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# ========= CONFIGURATION =========
# Input/Output paths
IN_DIR  = Path('/u/a/dn564/Documents/sampledv9/')
OUT_DIR = Path('/u/a/dn564/Documents/sampledv9_cropped/')
EXTS = (".jpg", ".jpeg", ".png")

# BALANCE AND SPLIT SETTINGS
TOTAL_PER_CLASS = 500  # Total samples per class (will be split 70/20/10)

# Calculate split sizes --> target
TRAIN_SIZE = int(TOTAL_PER_CLASS * 0.7)  # 350 samples
VAL_SIZE = int(TOTAL_PER_CLASS * 0.2)    # 100 samples  
TEST_SIZE = int(TOTAL_PER_CLASS * 0.1)   # 50 samples

print(f"Target distribution per class:")
print(f"  Train: {TRAIN_SIZE} (70%)")
print(f"  Val:   {VAL_SIZE} (20%)")
print(f"  Test:  {TEST_SIZE} (10%)")
print(f"  Total: {TOTAL_PER_CLASS}")

# ========= OPTIMAL SIZE CONSTRAINTS FOR ALL MODELS =========
MIN_CROP_SIZE = 224   # ResNet native size
MAX_CROP_SIZE = 640   # YOLOv8 native size

# For no_defect crops: generate at strategic sizes
NODEFECT_SIZES = [
    (224, 224),  # for ResNet
    (299, 299),  # for Xception
    (320, 320),  # Good balance
    (416, 416),  # YOLOv8 small
    (512, 512),  # Good context
]
NODEFECT_CROPS_PER_SIZE = 3  
# Padding for context around defects
PADDING = 0.15  # 15% padding

# YOLO class mapping
ID2NAME = {
    0: "crack",
    1: "over_extrusion", 
    2: "under_extrusion",
    3: "warping",
}

NODEFECT_NAME = "no_defect"
ALL_CLASSES = list(ID2NAME.values()) + [NODEFECT_NAME]

# =========================
def validate_and_adjust_crop(x1, y1, x2, y2, W, H):
    """Ensure crop meets size requirements"""
    w = x2 - x1
    h = y2 - y1
    
    # Enforce minimum size
    if w < MIN_CROP_SIZE:
        center_x = (x1 + x2) // 2
        x1 = max(0, center_x - MIN_CROP_SIZE // 2)
        x2 = min(W, x1 + MIN_CROP_SIZE)
        if x2 - x1 < MIN_CROP_SIZE:
            x1 = max(0, x2 - MIN_CROP_SIZE)
    
    if h < MIN_CROP_SIZE:
        center_y = (y1 + y2) // 2
        y1 = max(0, center_y - MIN_CROP_SIZE // 2)
        y2 = min(H, y1 + MIN_CROP_SIZE)
        if y2 - y1 < MIN_CROP_SIZE:
            y1 = max(0, y2 - MIN_CROP_SIZE)
    
    # Cap maximum size
    if w > MAX_CROP_SIZE:
        center_x = (x1 + x2) // 2
        x1 = center_x - MAX_CROP_SIZE // 2
        x2 = x1 + MAX_CROP_SIZE
    
    if h > MAX_CROP_SIZE:
        center_y = (y1 + y2) // 2
        y1 = center_y - MAX_CROP_SIZE // 2
        y2 = y1 + MAX_CROP_SIZE
    
    # Final bounds check
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2))
    y2 = min(H, int(y2))
    
    return x1, y1, x2, y2

def yolo_to_xyxy(xc, yc, w, h, W, H, pad=0.0):
    """Convert YOLO format to pixel coordinates with padding"""
    bw, bh = w * W, h * H
    
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    
    # Add padding
    x1 -= bw * pad
    y1 -= bh * pad
    x2 += bw * pad
    y2 += bh * pad
    
    # Apply size constraints
    x1, y1, x2, y2 = validate_and_adjust_crop(x1, y1, x2, y2, W, H)
    
    return (x1, y1, x2, y2)

def crop_and_save(img, xyxy, filepath):
    """Crop and save image region"""
    x1, y1, x2, y2 = xyxy
    filepath.parent.mkdir(parents=True, exist_ok=True)
    img.crop((x1, y1, x2, y2)).save(filepath)

# Create output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print(" PROPERLY BALANCED DATASET WITH 70/20/10 SPLIT")
print("="*70)
print(f" Total per class: {TOTAL_PER_CLASS}")
print(f" Train: {TRAIN_SIZE} | Val: {VAL_SIZE} | Test: {TEST_SIZE}")
print(f" Crop sizes: {MIN_CROP_SIZE}×{MIN_CROP_SIZE} to {MAX_CROP_SIZE}×{MAX_CROP_SIZE}")
print("="*70)

# ============ STEP 1: COLLECT ALL CROPS FROM ALL IMAGES ============
print("\n[Step 1] Collecting all possible crops from entire dataset...")

# Store ALL crops by class (ignore original train/val/test folders)
all_crops_by_class = defaultdict(list)

# Gather all images
images = [p for p in IN_DIR.rglob("*") if p.suffix.lower() in EXTS]
print(f"Found {len(images)} total images")

for img_idx, img_path in enumerate(images):
    if (img_idx + 1) % 200 == 0:
        print(f"  Processing image {img_idx + 1}/{len(images)}")
    
    # Get class name from folder structure
    rel_parent = img_path.parent.relative_to(IN_DIR)
    parts = rel_parent.parts
    
    # Skip the split folder name, just get the class
    if len(parts) >= 2:
        class_name_in = parts[1]
    elif len(parts) == 1:
        class_name_in = parts[0]
    else:
        continue
    
    # Check for label file
    lbl_path = img_path.with_suffix(".txt")
    
    # Check if it's a no_defect image
    is_no_defect = (class_name_in == NODEFECT_NAME or 
                   (lbl_path.exists() and lbl_path.stat().st_size == 0))
    
    try:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
    except Exception as e:
        print(f"  Error opening {img_path}: {e}")
        continue
    
    if is_no_defect:
        # Generate multiple crops for no_defect
        for crop_w, crop_h in NODEFECT_SIZES:
            for attempt in range(NODEFECT_CROPS_PER_SIZE):
                if W >= crop_w and H >= crop_h:
                    x1 = random.randint(0, W - crop_w)
                    y1 = random.randint(0, H - crop_h)
                    x2 = x1 + crop_w
                    y2 = y1 + crop_h
                    
                    crop_info = {
                        'img_path': img_path,
                        'bbox': (x1, y1, x2, y2),
                        'name': f"{img_path.stem}_nodef_{crop_w}_{attempt}.jpg"
                    }
                    all_crops_by_class[NODEFECT_NAME].append(crop_info)
    
    elif lbl_path.exists():
        # Process defect images
        rows = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                
                parts = ln.split()
                if len(parts) < 5:
                    continue
                
                try:
                    cid = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        continue
                    
                    if cid not in ID2NAME:
                        continue
                    
                    rows.append((cid, xc, yc, w, h))
                except ValueError:
                    continue
        
        # Extract each defect
        for i, (cid, xc, yc, w, h) in enumerate(rows):
            xyxy = yolo_to_xyxy(xc, yc, w, h, W, H, PADDING)
            
            # Validate size
            crop_w = xyxy[2] - xyxy[0]
            crop_h = xyxy[3] - xyxy[1]
            
            if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                x1, y1, x2, y2 = validate_and_adjust_crop(*xyxy, W, H)
                xyxy = (x1, y1, x2, y2)
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                    continue  # Skip if still too small
            
            class_name = ID2NAME[cid]
            
            crop_info = {
                'img_path': img_path,
                'bbox': xyxy,
                'name': f"{img_path.stem}_obj{i}.jpg"
            }
            all_crops_by_class[class_name].append(crop_info)

# ============ STEP 2: SHOW AVAILABLE CROPS ============
print("\n[Step 2] Available crops per class (before balancing):")
print("-" * 50)

min_available = float('inf')
for class_name in ALL_CLASSES:
    count = len(all_crops_by_class[class_name])
    print(f"  {class_name:20s}: {count:6d} crops")
    if count < min_available:
        min_available = count

print(f"\n  Minimum available: {min_available} crops")
print(f"  Target needed: {TOTAL_PER_CLASS} crops per class")

if min_available < TOTAL_PER_CLASS:
    print(f"\n  ⚠️  Warning: Not enough samples!")
    print(f"  Adjusting target to {min_available} per class")
    TOTAL_PER_CLASS = min_available
    TRAIN_SIZE = int(TOTAL_PER_CLASS * 0.7)
    VAL_SIZE = int(TOTAL_PER_CLASS * 0.2)
    TEST_SIZE = TOTAL_PER_CLASS - TRAIN_SIZE - VAL_SIZE  # Ensure we use all samples

# ============ STEP 3: BALANCE AND SPLIT ============
print("\n[Step 3] Balancing classes and splitting into train/val/test...")
print("-" * 50)

final_splits = {
    'train': defaultdict(list),
    'val': defaultdict(list),
    'test': defaultdict(list)
}

for class_name in ALL_CLASSES:
    available_crops = all_crops_by_class[class_name]
    
    # Randomly sample exactly TOTAL_PER_CLASS crops
    if len(available_crops) >= TOTAL_PER_CLASS:
        selected_crops = random.sample(available_crops, TOTAL_PER_CLASS)
    else:
        selected_crops = available_crops
        print(f"  Warning: {class_name} only has {len(available_crops)} crops")
    
    # Shuffle the selected crops
    random.shuffle(selected_crops)
    
    # Split into train/val/test
    train_crops = selected_crops[:TRAIN_SIZE]
    val_crops = selected_crops[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_crops = selected_crops[TRAIN_SIZE + VAL_SIZE:]
    
    final_splits['train'][class_name] = train_crops
    final_splits['val'][class_name] = val_crops
    final_splits['test'][class_name] = test_crops
    
    print(f"  {class_name:20s}: {len(train_crops)} train, {len(val_crops)} val, {len(test_crops)} test")

# ============ STEP 4: SAVE THE BALANCED AND SPLIT DATASET ============
print("\n[Step 4] Saving final dataset...")
print("-" * 50)

total_saved = 0
save_errors = 0

for split_name in ['train', 'val', 'test']:
    for class_name in ALL_CLASSES:
        crops = final_splits[split_name][class_name]
        
        if not crops:
            continue
        
        # Create output folder
        # Change 'val' to 'valid' if your structure uses 'valid'
        output_split = 'valid' if split_name == 'val' else split_name
        out_folder = OUT_DIR / output_split / class_name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"  Saving {output_split}/{class_name}: {len(crops)} images", end="")
        
        for idx, crop_info in enumerate(crops):
            if (idx + 1) % 50 == 0:
                print(".", end="", flush=True)
            
            try:
                # Load image
                img = Image.open(crop_info['img_path']).convert("RGB")
                
                # Crop and save
                out_path = out_folder / f"{split_name}_{idx:04d}_{crop_info['name']}"
                crop_and_save(img, crop_info['bbox'], out_path)
                total_saved += 1
                
            except Exception as e:
                save_errors += 1
                if save_errors <= 5:
                    print(f"\n    Error: {e}")
        print()  # New line after dots

print(f"\n  Total images saved: {total_saved}")
if save_errors > 0:
    print(f"  Errors encountered: {save_errors}")

# ============ FINAL VERIFICATION ============
print("\n" + "="*70)
print(" FINAL DATASET VERIFICATION")
print("="*70)

for split_dir in sorted(OUT_DIR.iterdir()):
    if split_dir.is_dir():
        print(f"\n  {split_dir.name.upper()}:")
        total_split = 0
        
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                print(f"    {class_dir.name:20s}: {count:4d} images")
                total_split += count
        
        print(f"    {'TOTAL':20s}: {total_split:4d} images")

# Calculate and show final statistics
print("\n" + "="*70)
print(" SUMMARY")
print("="*70)

total_images = sum(
    len(list(OUT_DIR.rglob(f"*/{cls}/*.jpg"))) + 
    len(list(OUT_DIR.rglob(f"*/{cls}/*.png")))
    for cls in ALL_CLASSES
)

print(f"\n  Total images in dataset: {total_images}")
print(f"  Classes: {len(ALL_CLASSES)}")
print(f"  Images per class: {TOTAL_PER_CLASS}")
print(f"  Split ratio: 70% / 20% / 10%")

# Verify balance
print("\n  Class balance check:")
is_balanced = True
for class_name in ALL_CLASSES:
    train_count = len(final_splits['train'][class_name])
    val_count = len(final_splits['val'][class_name])
    test_count = len(final_splits['test'][class_name])
    total = train_count + val_count + test_count
    
    if total != TOTAL_PER_CLASS:
        is_balanced = False
        print(f"     {class_name}: {total} (expected {TOTAL_PER_CLASS})")
    else:
        print(f"{class_name}: {total}")

if is_balanced:
    print("\n   Dataset is perfectly balanced!")
else:
    print("\n    Dataset has imbalances - review needed")

print("\n" + "="*70)
print(f" ✓ Dataset saved to: {OUT_DIR}")
print("="*70)