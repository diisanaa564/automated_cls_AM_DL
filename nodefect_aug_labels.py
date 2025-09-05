import os
import cv2
import random
import numpy as np

# === CONFIG ===
input_dir = r"C:\datasets\no_defect\no_defect" 
output_dir = r"C:\datasets\no_defect_aug_labeled"
target_total = 900

# === Create folders ===
os.makedirs(output_dir, exist_ok=True)
label_dir = os.path.join(output_dir, "labels")
image_dir = os.path.join(output_dir, "images")
os.makedirs(label_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

def augment_image(img):
    """Apply random augmentations to image"""
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        brightness = random.randint(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
    if random.random() > 0.5:
        angle = random.randint(-8, 8)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
    if random.random() > 0.5:
        scale = random.uniform(0.95, 1.05)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        if scale > 1:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            img = img[start_y:start_y+h, start_x:start_x+w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_y, h-new_h-pad_y, 
                                     pad_x, w-new_w-pad_x, cv2.BORDER_REPLICATE)
    return img

# === Start processing ===
i = 0
original_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(original_files)} original images")

augmentations_per_image = target_total // len(original_files)
print(f"Each image will produce about {augmentations_per_image} versions")

for file in original_files:
    if i >= target_total:
        break

    img_path = os.path.join(input_dir, file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {file}")
        continue

    for aug_idx in range(augmentations_per_image):
        if i >= target_total:
            break

        img_processed = img.copy() if aug_idx == 0 else augment_image(img.copy())
        img_name = f"no_defect_{i:04d}.jpg"
        label_name = f"no_defect_{i:04d}.txt"

        cv2.imwrite(os.path.join(image_dir, img_name), img_processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(label_dir, label_name), "w") as f:
            pass  # empty label

        i += 1

print(f"\nDone! Created exactly {i} images with empty labels.")
print(f"Images saved to: {image_dir}")
print(f"Labels saved to: {label_dir}")
