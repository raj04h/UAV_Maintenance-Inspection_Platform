import os
import yaml
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# === USER INPUT ===
source_dir = Path(r"C:\Users\TE\Downloads\project-1-at-2025-09-17-15-52-51e88474")
output_classification_dir = Path(r"D:\Himanshu_ML\classified_test_images")

# Define source folders
images_source_dir = source_dir / "images"
labels_source_dir = source_dir / "labels"
notes_file_path = source_dir / "notes.json"

# YOLO dataset structure
yolo_dataset_dir = source_dir
images_train_dir = yolo_dataset_dir / "images" / "train"
images_valid_dir = yolo_dataset_dir / "images" / "valid"
labels_train_dir = yolo_dataset_dir / "labels" / "train"
labels_valid_dir = yolo_dataset_dir / "labels" / "valid"

# === 1. Create YOLO folder structure ===
for path in [images_train_dir, images_valid_dir, labels_train_dir, labels_valid_dir]:
    path.mkdir(parents=True, exist_ok=True)

# === 2. Load class names from notes.json ===
class_names = []
if notes_file_path.exists():
    with open(notes_file_path, "r") as f:
        data = json.load(f)
        for category in data.get("categories", []):
            class_names.append(category["name"])
    print(f"✅ Found classes: {class_names}")
else:
    raise FileNotFoundError(f"notes.json not found at {notes_file_path}")

# === 3. Get image/label pairs ===
image_files = [f.stem for ext in ("*.jpg", "*.jpeg", "*.png") for f in images_source_dir.glob(ext)]
label_files = [f.stem for f in labels_source_dir.glob("*.txt")]
valid_pairs = list(set(image_files) & set(label_files))

if not valid_pairs:
    raise RuntimeError("❌ No matching image and label files found.")

random.shuffle(valid_pairs)
print(f"📊 Found {len(valid_pairs)} valid image-label pairs.")

# === 4. Train/valid split ===
split_ratio = 0.8
split_idx = int(len(valid_pairs) * split_ratio)
train_split = valid_pairs[:split_idx]
valid_split = valid_pairs[split_idx:]
print(f"🔀 Split: {len(train_split)} training | {len(valid_split)} validation")

# === 5. Copy files to YOLO dirs ===
for file_name in train_split:
    image_path = list(images_source_dir.glob(f"{file_name}.*"))[0]
    shutil.copy(image_path, images_train_dir / image_path.name)
    shutil.copy(labels_source_dir / f"{file_name}.txt", labels_train_dir / f"{file_name}.txt")

for file_name in valid_split:
    image_path = list(images_source_dir.glob(f"{file_name}.*"))[0]
    shutil.copy(image_path, images_valid_dir / image_path.name)
    shutil.copy(labels_source_dir / f"{file_name}.txt", labels_valid_dir / f"{file_name}.txt")

# === 6. Create dataset.yaml ===
yaml_content = {
    "path": str(yolo_dataset_dir.absolute()),
    "train": "images/train",
    "val": "images/valid",
    "nc": len(class_names),
    "names": class_names
}
with open(yolo_dataset_dir / "dataset.yaml", "w") as f:
    yaml.dump(yaml_content, f)

print(f"\n✅ YOLO dataset.yaml created at: {yolo_dataset_dir / 'dataset.yaml'}")

# === 7. OPTIONAL: Classification buckets + stats ===
for cls in class_names:
    os.makedirs(output_classification_dir / cls, exist_ok=True)

occurrences = defaultdict(int)
images_with = defaultdict(int)

for label_file in labels_source_dir.glob("*.txt"):
    image_stem = label_file.stem
    # find matching image
    image_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        img = images_source_dir / f"{image_stem}{ext}"
        if img.exists():
            image_path = img
            break
    if image_path is None:
        continue

    class_ids_in_image = set()
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id >= len(class_names):
                continue
            occurrences[cls_id] += 1
            class_ids_in_image.add(cls_id)

    for cls_id in class_ids_in_image:
        images_with[cls_id] += 1
        dest_path = output_classification_dir / class_names[cls_id] / image_path.name
        shutil.copy(image_path, dest_path)

print("\n📊 Class stats:")
print("Class          | #images | #instances")
print("--------------------------------------")
for cls_id, name in enumerate(class_names):
    print(f"{name:13} | {images_with[cls_id]:7d} | {occurrences[cls_id]:10d}")

print(f"\n,Classification dataset created at: {output_classification_dir}")
