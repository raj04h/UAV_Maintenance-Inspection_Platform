import json
import cv2
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Load COCO JSON ---
img_dir   = r"D:\Himanshu_ML\TE_UAV Data\TE_defected\cat_defect1_data\train" # folder with images
coco_file = img_dir+"\_annotations.coco.json"

with open(coco_file, "r") as f:
    coco = json.load(f)

# Maps
id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

# --- Count images per class ---
class_to_images = defaultdict(set)
for ann in coco["annotations"]:
    cat_name = id_to_name[ann["category_id"]]
    class_to_images[cat_name].add(ann["image_id"])

image_counts={}
for cls, imgs in class_to_images.items():
    image_counts[cls]=len(imgs)

print("Image counts per class:")
for cls, count in image_counts.items():
    print(f"{cls}: {count} images")

# --- Pick a random annotation (or filter by class if you want) ---
ann = random.choice(coco["annotations"])

# Get info
cat_name = id_to_name[ann["category_id"]]
fname = id_to_filename[ann["image_id"]]
x, y, w, h = ann["bbox"]  # COCO format = [x, y, width, height]

# Load image
img_path = f"{img_dir}/{fname}"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Draw box + label
cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
cv2.putText(img, cat_name, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (255,0,0), 2)

# Show
plt.imshow(img)
plt.axis("off")
plt.show()
