# ==========================================================
# UAV Defect Detection - Faster R-CNN + Custom Backbone
# ==========================================================

import os
import cv2
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign, DeformConv2d
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
from typing import List, Dict, Any, Tuple
from PIL import Image

# ==========================================================
# CONFIG
# ==========================================================
class Config:
    Dataset_src = r"D:\Himanshu_ML\TE_UAV Data\TE_defected\cat_defect1_data"

    TRAIN_IMAGE_DIR = os.path.join(Dataset_src, "train")
    TRAIN_ANNOTATION_PATH = os.path.join(TRAIN_IMAGE_DIR, "_annotations.coco.json")

    VAL_IMAGE_DIR = os.path.join(Dataset_src, "valid")
    VAL_ANNOTATION_PATH = os.path.join(VAL_IMAGE_DIR, "_annotations.coco.json")

    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    LR = 1e-4
    WD = 1e-4

    FG_IOU = 0.5
    BG_IOU = 0.3
    BATCH_PER_IMG = 256
    POSITIVE_FRACTION = 0.5
    PRE_NMS_TRAIN = 2000
    PRE_NMS_TEST = 1000
    POST_NMS_TRAIN = 2000
    POST_NMS_TEST = 300
    NMS_THRESH = 0.7

# ==========================================================
# CLASSES
# ==========================================================
CLASS_NAMES = [
    "background",   # class 0
    "dent",         # class 1
    "crack",        # class 2
    "paint-off",    # class 3
    "scratch",      # class 4
    "missing-head", # class 5
    "NA"            # class 6
]

# ==========================================================
# PREPROCESSING
# ==========================================================
class Preprocessing:
    def __init__(self, use_clahe=True, use_flip=True, use_brightness=True):
        self.use_clahe = use_clahe
        self.use_flip = use_flip
        self.use_brightness = use_brightness

    def __call__(self, img: np.ndarray) -> np.ndarray:
        processed_img = img.copy()

        # CLAHE
        if self.use_clahe:
            lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Random flip
        if self.use_flip and random.random() > 0.5:
            processed_img = cv2.flip(processed_img, 1)

        # Random brightness
        if self.use_brightness:
            factor = 1.0 + (random.random() - 0.5) * 0.2
            processed_img = np.clip(processed_img * factor, 0, 255).astype(np.uint8)

        return processed_img

# ==========================================================
# ATTENTION MODULES (CBAM)
# ==========================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(x))

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_, _ = x.max(dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ==========================================================
# BACKBONE WITH FPN + CBAM
# ==========================================================
class CustomBackboneWithFPN(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.stage1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stage2 = base.layer1
        self.stage3 = base.layer2
        self.stage4 = base.layer3
        self.stage5 = base.layer4

        self.l5 = nn.Conv2d(2048, 256, 1)
        self.l4 = nn.Conv2d(1024, 256, 1)
        self.l3 = nn.Conv2d(512, 256, 1)
        self.l2 = nn.Conv2d(256, 256, 1)

        self.s4 = nn.Conv2d(256, 256, 3, padding=1)
        self.s3 = nn.Conv2d(256, 256, 3, padding=1)
        self.s2 = nn.Conv2d(256, 256, 3, padding=1)

        self.deform = DeformConv2d(256, 256, 3, padding=1)
        self.cbam = CBAM(256)

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode='nearest') + y

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        p5 = self.l5(c5)
        p4 = self._upsample_add(p5, self.l4(c4))
        p3 = self._upsample_add(p4, self.l3(c3))
        p2 = self._upsample_add(p3, self.l2(c2))

        offset = torch.zeros(p5.shape[0], 2*3*3, p5.shape[2], p5.shape[3], device=p5.device)
        p5 = self.cbam(self.deform(p5, offset))
        p4, p3, p2 = self.s4(p4), self.s3(p3), self.s2(p2)

        return OrderedDict({"0": p2, "1": p3, "2": p4, "3": p5})

# ==========================================================
# COCO DATA LOADER
# ==========================================================
def load_coco_data(json_path: str, image_dir: str) -> Tuple[List[Dict[str, Any]], int]:
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = {'boxes': [], 'labels': []}

        x, y, w, h = ann['bbox']
        annotations_by_image[image_id]['boxes'].append([x, y, x+w, y+h])
        annotations_by_image[image_id]['labels'].append(ann['category_id'])

    dataset_list = []
    for image_id, anns in annotations_by_image.items():
        image_path = os.path.join(image_dir, id_to_filename[image_id])
        dataset_list.append({
            'image_path': image_path,
            'boxes': torch.tensor(anns['boxes'], dtype=torch.float32),
            'labels': torch.tensor(anns['labels'], dtype=torch.int64)
        })

    num_classes = len(CLASS_NAMES)  # fixed
    return dataset_list, num_classes

# ==========================================================
# DATASET
# ==========================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class DroneDefectDataset(Dataset):
    def __init__(self, data_list, preprocess=None):
        self.data_list = data_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img = cv2.imread(item['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.preprocess:
            img = self.preprocess(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        img_tensor = TF.to_tensor(img)
        img_tensor = normalize(img_tensor)

        target = {
            'boxes': item['boxes'],
            'labels': item['labels']
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================================
# MODEL
# ==========================================================
def build_model(num_classes):
    backbone = CustomBackboneWithFPN()
    backbone.out_channels = 256

    anchor_gen = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    rpn_head = RPNHead(256, anchor_gen.num_anchors_per_location()[0])

    rpn = RegionProposalNetwork(
        anchor_generator=anchor_gen,
        head=rpn_head,
        fg_iou_thresh=Config.FG_IOU,
        bg_iou_thresh=Config.BG_IOU,
        batch_size_per_image=Config.BATCH_PER_IMG,
        positive_fraction=Config.POSITIVE_FRACTION,
        pre_nms_top_n={"training": Config.PRE_NMS_TRAIN, "testing": Config.PRE_NMS_TEST},
        post_nms_top_n={"training": Config.POST_NMS_TRAIN, "testing": Config.POST_NMS_TEST},
        nms_thresh=Config.NMS_THRESH
    )

    roi_pool = MultiScaleRoIAlign(featmap_names=["0","1","2","3"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pool,
        image_mean=[0.485,0.456,0.406],
        image_std=[0.229,0.224,0.225]
    )
    model.rpn = rpn
    return model

# ==========================================================
# TRAIN LOOP
# ==========================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0
    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/len(loader)

# ==========================================================
# MAIN TRAIN
# ==========================================================
if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_data_list, num_classes = load_coco_data(cfg.TRAIN_ANNOTATION_PATH, cfg.TRAIN_IMAGE_DIR)
    val_data_list, _ = load_coco_data(cfg.VAL_ANNOTATION_PATH, cfg.VAL_IMAGE_DIR)

    prep = Preprocessing()
    train_ds = DroneDefectDataset(train_data_list, preprocess=prep)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # build model
    model = build_model(num_classes)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)

    # training
    for epoch in range(cfg.NUM_EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Loss: {loss:.4f}")

    torch.save(model.state_dict(), "part_drone_detection.pth")
    print("✅ Model saved as 'part_drone_detection.pth'")

    # ======================================================
    # TESTING (single image)
    # ======================================================
    model.eval()
    test_img_path = r"C:\Users\TE\Pictures\Screenshots\Screenshot 2025-09-12 124521.png"
    image = Image.open(test_img_path).convert("RGB")
    img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]["boxes"].cpu()
    labels = outputs[0]["labels"].cpu()
    scores = outputs[0]["scores"].cpu()

    fig, ax = plt.subplots(1, figsize=(12,8))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.3:  # adjust threshold
            x1,y1,x2,y2 = box
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-10, f"{CLASS_NAMES[label]} {score:.2f}",
                    color="red", fontsize=12, bbox=dict(facecolor="yellow", alpha=0.3))

    plt.show()
