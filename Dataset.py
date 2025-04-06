import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TrainDataset(Dataset):
    def __init__(
        self, root="data/train", annotation_file="data/train.json", transform=None
    ):
        self.root = root
        self.transform = transform

        # read json
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # map image_id to file name
        self.image_id_to_file = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }

        # map image_id to annotation
        self.image_id_to_annotations = {
            img_id: [] for img_id in self.image_id_to_file.keys()
        }
        for ann in coco_data["annotations"]:
            self.image_id_to_annotations[ann["image_id"]].append(ann)

        self.image_ids = list(self.image_id_to_file.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root, self.image_id_to_file[image_id])
        image = Image.open(image_path).convert("RGB")
        annotations = self.image_id_to_annotations[image_id]
        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        if self.transform:
            image = self.transform(image)

        return image, target


class TestDataset(Dataset):
    def __init__(self, root="data/test", transform=None):
        self.root = root
        self.transform = transform

        self.image_files = sorted(os.listdir(root))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, os.path.splitext(self.image_files[idx])[0]
