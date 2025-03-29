import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, root="data/train", annotation_file="data/train.json", transform=None):
        """
        COCO 格式的數據集讀取
        :param root: 圖片存放的資料夾
        :param annotation_file: 標註文件 (COCO JSON)
        :param transform: 圖片轉換
        """
        self.root = root
        self.transform = transform

        # 讀取 JSON 文件
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # 建立 image_id 到圖片文件的對應關係
        self.image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

        # 建立 image_id 到標註數據的映射
        self.image_id_to_annotations = {img_id: [] for img_id in self.image_id_to_file.keys()}
        for ann in coco_data["annotations"]:
            self.image_id_to_annotations[ann["image_id"]].append(ann)

        # 取得所有圖片 ID
        self.image_ids = list(self.image_id_to_file.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        返回 (image, target)，其中 target 包含：
        - boxes: bounding boxes (N, 4)
        - labels: 類別標籤 (N,)
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root, self.image_id_to_file[image_id])
        
        # 讀取圖片
        image = Image.open(image_path).convert("RGB")

        # 讀取標註
        annotations = self.image_id_to_annotations[image_id]
        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]  # COCO 格式 bbox = [x_min, y_min, width, height]
            boxes.append([x, y, x + w, y + h])  # 轉換為 [x1, y1, x2, y2]
            labels.append(ann["category_id"])  # 類別 ID

        # 轉換為 PyTorch Tensor
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id])
        }

        # 轉換圖片格式
        if self.transform:
            image = self.transform(image)

        return image, target

class TestDataset(Dataset):
    def __init__(self, root="data/test", transform=None):
        """
        測試數據集
        :param root: 測試圖片存放的資料夾
        :param transform: 圖片轉換
        """
        self.root = root
        self.transform = transform

        # 取得所有圖片文件
        self.image_files = sorted(os.listdir(root))  # 確保順序一致

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        返回 (image, image_id)，不含標註
        """
        image_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        # 圖片轉換
        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]
