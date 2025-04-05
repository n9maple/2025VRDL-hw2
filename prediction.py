import torch
import json
import random
import os
import cv2
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from Dataset import TestDataset  # 請確保你的 TestDataset 已經正確實作
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# 設定裝置
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 讀取測試數據集
test_dataset = TestDataset(transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 載入模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
num_classes = 11
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 加載訓練好的權重
model.load_state_dict(torch.load("save_model/epoch2.pth", map_location=device))  # 請替換成你的模型權重
model.to(device)
model.eval()

# 儲存預測結果的列表
predictions = []

# 創建 save_result 資料夾
os.makedirs("save_result", exist_ok=True)

# 進行推理
with torch.no_grad():
    sampled_images = random.sample(range(len(test_dataset)), 3)  # 隨機選取 3 張圖片
    for idx, (image, image_id) in enumerate(tqdm(test_loader, desc="Inference")):
        image = [img.to(device) for img in image]

        # 取得預測結果
        outputs = model(image)

        # 解析預測結果
        for output in outputs:
            boxes = output["boxes"].cpu().numpy()  # 預測的 bbox
            scores = output["scores"].cpu().numpy()  # 預測的信心分數
            labels = output["labels"].cpu().numpy()  # 類別 ID
            img_id = image_id[0]  # 單 batch 圖片的 ID

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                predictions.append({
                    "image_id": img_id,
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],  # COCO 格式的 bbox
                    "score": float(score),
                    "category_id": int(label)
                })

            # 若這張圖片被選中，則標示預測框並儲存
            if idx in sampled_images:
                img_path = f"data/test/{img_id}.png"  # 假設 test 圖片命名方式為 {image_id}.png
                img = cv2.imread(img_path)
                if img is not None:
                    for box, score, label in zip(boxes, scores, labels):
                        x_min, y_min, x_max, y_max = map(int, box)
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        text = f"{label}: {score:.2f}"
                        cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    save_path = f"save_result/{img_id}.png"
                    cv2.imwrite(save_path, img)

# 儲存為 COCO 格式 JSON
with open("pred.json", "w") as f:
    json.dump(predictions, f, indent=4)

print("Inference 完成，結果已儲存至 pred.json 和 save_result/ 資料夾內")
