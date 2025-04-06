import torch
import json
import random
import os
import cv2
import argparse
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from Dataset import TestDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default="data/test", help="Path to the test images")
    parser.add_argument("--model_path", type=str, default="save_model/epoch1.pth", help="Path to the saved model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for test")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    parser.add_argument("--save_dir", type=str, default="save_result", help="The directory where the result is saved")
    args = parser.parse_args()

    # 讀取測試數據集
    test_dataset = TestDataset(root=args.test_root, transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # 載入模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 加載訓練好的權重
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))  # 請替換成你的模型權重
    model.to(args.device)
    model.eval()

    # 儲存預測結果的列表
    predictions = []

    # 創建 save_result 資料夾
    os.makedirs(args.save_dir, exist_ok=True)

    # 進行推理
    with torch.no_grad():
        sampled_images = random.sample(range(len(test_dataset)), 3)  # 隨機選取 3 張圖片
        for idx, (images, image_ids) in enumerate(tqdm(test_loader, desc="Inference")):
            images = [img.to(args.device) for img in images]

            # 取得預測結果
            outputs = model(images)

            # 解析預測結果
            for image_id, output in zip(image_ids, outputs):
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    predictions.append({
                        "image_id": int(image_id),
                        "bbox": [float(x_min), float(y_min), float(width), float(height)],  # COCO 格式的 bbox
                        "score": float(score),
                        "category_id": int(label)
                    })

                # # 若這張圖片被選中，則標示預測框並儲存
                # if idx in sampled_images:
                #     img_path = f"data/test/{image_id}.png"  # 假設 test 圖片命名方式為 {image_id}.png
                #     img = cv2.imread(img_path)
                #     if img is not None:
                #         for box, score, label in zip(boxes, scores, labels):
                #             x_min, y_min, x_max, y_max = map(int, box)
                #             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                #             text = f"{label}: {score:.2f}"
                #             cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                #         save_path = f"save_result/{image_id}.png"
                #         cv2.imwrite(save_path, img)

    # 儲存為 COCO 格式 JSON
    with open(os.path.join(args.save_dir, "pred.json"), "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Inference finished. The result is saved in {args.save_dir} folder")

if __name__ == "__main__":
    main()