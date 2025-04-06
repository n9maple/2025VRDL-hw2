import torch
import torchvision
import argparse
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, FastRCNNPredictor
from tqdm import tqdm
from Dataset import TrainDataset
from torch.utils.data import Subset
from map import COCOmap
import os

def train_one_epoch(model, train_loader, optimizer, device):
    """ 訓練一個 epoch，並回傳訓練損失 """
    model.train()
    total_train_loss = 0.0
    train_pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, targets in train_pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)  # 計算 loss
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_pbar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss


def validate(model, valid_loader, device):
    """ 驗證模型，計算 Validation mAP """
    model.eval()
    all_preds, all_gts = [], []

    valid_pbar = tqdm(valid_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, targets in valid_pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 只使用 images 進行前向傳播，因為驗證階段不需要 loss
            preds = model(images)
            
            # 獲取預測結果
            all_preds.extend(single_img_pred for single_img_pred in preds)
            all_gts.extend(single_img_gt for single_img_gt in targets)
                

    # 計算 mAP
    mAP = COCOmap(all_preds, all_gts)

    return mAP



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="data/train", help="Path to training images")
    parser.add_argument("--train_annotation", type=str, default="data/train.json", help="Path to COCO format training annotation file")
    parser.add_argument("--valid_root", type=str, default="data/valid", help="Path to validation images")
    parser.add_argument("--valid_annotation", type=str, default="data/valid.json", help="Path to COCO format validation annotation file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    parser.add_argument("--save_dir", type=str, default="save_model", help="The directory where the weight is saved")
    args = parser.parse_args()


    # 初始化模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(args.device)

    # 加載數據集
    train_dataset = TrainDataset(args.train_root, args.train_annotation, transform=torchvision.transforms.ToTensor())
    valid_dataset = TrainDataset(args.valid_root, args.valid_annotation, transform=torchvision.transforms.ToTensor())
    #### Test Code
    # subset_size = 50
    # indices = list(range(subset_size))  # 取前 100 筆資料
    # train_subset = Subset(train_dataset, indices)
    # val_subset = Subset(valid_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: tuple(zip(*x)))

    # 設定優化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # 創建儲存模型的資料夾
    
    os.makedirs(args.save_dir, exist_ok=True)
    # 訓練循環
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        mAP = validate(model, valid_loader, args.device)

        print(f"Train Loss: {train_loss:.4f} | mAP: {mAP:.4f}")

        # 儲存模型
        torch.save(model.state_dict(), f"{args.save_dir}/epoch{epoch+1}.pth")


if __name__ == "__main__":
    main()
