import torch
import torchvision
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FastRCNNPredictor,
)
from rich import print
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)
from Dataset import TrainDataset
from torch.utils.data import Subset
from utils import COCOmap
import os


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_train_loss = 0.0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[green]Training...", total=len(train_loader))

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress.update(task, advance=1)

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss


def validate(model, valid_loader, device):
    model.eval()
    all_preds, all_gts = [], []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Validation...", total=len(valid_loader))

        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                preds = model(images)
                all_preds.extend(single_img_pred for single_img_pred in preds)
                all_gts.extend(single_img_gt for single_img_gt in targets)
                progress.update(task, advance=1)

    mAP = COCOmap(all_preds, all_gts)
    return mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tr",
        "--train_root",
        type=str,
        default="data/train",
        help="Path to training images",
    )
    parser.add_argument(
        "-ta",
        "--train_annotation",
        type=str,
        default="data/train.json",
        help="Path to COCO format training annotation file",
    )
    parser.add_argument(
        "-vr",
        "--valid_root",
        type=str,
        default="data/valid",
        help="Path to validation images",
    )
    parser.add_argument(
        "-va",
        "--valid_annotation",
        type=str,
        default="data/valid.json",
        help="Path to COCO format validation annotation file",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("-l", "--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="save_model",
        help="The directory where the weight is saved",
    )
    parser.add_argument(
        "-pt",
        "--partial_training_data",
        type=float,
        default=-1,
        help="Ratio of training data to be used (0~1); if out of range, use all data",
    )
    parser.add_argument(
        "-pv",
        "--partial_validation_data",
        type=float,
        default=-1,
        help="Ratio of validation data to be used (0~1); if out of range, use all data",
    )
    args = parser.parse_args()

    # initialize model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    )
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(args.device)

    # load dataset
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_dataset = TrainDataset(
        args.train_root,
        args.train_annotation,
        transform=trans,
    )
    valid_dataset = TrainDataset(
        args.valid_root,
        args.valid_annotation,
        transform=trans,
    )

    # use partial dataset
    if args.partial_training_data < 1 and args.partial_training_data > 0:
        subset_size = int(len(train_dataset) * args.partial_training_data)
        indices = list(range(subset_size))
        train_subset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    if args.partial_validation_data < 1 and args.partial_validation_data > 0:
        subset_size = int(len(valid_dataset) * args.partial_validation_data)
        indices = list(range(subset_size))
        valid_subset = Subset(train_dataset, indices)
        valid_loader = DataLoader(
            valid_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )
    else:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    # set optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )
    os.makedirs(args.save_dir, exist_ok=True)
    losses = []
    mAPs = []
    for epoch in range(args.epochs):
        print(f"\n[yellow2]Epoch \[{epoch+1}/{args.epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        mAP = validate(model, valid_loader, args.device)
        losses.append(train_loss)
        mAPs.append(mAP)

        print(f"Train Loss: {train_loss:.4f} | Validation mAP: {mAP:.4f}")

        # save model
        torch.save(model.state_dict(), f"{args.save_dir}/epoch{epoch+1}.pth")
        print(f"save model in [yellow2]{args.save_dir}/epoch{epoch+1}.pth[/yellow2]")

    np.save(os.path.join(args.save_dir, "loss.npy"), np.array(losses))
    np.save(os.path.join(args.save_dir, "mAP.npy"), np.array(mAPs))
    print(
        f"\n[green]save loss and mAP numpy file in [yellow2]{args.save_dir} folder[/yellow2]"
    )


if __name__ == "__main__":
    main()
