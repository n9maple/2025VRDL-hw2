import torch
import json
import os
import argparse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from rich import print
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)
from Dataset import TestDataset
from number_prediction import num_pred
from utils import prediction_draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tr",
        "--test_root",
        type=str,
        default="data/test",
        help="Path to the test images",
    )
    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        default="save_model/epoch1.pth",
        help="Path to the saved model",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=4, help="Batch size for test"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for test",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        type=str,
        default="save_result",
        help="The directory where the result is saved",
    )
    parser.add_argument(
        "-st",
        "--score_thre",
        type=float,
        default=0.8,
        help="The score threshold to pick the box when predicting the whole number",
    )
    args = parser.parse_args()

    # load dataset
    test_dataset = TestDataset(
        root=args.test_root, transform=torchvision.transforms.ToTensor()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # load weight
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    predictions = []
    os.makedirs(args.save_dir, exist_ok=True)

    # inference
    with torch.no_grad():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Prediction...", total=len(test_loader))
            for images, image_ids in test_loader:
                images = [img.to(args.device) for img in images]
                outputs = model(images)
                for image_id, output in zip(image_ids, outputs):
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        x_min, y_min, x_max, y_max = box
                        width = x_max - x_min
                        height = y_max - y_min
                        predictions.append(
                            {
                                "image_id": int(image_id),
                                "bbox": [
                                    float(x_min),
                                    float(y_min),
                                    float(width),
                                    float(height),
                                ],
                                "score": float(score),
                                "category_id": int(label),
                            }
                        )
                progress.update(task, advance=1)

    # save prediction
    with open(os.path.join(args.save_dir, "pred.json"), "w") as f:
        json.dump(predictions, f, indent=4)

    num_pred(
        os.path.join(args.save_dir, "pred.json"),
        os.path.join(args.save_dir, "pred.csv"),
        args.score_thre,
    )
    prediction_draw(
        os.path.join(args.save_dir, "pred.json"),
        args.test_root,
        args.save_dir,
        pred_threshold=args.score_thre,
    )
    print(f"[green]Inference finished. The result is saved in {args.save_dir} folder")

if __name__ == "__main__":
    main()
