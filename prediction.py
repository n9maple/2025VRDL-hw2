import torch
import json
import os
import argparse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from Dataset import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_root", type=str, default="data/test", help="Path to the test images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="save_model/epoch1.pth",
        help="Path to the saved model",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for test")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for test",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save_result",
        help="The directory where the result is saved",
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
        for idx, (images, image_ids) in enumerate(tqdm(test_loader, desc="Inference")):
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

    # save prediction
    with open(os.path.join(args.save_dir, "pred.json"), "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Inference finished. The result is saved in {args.save_dir} folder")


if __name__ == "__main__":
    main()
