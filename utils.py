from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import cv2
import json
import random


def COCOmap(predictions, ground_truths):
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds=predictions, target=ground_truths)
    result = metric.compute()
    return result["map"]


def prediction_draw(
    box_pred_path,
    image_folder_path,
    result_save_path,
    number_of_draw=10,
    random_select=True,
    pred_threshold=0.8,
):
    with open(box_pred_path, "r") as f:
        box_data = json.load(f)

    box_dict = {}
    for box in box_data:
        if box["score"] >= pred_threshold:
            box_dict.setdefault(box["image_id"], []).append(
                (box["category_id"] - 1, box["bbox"], box["score"])
            )

    if random_select:
        selected_ids = random.sample(list(box_dict.keys()), number_of_draw)
    else:
        sorted_box_dict = dict(sorted(box_dict.items()))
        selected_ids = list(sorted_box_dict.keys())[:10]

    image_size_factor = 3
    for image_id in selected_ids:
        img = cv2.imread(os.path.join(image_folder_path, str(image_id) + ".png"))
        resized_image = cv2.resize(
            img, None, fx=image_size_factor, fy=image_size_factor
        )
        for digit, bbox, score in box_dict[image_id]:
            bbox_int = [int(num * image_size_factor) for num in bbox]
            cv2.rectangle(
                resized_image,
                (bbox_int[0], bbox_int[1]),
                (bbox_int[0] + bbox_int[2], bbox_int[1] + bbox_int[3]),
                (0, 255, 0),
                1,
            )
            cv2.putText(
                resized_image,
                f"{digit}",
                (bbox_int[0], bbox_int[1] - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                resized_image,
                f"{score:.2f}",
                (
                    bbox_int[0]
                    + cv2.getTextSize(f"{digit}", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                    + 3,
                    bbox_int[1] - 1,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
            )
        cv2.imwrite(
            os.path.join(result_save_path, str(image_id) + ".png"), resized_image
        )

    print(f"The labeled images are saved in {result_save_path}")
