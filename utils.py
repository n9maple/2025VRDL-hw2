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
                (box["category_id"] - 1, box["bbox"], box["socre"])
            )

    if random_select:
        selected_ids = random.sample(list(box_dict.keys()), number_of_draw)
    else:
        sorted_box_dict = dict(sorted(box_dict.items()))
        selected_ids = list(sorted_box_dict.keys())[:10]
    for image_id in selected_ids:
        img = cv2.imread(os.path.join(image_folder_path, image_id + ".png"))
        for digit, bbox, score in box_dict[image_id]:
            bbox_int = [int(num) for num in bbox]
            cv2.rectangle(
                img,
                (bbox_int[0], bbox_int[1]),
                (bbox_int[0] + bbox_int[2], bbox_int[1] + bbox_int[3]),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                img,
                (bbox_int[0] + 2, bbox_int[1] + 2),
                (bbox_int[0] + 50, bbox_int[1] + 10),
                (225, 255, 225),
                -1,
            )
            cv2.putText(
                img,
                "label:" + digit + " score:" + score,
                (bbox_int[0] + 2, bbox_int[1] + bbox_int[3] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
            )
        cv2.imwrite(os.path.join(result_save_path, image_id + ".png"), img)

    print(f"The labeled images are saved in {result_save_path}")
