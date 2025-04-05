from torchvision.ops import box_iou
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def compute_map(all_preds, all_gts, iou_threshold=0.5):
    """
    計算 Mean Average Precision (mAP)。
    - `all_preds`: 預測結果 (list of dicts, each contains "boxes" and "labels")
    - `all_gts`: Ground Truth 標註 (list of dicts, each contains "boxes" and "labels")
    - `iou_threshold`: IoU 門檻（預設 0.5）
    """
    aps = []  # 儲存所有類別的 AP 值
    all_classes = set()  # 儲存所有類別

    # 收集所有類別
    for pred in all_preds:
        all_classes.update(pred["labels"].tolist())
    for gt in all_gts:
        all_classes.update(gt["labels"].tolist())

    # 遍歷所有類別，分開計算 AP
    for cls in all_classes:
        tp_list, fp_list, scores = [], [], []

        # 遍歷所有圖片的 GT 和 預測
        for pred, gt in zip(all_preds, all_gts):
            # 取得該類別的 GT 和 預測框
            gt_boxes = gt["boxes"][gt["labels"] == cls]  # 只取出該類別的 GT
            pred_boxes = pred["boxes"][pred["labels"] == cls]  # 只取出該類別的預測
            if "scores" in pred and len(pred["scores"]) > 0:
                pred_scores = pred["scores"][pred["labels"] == cls]
            else:
                pred_scores = torch.tensor([])  # 空張量
            
            scores.extend(pred_scores.cpu().tolist())  # 收集所有 scores
            
            # 若沒有 GT，則所有預測都是 FP
            if len(gt_boxes) == 0:
                fp_list.extend([1] * len(pred_boxes))
                tp_list.extend([0] * len(pred_boxes))
                continue

            # 計算 IoU
            ious = box_iou(pred_boxes, gt_boxes)
            matched = ious > iou_threshold  # 找出匹配的框

            # 計算 TP 和 FP
            tp = matched.sum(dim=1).cpu().tolist()  # 有匹配的為 TP
            fp = (1 - matched.sum(dim=1)).cpu().tolist()  # 無匹配的為 FP

            tp_list.extend(tp)
            fp_list.extend(fp)

        # 按照 score 降序排列
        sorted_indices = np.argsort(-np.array(scores))
        tp_list = np.array(tp_list)[sorted_indices]
        fp_list = np.array(fp_list)[sorted_indices]

        # 計算 Precision 和 Recall
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / len(gt_boxes)
        # 計算 AP (PR 曲線下面積)
        ap = np.trapezoid(precisions, recalls)
        aps.append(ap)

    # 計算 mAP (所有類別的 AP 平均值)
    return np.mean(aps) if aps else 0.0

def compute_map_new(all_preds, all_gts, iou_threshold=0.5, class_num = 11):

    all_preds_box, all_gts_box = [], {} # all_preds_box : list of list, all_gts_box : {image_id:[box tensor(n*4), label tensor(n)]}
    class_pred_count, class_gt_count = {}, {} # {class: number}
    for i in range(class_num):
        class_pred_count[i] = 0
        class_gt_count[i] = 0

    for single_img_pred, single_img_gt in zip(all_preds, all_gts):
        image_id = single_img_gt['image_id'].item()
        for box, label, score in zip(single_img_pred['boxes'], single_img_pred['labels'], single_img_pred['scores']):
            label_int = label.item()
            score_float = score.item()
            if label_int == 0: # ignore background
                continue
            all_preds_box.append([image_id, box, label_int, score_float])
            class_pred_count[label_int] += 1

        all_gts_box[image_id] = [single_img_gt['boxes'], single_img_gt['labels']]
        for label in single_img_gt['labels']:
            label_int = label.item()
            class_gt_count[label_int] += 1
    
    all_class_pt = {} # {class:[list of binary pt (np.array)]}
    all_class_pred_accum = {}
    all_class_pt_ind = {}
    all_class_pt_count = {}
    for i in range(class_num):
        all_class_pt[i] = np.zeros(class_pred_count[i], dtype=int)
        all_class_pred_accum[i] = np.arange(1, class_pred_count[i]+1, dtype=int)
        all_class_pt_ind[i] = 0
        all_class_pt_count[i] = 0
    # sort all_preds_box by score
    sorted_preds_box = sorted(all_preds_box, key=lambda x: x[3], reverse=True)
    for [image_id, box, label, score] in sorted_preds_box:
        box_match = False
        ious = box_iou(box.unsqueeze(0), all_gts_box[image_id][0]).squeeze(0)
        indices = torch.nonzero(ious > iou_threshold)
        for ind in indices:
            ind = ind.item()
            if all_gts_box[image_id][1][ind].item() == label:
                box_match = True
                break
        if box_match:
            all_class_pt_count[label] += 1
            all_class_pt[label][all_class_pt_ind[label]] = all_class_pt_count[label]
        all_class_pt_ind[label] += 1

    aps = []
    for label in all_class_pt:
        if label == 0: # ignore background
            continue
        precision = all_class_pt[label]/all_class_pred_accum[label]
        recall = all_class_pt[label]/class_gt_count[label]
        ap = np.trapezoid(precision, recall)
        aps.append(ap)
    return np.mean(aps) if aps else 0.0

def COCOmap(predictions, ground_truths):
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds=predictions, target=ground_truths)
    result = metric.compute()
    return result['map']