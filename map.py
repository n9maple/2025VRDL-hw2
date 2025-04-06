from torchmetrics.detection.mean_ap import MeanAveragePrecision


def COCOmap(predictions, ground_truths):
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds=predictions, target=ground_truths)
    result = metric.compute()
    return result["map"]
