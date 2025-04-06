import json
import pandas as pd
import argparse
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--box_pred_path",
        type=str,
        default="save_result/pred.json",
        help="Path to the box prediction json file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="save_result/pred.csv",
        help="Path to the saved result",
    )
    parser.add_argument(
        "--score_thre",
        type=float,
        default=0.8,
        help="The score threshold to pick the box",
    )
    args = parser.parse_args()
    num_pred(args.box_pred_path, args.save_path, args.score_thre)


def num_pred(box_pred_path, save_path, score_thre):
    with open(box_pred_path, "r") as f:
        box_data = json.load(f)

    columns = ["image_id", "pred_label"]
    pred_df = pd.DataFrame(columns=columns)

    box_dict = {}  # {image_id:[(digit, x_center_coordinate)]}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[green]Filter boxes...", total=len(box_data))
        for box in box_data:
            if box["score"] >= score_thre:
                x_center = (
                    box["bbox"][0] + box["bbox"][2] / 2
                )  # find the x-center of the box
                box_dict.setdefault(box["image_id"], []).append(
                    (box["category_id"] - 1, x_center)
                )
            progress.update(task, advance=1)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Make predictions...", total=len(box_dict))
        for image_id in box_dict:
            sorted_digit = sorted(
                box_dict[image_id], key=lambda x: x[1]
            )  # sort by x_center
            pred = "".join(str(t[0]) for t in sorted_digit)
            pred_df.loc[len(pred_df)] = [image_id, pred]
            progress.update(task, advance=1)

    pred_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
