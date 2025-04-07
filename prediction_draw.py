from utils import prediction_draw
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--box_pred_path",
        type=str,
        default="data/test",
        help="Path to the box prediction json file",
    )
    parser.add_argument(
        "-i",
        "--image_folder_path",
        type=str,
        default="data/test",
        help="Path to the image folder",
    )
    parser.add_argument(
        "-r",
        "--result_save_path",
        type=str,
        default="data/test",
        help="Path to save the images",
    )
    parser.add_argument(
        "-n",
        "--number_of_draw",
        type=int,
        default=10,
        help="Number of images to be drawn",
    )
    parser.add_argument(
        "-s",
        "--random_select",
        action="store_true",
        help="The drawn image will be selected randomly if set true. Otherwise the first N images will be selected",
    )
    parser.add_argument(
        "-t",
        "--pred_threshold",
        type=float,
        default=0.8,
        help="The threshold to predict a box (this value should between 0 and 1, and 0.8~0.9 is suggested)",
    )
    args = parser.parse_args()
    prediction_draw(
        args.box_pred_path,
        args.image_folder_path,
        args.result_save_path,
        args.number_of_draw,
        args.random_select,
        args.pred_threshold,
    )


if __name__ == "__main__":
    main()
