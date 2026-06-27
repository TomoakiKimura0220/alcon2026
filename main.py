import cv2
import numpy as np

from io_utils import (
    read_input_csv,
    write_output_csv,
    make_output_path,
    to_csv_path,
)
from segmenter import (
    prototype_rice_weed_segmentation,
    make_result_image,
    judge_level,
)


INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"


def process_one(width: int, height: int, image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    actual_h, actual_w = image.shape[:2]

    if actual_w != width or actual_h != height:
        print(
            f"[WARN] サイズ不一致: {image_path} "
            f"input.csv=({width}, {height}), actual=({actual_w}, {actual_h})"
        )

    rice_mask, weed_mask = prototype_rice_weed_segmentation(image)

    p = int(np.count_nonzero(rice_mask))
    w = int(np.count_nonzero(weed_mask))

    if p + w == 0:
        ratio = 0.0
    else:
        ratio = round(w / (p + w) * 100.0, 1)

    judge = judge_level(ratio)

    result_image = make_result_image(rice_mask, weed_mask)

    output_path = make_output_path(image_path)
    cv2.imwrite(str(output_path), result_image)

    return [
        width,
        height,
        to_csv_path(output_path),
        judge,
        p,
        w,
        ratio,
    ]


def main():
    input_rows = read_input_csv(INPUT_CSV)

    output_rows = []

    for width, height, image_path in input_rows:
        print(f"[INFO] processing: {image_path}")
        output_row = process_one(width, height, image_path)
        output_rows.append(output_row)

    write_output_csv(output_rows, OUTPUT_CSV)

    print(f"[INFO] done: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()