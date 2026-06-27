from pathlib import Path
import csv

import cv2
import numpy as np


INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"


def normalize_path(path_str: str) -> Path:
    """
    input.csv内の 000\\P1060881.JPG のようなWindows風パスを
    macOS/Linuxでも読めるPathに変換する。
    """
    return Path(path_str.replace("\\", "/"))


def make_output_path(input_path: Path) -> Path:
    """
    000/P1060881.JPG -> 000/P1060881-output.JPG
    """
    return input_path.with_name(f"{input_path.stem}-output{input_path.suffix}")


def read_input_csv(csv_path: str) -> list[tuple[int, int, Path]]:
    rows = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        first = next(reader)
        n = int(first[0])

        for i, row in enumerate(reader, start=1):
            if not row or len(row) < 3:
                continue

            width = int(row[0])
            height = int(row[1])
            image_path = normalize_path(row[2].strip())

            rows.append((width, height, image_path))

    if len(rows) != n:
        print(f"[WARN] input.csvの枚数 N={n} と実際の行数 {len(rows)} が一致しません")

    return rows


def exg_vegetation_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    仮の植物領域抽出。
    ExG = 2G - R - B を使って緑っぽい領域を抽出する。
    """
    b, g, r = cv2.split(image_bgr.astype(np.float32))

    exg = 2 * g - r - b
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    exg_blur = cv2.GaussianBlur(exg, (5, 5), 0)

    _, mask = cv2.threshold(
        exg_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def prototype_rice_weed_segmentation(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    YOLO導入前の仮処理。

    戻り値:
        rice_mask: 水稲領域 0/255
        weed_mask: 雑草領域 0/255

    現段階では雑草判定の本質部分は仮。
    植物領域を抽出し、画像下側・小領域などを雑草寄りにする簡易処理。
    """
    plant_mask = exg_vegetation_mask(image_bgr)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        plant_mask,
        connectivity=8,
    )

    rice_mask = np.zeros_like(plant_mask)
    weed_mask = np.zeros_like(plant_mask)

    h, w = plant_mask.shape

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        cx, cy = centroids[label_id]

        if area < 50:
            continue

        component = (labels == label_id)

        # 仮ルール:
        # 小さい孤立領域や画面下側の植物を雑草寄りにする。
        # ここはあとでYOLOや水稲列推定に差し替える。
        if area < 1500 or cy > h * 0.65:
            weed_mask[component] = 255
        else:
            rice_mask[component] = 255

    return rice_mask, weed_mask


def make_result_image(rice_mask: np.ndarray, weed_mask: np.ndarray) -> np.ndarray:
    """
    仕様通りの結果画像を作る。
    水稲: 128
    雑草: 255
    その他: 0
    """
    result = np.zeros((*rice_mask.shape, 3), dtype=np.uint8)

    result[rice_mask > 0] = (128, 128, 128)
    result[weed_mask > 0] = (255, 255, 255)

    return result


def judge_level(ratio: float) -> int:
    """
    雑草比率 r[%] から4段階判定。
    閾値はプロトタイプ用の仮値。
    ここは正解付きテスト画像を見てチューニングする。
    """
    if ratio < 5.0:
        return 0
    elif ratio < 15.0:
        return 1
    elif ratio < 30.0:
        return 2
    else:
        return 3


def process_one(width: int, height: int, image_path: Path) -> list:
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

    # output.csv上は仕様に合わせて Windows風パスで書く
    output_name_for_csv = str(output_path).replace("/", "\\")

    return [
        width,
        height,
        output_name_for_csv,
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
        row = process_one(width, height, image_path)
        output_rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([len(output_rows), "", ""])

        for row in output_rows:
            writer.writerow(row)

    print(f"[INFO] done: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()