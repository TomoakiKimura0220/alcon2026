"""
ALCON2026 水稲・雑草セグメンテーション実行ファイル。

このファイルは、仕様書で決められている「入力CSVの読み込み」から
「結果画像の保存」「output.csvの保存」までの全体フローを担当する。

現在の推論器:
    RiceSEGで学習したALCON用3クラスU-Net

入出力仕様の概要
----------------

入力:
    - カレントフォルダに存在する input.csv を読む。
    - input.csv の1行目には、処理対象画像数 N が書かれている。
    - input.csv の2行目以降には、以下の3項目が N 行分書かれている。

        width, height, image_path

      例:

        5184,3888,000\P1060881.JPG

出力画像:
    - 各入力画像に対して、拡張子の前に「-output」を追加したJPG画像を保存する。
    - 現在は目視確認のため、riceseg_segmenter.py 側でクラスごとに色分けしている。

        other: 黒
        rice : 緑
        weed : 赤

    - ALCON提出仕様では以下の3値画像に戻す必要がある。

        other: 黒
        rice : 灰色
        weed : 白

出力CSV:
    - カレントフォルダに output.csv を保存する。
    - output.csv の1行目には、処理した画像数 N を書く。
    - 2行目以降には、各画像について以下の7項目を書く。

        width, height, output_image_path, judge, p, w, r

      p:
          水稲領域の画素数。
      w:
          雑草領域の画素数。
      r:
          雑草比率[%]。
          r = w / (p + w) * 100.0 で計算する。
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from io_utils import (
    read_input_csv,
    write_output_csv,
    make_output_path,
    to_csv_path,
)
from riceseg_segmenter import (
    riceseg_alcon_segmentation as prototype_rice_weed_segmentation,
    make_result_image,
    judge_level,
)


INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"


def process_one(width: int, height: int, image_path: Path) -> list:
    """
    input.csv の1行分に対応する画像を処理する。

    処理内容:
        1. 画像を読み込む。
        2. 入力CSVに書かれた width, height と実画像サイズを確認する。
        3. RiceSEG U-Netで水稲マスク rice_mask と雑草マスク weed_mask を推定する。
        4. 水稲画素数 p と雑草画素数 w を数える。
        5. 雑草比率 r = w / (p + w) * 100.0 を計算する。
        6. r から4段階判定 judge を求める。
        7. 結果画像を保存する。
        8. output.csv に書き込む1行分のデータを返す。
    """
    start_time = perf_counter()
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

    elapsed_time = perf_counter() - start_time

    print(
        f"[INFO] result: judge={judge}, "
        f"p={p}, w={w}, r={ratio}, "
        f"time={elapsed_time:.2f}s, output={output_path}"
    )

    return [
        width,
        height,
        to_csv_path(output_path),
        judge,
        p,
        w,
        ratio,
    ]


def main() -> None:
    """ALCON2026 推論処理のメイン関数。"""
    total_start_time = perf_counter()
    input_rows = read_input_csv(INPUT_CSV)

    output_rows = []

    for width, height, image_path in input_rows:
        print(f"[INFO] processing: {image_path}")
        output_row = process_one(width, height, image_path)
        output_rows.append(output_row)

    write_output_csv(output_rows, OUTPUT_CSV)

    total_elapsed_time = perf_counter() - total_start_time

    print(f"[INFO] done: {OUTPUT_CSV}")
    print(f"[INFO] total time: {total_elapsed_time:.2f}s")

    if output_rows:
        average_time = total_elapsed_time / len(output_rows)
        print(f"[INFO] average time: {average_time:.2f}s/image")


if __name__ == "__main__":
    main()