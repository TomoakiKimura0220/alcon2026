import cv2
import numpy as np

from io_utils import (
    read_input_csv,
    write_output_csv,
    make_output_path,
    to_csv_path,
)
# 通常のプロトタイプ処理を使う場合はこちら。
# from segmenter import (
#     prototype_rice_weed_segmentation,
#     make_result_image,
#     judge_level,
# )

# YOLO-seg版を試す場合はこちら。
from yolo_segmenter import (
    yolo_rice_weed_segmentation as prototype_rice_weed_segmentation,
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
"""
ALCON2026 水稲・雑草セグメンテーション プロトタイプ実行ファイル。

このファイルは、仕様書で決められている「入力CSVの読み込み」から
「結果画像の保存」「output.csvの保存」までの全体フローを担当する。

入出力仕様の概要
----------------

入力:
    - カレントフォルダに存在する input.csv を読む。
    - input.csv の1行目には、処理対象画像数 N が書かれている。
    - input.csv の2行目以降には、以下の3項目が N 行分書かれている。

        width, height, image_path

      例:

        5184,3888,000\P1060881.JPG

      width:
          入力画像のx方向サイズ。
      height:
          入力画像のy方向サイズ。
      image_path:
          処理対象画像の相対パス。
          仕様書では Windows 形式の区切り文字「\」が使われる。
          実際の読み込み時には io_utils.py 側で macOS/Linux でも扱えるように変換する。

出力画像:
    - 各入力画像に対して、拡張子の前に「-output」を追加したJPG画像を保存する。
    - 例:

        入力:  000\P1060881.JPG
        出力:  000\P1060881-output.JPG

    - 出力画像の画素値は仕様書に従う。

        その他: 0x00, 0x00, 0x00 = 黒
        水稲:   0x80, 0x80, 0x80 = 灰色
        雑草:   0xff, 0xff, 0xff = 白

出力CSV:
    - カレントフォルダに output.csv を保存する。
    - output.csv の1行目には、処理した画像数 N を書く。
    - 2行目以降には、各画像について以下の7項目を書く。

        width, height, output_image_path, judge, p, w, r

      width:
          入力画像のx方向サイズ。
      height:
          入力画像のy方向サイズ。
      output_image_path:
          保存した結果画像の相対パス。
          仕様書に合わせて Windows 形式の区切り文字「\」で書き出す。
      judge:
          雑草量の4段階判定結果。
              0: 少ない
              1: 中程度
              2: 多い
              3: 甚大
      p:
          水稲領域の画素数。
      w:
          雑草領域の画素数。
      r:
          雑草比率[%]。
          r = w / (p + w) * 100.0 で計算する。
          現在は小数1桁に丸める。

責務分担:
    - main.py:
        全体の実行フローを管理する。
    - io_utils.py:
        input.csv の読み込み、output.csv の書き込み、出力パス生成を担当する。
    - segmenter.py:
        水稲・雑草領域の推定、結果画像生成、4段階判定を担当する。
"""

from pathlib import Path

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


# 仕様書で指定されている入力CSVファイル名。
# カレントフォルダ直下に存在する前提で処理する。
INPUT_CSV = "input.csv"

# 仕様書で指定されている出力CSVファイル名。
# カレントフォルダ直下に保存する。
OUTPUT_CSV = "output.csv"


def process_one(width: int, height: int, image_path: Path) -> list:
    """
    input.csv の1行分に対応する画像を処理する。

    処理内容:
        1. image_path の画像を読み込む。
        2. 入力CSVに書かれた width, height と実画像サイズを確認する。
        3. 水稲マスク rice_mask と雑草マスク weed_mask を推定する。
        4. 水稲画素数 p と雑草画素数 w を数える。
        5. 雑草比率 r = w / (p + w) * 100.0 を計算する。
        6. r から4段階判定 judge を求める。
        7. 仕様通りの画素値を持つ結果画像を保存する。
        8. output.csv に書き込む1行分のデータを返す。

    Args:
        width:
            input.csv に書かれている入力画像のx方向サイズ。
        height:
            input.csv に書かれている入力画像のy方向サイズ。
        image_path:
            処理対象画像のパス。
            io_utils.read_input_csv() により、macOS/Linuxでも読めるPathに変換済み。

    Returns:
        output.csv の2行目以降に書き込む1行分のリスト。

        形式:
            [
                width,
                height,
                output_image_path,
                judge,
                p,
                w,
                r,
            ]

        output_image_path は、仕様書に合わせて Windows 形式の区切り文字「\\」で返す。
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    actual_h, actual_w = image.shape[:2]

    # input.csv に記載されたサイズと実画像サイズが異なる場合は警告する。
    # ただし、ここでは処理を止めない。
    if actual_w != width or actual_h != height:
        print(
            f"[WARN] サイズ不一致: {image_path} "
            f"input.csv=({width}, {height}), actual=({actual_w}, {actual_h})"
        )

    # 水稲・雑草マスクを作成する。
    # 現在は import 側で prototype版 / YOLO版 を切り替える。
    # main.py側では rice_mask, weed_mask を受け取るだけにしておく。
    rice_mask, weed_mask = prototype_rice_weed_segmentation(image)

    # p: 水稲領域の画素数。
    # w: 雑草領域の画素数。
    p = int(np.count_nonzero(rice_mask))
    w = int(np.count_nonzero(weed_mask))

    # 雑草比率 r[%] を計算する。
    # 仕様書の式: r = w / (p + w) * 100.0
    # p + w が0の場合は、植物領域が存在しないものとして r = 0.0 とする。
    if p + w == 0:
        ratio = 0.0
    else:
        ratio = round(w / (p + w) * 100.0, 1)

    # 4段階判定。
    # 現在の閾値は segmenter.py の judge_level() 側で定義している。
    judge = judge_level(ratio)

    # 仕様書に従った3値画像を作成する。
    # その他: 黒、水稲: 灰色、雑草: 白。
    result_image = make_result_image(rice_mask, weed_mask)

    # 入力画像名の拡張子直前に「-output」を追加したパスを作る。
    output_path = make_output_path(image_path)

    # 結果画像をJPG形式で保存する。
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


def main() -> None:
    """
    ALCON2026 プロトタイプのメイン処理。

    全体フロー:
        1. input.csv を読み込む。
        2. input.csv に書かれた画像を先頭から順番に処理する。
        3. 各画像に対して結果画像を保存する。
        4. 各画像の判定結果、画素数、雑草比率を output_rows に蓄積する。
        5. 最後に output.csv を保存する。
    """
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