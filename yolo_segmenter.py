from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# 学習済みYOLO-segモデルのパス。
# 最初は自前学習前なので、仮に yolo11n-seg.pt を使う。
# 自前学習後は runs/segment/.../weights/best.pt に差し替える。
MODEL_PATH = "yolo11n-seg.pt"

# YOLOのクラスID。
# 自前学習時の data.yaml と一致させること。
RICE_CLASS_ID = 0
WEED_CLASS_ID = 1

# モデルは毎画像ごとに読み込むと遅いので、モジュール読み込み時に1回だけロードする。
_model = YOLO(MODEL_PATH)


def yolo_rice_weed_segmentation(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    YOLO-segを使って水稲マスクと雑草マスクを作成する。

    Args:
        image_bgr:
            OpenCVで読み込んだBGR画像。

    Returns:
        rice_mask:
            水稲領域を255、それ以外を0とする2値画像。
        weed_mask:
            雑草領域を255、それ以外を0とする2値画像。

    注意:
        現時点で yolo11n-seg.pt を使う場合、COCO学習済みモデルなので
        rice / weed クラスは存在しない。
        そのため、本格動作には水稲・雑草データで学習した best.pt が必要。
    """
    h, w = image_bgr.shape[:2]

    rice_mask = np.zeros((h, w), dtype=np.uint8)
    weed_mask = np.zeros((h, w), dtype=np.uint8)

    results = _model.predict(
        source=image_bgr,
        imgsz=640,
        conf=0.25,
        verbose=False,
    )

    if len(results) == 0:
        return rice_mask, weed_mask

    result = results[0]

    if result.masks is None or result.boxes is None:
        return rice_mask, weed_mask

    masks = result.masks.data.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    for mask, class_id in zip(masks, class_ids):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = mask_resized > 0.5

        if class_id == RICE_CLASS_ID:
            rice_mask[mask_binary] = 255
        elif class_id == WEED_CLASS_ID:
            weed_mask[mask_binary] = 255

    # 重なりがあった場合は雑草を優先する。
    rice_mask[weed_mask > 0] = 0

    return rice_mask, weed_mask


def make_result_image(
    rice_mask: np.ndarray,
    weed_mask: np.ndarray,
) -> np.ndarray:
    """
    仕様通りの結果画像を作る。

    その他: 0,0,0
    水稲: 128,128,128
    雑草: 255,255,255
    """
    result = np.zeros((*rice_mask.shape, 3), dtype=np.uint8)

    result[rice_mask > 0] = (128, 128, 128)
    result[weed_mask > 0] = (255, 255, 255)

    return result


def judge_level(ratio: float) -> int:
    """
    雑草比率 r[%] から4段階判定する。

    現在はプロトタイプ用に25%刻み。
    正解付きデータで後から調整する。
    """
    if ratio < 25.0:
        return 0
    elif ratio < 50.0:
        return 1
    elif ratio < 75.0:
        return 2
    else:
        return 3