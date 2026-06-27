import cv2
import numpy as np


def exg_vegetation_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    ExG = 2G - R - B を使って植物領域を抽出する仮処理。
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


def prototype_rice_weed_segmentation(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    YOLO導入前の仮処理。

    戻り値:
        rice_mask: 水稲領域 0/255
        weed_mask: 雑草領域 0/255
    """
    plant_mask = exg_vegetation_mask(image_bgr)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        plant_mask,
        connectivity=8,
    )

    rice_mask = np.zeros_like(plant_mask)
    weed_mask = np.zeros_like(plant_mask)

    h, _ = plant_mask.shape

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        _, cy = centroids[label_id]

        if area < 50:
            continue

        component = labels == label_id

        # 仮ルール:
        # 小さい孤立領域や画面下側の植物を雑草寄りにする。
        # TODO: YOLO-seg推論や水稲列推定に差し替える。
        if area < 1500 or cy > h * 0.65:
            weed_mask[component] = 255
        else:
            rice_mask[component] = 255

    return rice_mask, weed_mask


def make_result_image(
    rice_mask: np.ndarray,
    weed_mask: np.ndarray,
) -> np.ndarray:
    """
    仕様通りの結果画像を作る。

    水稲: 128,128,128
    雑草: 255,255,255
    その他: 0,0,0
    """
    result = np.zeros((*rice_mask.shape, 3), dtype=np.uint8)

    result[rice_mask > 0] = (128, 128, 128)
    result[weed_mask > 0] = (255, 255, 255)

    return result


def judge_level(ratio: float) -> int:
    """
    雑草比率 r[%] から4段階判定。
    プロトタイプでは25%刻み。

    0: 少ない
    1: 中程度
    2: 多い
    3: 甚大
    """
    if ratio < 25.0:
        return 0
    elif ratio < 50.0:
        return 1
    elif ratio < 75.0:
        return 2
    else:
        return 3