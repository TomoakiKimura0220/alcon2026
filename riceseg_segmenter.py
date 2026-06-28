"""
RiceSEGで学習したALCON用3クラスU-Netを使うセグメンテーション処理。

目的:
    train_riceseg_unet.py で学習した .pth を読み込み、
    ALCON公開データの画像に対して以下の3クラスを推論する。

ALCON用3クラス:
    0: other = 背景・水面・土など
    1: rice  = 水稲
    2: weed  = 雑草・浮草

main.py から使う関数:
    - riceseg_alcon_segmentation(image_bgr)
    - make_result_image(rice_mask, weed_mask)
    - judge_level(ratio)

処理の概要:
    1. 入力画像を512×512パッチに分割する。
    2. 各パッチをRiceSEG U-Netで other / rice / weed の3クラスに推論する。
    3. 各パッチの推論結果を元画像サイズのラベル画像に貼り戻す。
    4. rice / weed のboolマスクを返す。

現在の実験条件:
    パッチ分割の効果を確認するため、HSVベースの田んぼ領域マスクは適用しない。

出力画像の色:
    other: black
    rice : green
    weed : red

注意:
    ALCONの正式仕様では、出力画像は
        other: black
        rice : gray
        weed : white
    だが、このファイルでは目視確認しやすいようにクラスごとに色分けする。
    提出仕様に合わせる段階では、make_result_image() の色を gray/white に戻すこと。

補足:
    estimate_field_mask() / apply_field_mask() は残しているが、現在の
    riceseg_alcon_segmentation() では使用していない。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_riceseg_unet import UNet


CHECKPOINT_PATH = Path("checkpoints/riceseg_unet_alcon3_w45_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
NUM_CLASSES = 3

# パッチ分割推論の設定。
# RiceSEGは512×512画像で学習されているため、ALCON公開データ全体を512×512へ縮小せず、
# 元画像を512×512パッチに分けて推論する。
PATCH_SIZE = 512
PATCH_STRIDE = 512

# 田んぼ領域マスクの暫定パラメータ。
# RiceSEGの学習画像は基本的に田んぼ内パッチだが、ALCON公開データには空・建物・道路などの背景が写る。
# そのため、明らかに田んぼではない領域を先に除外してから rice / weed の画素数を数える。
FIELD_IGNORE_TOP_RATIO = 0.12
FIELD_MIN_AREA_RATIO = 0.01
FIELD_MORPH_KERNEL_SIZE = 15

# OpenCVはBGR順で色を扱う。
# 目視確認用の色分け。
COLOR_OTHER_BGR = (0, 0, 0)        # black
COLOR_RICE_BGR = (0, 180, 0)       # green
COLOR_WEED_BGR = (0, 0, 255)       # red

_model: UNet | None = None


def estimate_field_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    ALCON公開データ向けに、田んぼらしい領域のマスクを推定する。

    目的:
        RiceSEG U-Netが空・建物・道路などの背景を rice / weed と誤認するのを抑える。

    方針:
        - HSV色空間で青空っぽい領域を除外する。
        - 高輝度・低彩度の白い雲や白飛び領域を除外する。
        - 緑〜黄緑〜茶色系の植生・水田らしい領域を残す。
        - 画像の最上部は遠景・空が入りやすいため、暫定的に除外する。
        - モルフォロジー処理で小ノイズを減らす。
        - 小さすぎる連結成分を除外する。

    Args:
        image_bgr:
            OpenCVで読み込んだBGR画像。

    Returns:
        field_mask:
            shape=(H, W), dtype=bool
            田んぼ領域とみなす画素がTrue。
    """
    h, w = image_bgr.shape[:2]

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # OpenCVのHueは0〜179。
    # 緑〜黄緑〜黄色〜茶色寄りを広めに残す。
    # 稲・雑草・水田周辺の泥や濁った水面を残すため、かなり緩めの条件にしている。
    green_yellow_brown = (hue >= 10) & (hue <= 95) & (sat >= 25) & (val >= 25)

    # 暗い水面・影・泥を完全に消すと田んぼ領域が分断されるため、
    # 彩度が中程度以上の暗部も補助的に残す。
    dark_field_like = (sat >= 35) & (val >= 20) & (val <= 120)

    # 青空を除外する。
    sky_blue = (hue >= 95) & (hue <= 135) & (sat >= 30) & (val >= 80)

    # 雲・白い建物・白飛びに近い領域を除外する。
    bright_low_saturation = (sat <= 45) & (val >= 170)

    field_mask = (green_yellow_brown | dark_field_like) & ~sky_blue & ~bright_low_saturation

    # 画像上部は空・遠景が入りやすいので暫定的に除外する。
    ignore_top = int(h * FIELD_IGNORE_TOP_RATIO)
    if ignore_top > 0:
        field_mask[:ignore_top, :] = False

    field_mask_uint8 = field_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (FIELD_MORPH_KERNEL_SIZE, FIELD_MORPH_KERNEL_SIZE),
    )
    field_mask_uint8 = cv2.morphologyEx(field_mask_uint8, cv2.MORPH_CLOSE, kernel)
    field_mask_uint8 = cv2.morphologyEx(field_mask_uint8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(field_mask_uint8, connectivity=8)
    min_area = int(h * w * FIELD_MIN_AREA_RATIO)

    cleaned = np.zeros((h, w), dtype=np.uint8)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255

    # 条件が厳しすぎて何も残らない場合は、推論結果を全消ししないため全領域を田んぼ扱いに戻す。
    if np.count_nonzero(cleaned) == 0:
        print("[WARN] field_mask が空になったため、全領域を田んぼ領域として扱います。")
        return np.ones((h, w), dtype=bool)

    return cleaned.astype(bool)


def apply_field_mask(
    rice_mask: np.ndarray,
    weed_mask: np.ndarray,
    field_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    田んぼ領域外の rice / weed 推論結果を除外する。

    Args:
        rice_mask:
            水稲領域のboolマスク。
        weed_mask:
            雑草領域のboolマスク。
        field_mask:
            田んぼ領域のboolマスク。

    Returns:
        masked_rice:
            田んぼ領域外をFalseにした水稲マスク。
        masked_weed:
            田んぼ領域外をFalseにした雑草マスク。
    """
    masked_rice = rice_mask & field_mask
    masked_weed = weed_mask & field_mask

    return masked_rice, masked_weed


def make_field_mask_debug_image(image_bgr: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    """
    田んぼ領域マスク確認用のデバッグ画像を作る。

    田んぼ領域と判定された場所を元画像上に半透明の緑で重ねる。
    main.py からは未使用だが、必要に応じて確認用に使う。
    """
    overlay = image_bgr.copy()
    overlay[field_mask] = (0, 255, 0)

    debug = cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)
    return debug


def load_model() -> UNet:
    """checkpointからALCON用3クラスU-Netを読み込む。"""
    global _model

    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"checkpoint が存在しません: {CHECKPOINT_PATH}\n"
            "Colabで作成した .pth を checkpoints/riceseg_unet_best.pth に配置してください。"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
    base_channels = int(checkpoint.get("base_channels", 32))
    label_mode = checkpoint.get("label_mode", "unknown")

    if num_classes != NUM_CLASSES:
        raise ValueError(
            "ALCON用3クラスモデルではありません。"
            f" num_classes={num_classes}, expected={NUM_CLASSES}"
        )

    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("[INFO] loaded RiceSEG ALCON model")
    print(f"  checkpoint : {CHECKPOINT_PATH}")
    print(f"  device     : {DEVICE}")
    print(f"  epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"  best_loss  : {checkpoint.get('best_val_loss', 'unknown')}")
    print(f"  label_mode : {label_mode}")
    print(f"  classes    : {num_classes}")

    _model = model
    return _model


def _make_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    """
    画像端まで確実に推論するためのパッチ開始座標リストを作る。

    例:
        length=1200, patch_size=512, stride=512 の場合
        [0, 512, 688] のように最後のパッチが画像端に揃う。
    """
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


@torch.no_grad()
def _predict_patch(model: UNet, patch_bgr: np.ndarray) -> np.ndarray:
    """
    1枚のパッチをRiceSEG U-Netで推論し、クラスID画像を返す。

    Args:
        model:
            読み込み済みU-Net。
        patch_bgr:
            BGR形式のパッチ画像。

    Returns:
        pred_patch:
            shape=(patch_h, patch_w), dtype=uint8
            0=other, 1=rice, 2=weed。
    """
    patch_h, patch_w = patch_bgr.shape[:2]

    if patch_h != IMAGE_SIZE or patch_w != IMAGE_SIZE:
        resized_bgr = cv2.resize(
            patch_bgr,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        resized_bgr = patch_bgr

    patch_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    patch_tensor = torch.from_numpy(patch_rgb).float() / 255.0
    patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    logits = model(patch_tensor)

    if logits.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        logits = F.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    if patch_h != IMAGE_SIZE or patch_w != IMAGE_SIZE:
        pred = cv2.resize(
            pred,
            (patch_w, patch_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return pred


@torch.no_grad()
def riceseg_alcon_segmentation(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    BGR画像から水稲マスク・雑草マスクを推論する。

    現在はパッチ分割推論の効果を確認するため、背景除去用のfield_maskは使わない。

    Args:
        image_bgr:
            OpenCVで読み込んだBGR画像。

    Returns:
        rice_mask:
            shape=(H, W), dtype=bool
            水稲領域がTrue。
        weed_mask:
            shape=(H, W), dtype=bool
            雑草領域がTrue。
    """
    model = load_model()

    original_h, original_w = image_bgr.shape[:2]
    pred_full = np.zeros((original_h, original_w), dtype=np.uint8)

    y_starts = _make_patch_starts(original_h, PATCH_SIZE, PATCH_STRIDE)
    x_starts = _make_patch_starts(original_w, PATCH_SIZE, PATCH_STRIDE)

    for y in y_starts:
        y_end = min(y + PATCH_SIZE, original_h)
        for x in x_starts:
            x_end = min(x + PATCH_SIZE, original_w)

            patch_bgr = image_bgr[y:y_end, x:x_end]
            pred_patch = _predict_patch(model, patch_bgr)

            pred_full[y:y_end, x:x_end] = pred_patch

    rice_mask = pred_full == 1
    weed_mask = pred_full == 2

    return rice_mask, weed_mask


def make_result_image(rice_mask: np.ndarray, weed_mask: np.ndarray) -> np.ndarray:
    """
    クラスごとに色分けした出力画像を作る。

    Args:
        rice_mask:
            水稲領域のboolマスク。
        weed_mask:
            雑草領域のboolマスク。

    Returns:
        result:
            shape=(H, W, 3), dtype=uint8
            other=黒, rice=緑, weed=赤 のBGR画像。
    """
    h, w = rice_mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    result[:] = COLOR_OTHER_BGR
    result[rice_mask] = COLOR_RICE_BGR
    result[weed_mask] = COLOR_WEED_BGR

    return result


def make_alcon_spec_result_image(rice_mask: np.ndarray, weed_mask: np.ndarray) -> np.ndarray:
    """
    ALCON提出仕様に合わせた出力画像を作る。

    仕様:
        other: 0x00, 0x00, 0x00
        rice : 0x80, 0x80, 0x80
        weed : 0xff, 0xff, 0xff

    今は目視確認を優先するため main.py からは未使用。
    提出時に必要なら make_result_image の中身をこちらに差し替える。
    """
    h, w = rice_mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    result[rice_mask] = (128, 128, 128)
    result[weed_mask] = (255, 255, 255)

    return result


def judge_level(ratio: float) -> int:
    """
    雑草率から4段階判定を返す。

    Args:
        ratio:
            r = weed_pixels / (rice_pixels + weed_pixels) * 100

    Returns:
        0: 少ない
        1: 中程度
        2: 多い
        3: 甚大

    暫定閾値:
        公開データのフォルダ名 000/111/222/333 と照合して、後で調整する。
    """
    if ratio < 25.0:
        return 0
    if ratio < 50.0:
        return 1
    if ratio < 75.0:
        return 2
    return 3