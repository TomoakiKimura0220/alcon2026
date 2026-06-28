

"""
RiceSEG U-Net の検証用推論スクリプト。

目的:
    学習済みモデル checkpoints/riceseg_unet_best.pth を使って、
    RiceSEG Japan/TKO_3 の検証画像を推論し、目視確認用の画像を保存する。

入力:
    - checkpoints/riceseg_unet_best.pth
    - datasets/RiceSEG/global rice segmentation/Japan/TKO_3/rgb/*.jpg
    - datasets/RiceSEG/global rice segmentation/Japan/TKO_3/label/*.png

出力:
    - runs/riceseg_val/*.jpg

保存される確認画像:
    左: 元RGB画像
    中: 正解ラベルの可視化
    右: 予測ラベルの可視化

RiceSEG クラスIDの想定:
    0: background
    1: green vegetation
    2: senescent vegetation
    3: panicle
    4: weed
    5: duckweed

可視化色:
    0 background           : black
    1 green vegetation     : green
    2 senescent vegetation : yellow/brown
    3 panicle              : orange
    4 weed                 : white
    5 duckweed             : cyan

注意:
    このスクリプトはALCON形式のoutput.csvは作らない。
    あくまで「RiceSEGで学習したモデルが何を予測しているか」を見るための確認用。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from riceseg_dataset import RiceSEGDataset, RICESEG_NUM_CLASSES
from train_riceseg_unet import UNet


CHECKPOINT_PATH = Path("checkpoints/riceseg_unet_best.pth")
OUTPUT_DIR = Path("runs/riceseg_val")

DEVICE = "cpu"
IMAGE_SIZE = 512
MAX_SAMPLES = 100

# OpenCVはBGRで保存するため、ここではBGR順で色を定義する。
# label/pred のクラスID画像を、人間が見やすいカラー画像に変換するための色。
CLASS_COLORS_BGR = {
    0: (0, 0, 0),          # background: black
    1: (0, 180, 0),        # green vegetation: green
    2: (0, 180, 180),      # senescent vegetation: yellow-ish
    3: (0, 120, 255),      # panicle: orange
    4: (255, 255, 255),    # weed: white
    5: (255, 255, 0),      # duckweed: cyan
}

CLASS_NAMES = {
    0: "background",
    1: "green vegetation",
    2: "senescent vegetation",
    3: "panicle",
    4: "weed",
    5: "duckweed",
}


def colorize_label(label: np.ndarray) -> np.ndarray:
    """
    クラスID画像を可視化用カラー画像に変換する。

    Args:
        label:
            shape = (H, W)
            各画素が 0〜5 のクラスID。

    Returns:
        color:
            shape = (H, W, 3)
            OpenCV保存用のBGR画像。
    """
    h, w = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, bgr in CLASS_COLORS_BGR.items():
        color[label == class_id] = bgr

    return color


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    """画像上部にタイトル領域を追加する。"""
    h, w = image_bgr.shape[:2]
    title_h = 36

    canvas = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    canvas[title_h:, :] = image_bgr

    cv2.putText(
        canvas,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def build_comparison_image(
    image_bgr: np.ndarray,
    label_true: np.ndarray,
    label_pred: np.ndarray,
) -> np.ndarray:
    """
    元画像・正解ラベル・予測ラベルを横並びにした確認画像を作る。
    """
    image_resized = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    true_color = colorize_label(label_true)
    pred_color = colorize_label(label_pred)

    left = add_title(image_resized, "RGB")
    center = add_title(true_color, "Ground Truth")
    right = add_title(pred_color, "Prediction")

    return np.hstack([left, center, right])


def print_class_pixels(label: np.ndarray, prefix: str) -> None:
    """クラスごとの画素数を表示する。"""
    values, counts = np.unique(label, return_counts=True)
    count_by_class = {int(v): int(c) for v, c in zip(values, counts)}

    text_parts = []
    for class_id in range(RICESEG_NUM_CLASSES):
        count = count_by_class.get(class_id, 0)
        text_parts.append(f"{class_id}:{count}")

    print(f"{prefix} " + ", ".join(text_parts))


def load_model() -> UNet:
    """checkpointからU-Netを復元する。"""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"checkpoint が存在しません: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    base_channels = int(checkpoint.get("base_channels", 32))
    num_classes = int(checkpoint.get("num_classes", RICESEG_NUM_CLASSES))

    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("[INFO] loaded checkpoint")
    print(f"  path         : {CHECKPOINT_PATH}")
    print(f"  epoch        : {checkpoint.get('epoch', 'unknown')}")
    print(f"  best_val_loss: {checkpoint.get('best_val_loss', 'unknown')}")
    print(f"  image_size   : {checkpoint.get('image_size', 'unknown')}")
    print(f"  base_channels: {base_channels}")
    print(f"  num_classes  : {num_classes}")

    return model


@torch.no_grad()
def predict_one(model: UNet, image_tensor: torch.Tensor) -> np.ndarray:
    """
    1枚分の画像TensorからクラスID予測画像を作る。

    Args:
        image_tensor:
            shape = (3, H, W)
            value = 0.0〜1.0

    Returns:
        pred:
            shape = (H, W)
            dtype = uint8
    """
    x = image_tensor.unsqueeze(0).to(DEVICE)
    logits = model(x)

    if logits.shape[-2:] != image_tensor.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = RiceSEGDataset(
        country="Japan",
        regions=["TKO_1", "TKO_2", "TKO_3"],
        image_size=IMAGE_SIZE,
    )

    model = load_model()

    print("\n[INFO] prediction start")
    print(f"dataset samples: {len(dataset)}")
    print(f"max samples    : {MAX_SAMPLES}")
    print(f"output dir     : {OUTPUT_DIR}")

    n = min(MAX_SAMPLES, len(dataset))

    for index in range(n):
        image_tensor, label_tensor = dataset[index]
        sample = dataset.get_sample_info(index)

        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"画像を読み込めません: {sample.image_path}")

        label_true = label_tensor.cpu().numpy().astype(np.uint8)
        label_pred = predict_one(model, image_tensor)

        comparison = build_comparison_image(
            image_bgr=image_bgr,
            label_true=label_true,
            label_pred=label_pred,
        )

        output_path = OUTPUT_DIR / f"{index:03d}_{sample.image_path.stem}_compare.jpg"
        cv2.imwrite(str(output_path), comparison)

        print(f"\n[{index + 1}/{n}] {sample.image_path.name}")
        print(f"saved: {output_path}")
        print_class_pixels(label_true, "  true:")
        print_class_pixels(label_pred, "  pred:")

    print("\n[INFO] prediction finished")


def print_class_legend() -> None:
    """クラスIDとクラス名の対応を表示する。"""
    print("[CLASS LEGEND]")
    for class_id in range(RICESEG_NUM_CLASSES):
        print(f"  {class_id}: {CLASS_NAMES[class_id]}")


if __name__ == "__main__":
    print_class_legend()
    main()