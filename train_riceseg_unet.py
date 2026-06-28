"""
RiceSEG Japan データを使って U-Net を学習するスクリプト。

目的:
    RiceSEG の6クラスセマンティックセグメンテーションモデルを学習する。

入力:
    datasets/RiceSEG/global rice segmentation/Japan/TKO_1
    datasets/RiceSEG/global rice segmentation/Japan/TKO_2
    datasets/RiceSEG/global rice segmentation/Japan/TKO_3

データ分割:
    Japan/TKO_1 + TKO_2 + TKO_3 を混ぜて
    train: 80%
    val  : 20%

出力:
    checkpoints/riceseg_unet_best.pth

学習クラス:
    0: background
    1: green vegetation
    2: senescent vegetation
    3: panicle
    4: weed
    5: duckweed

ALCON用の水稲・雑草・その他への統合は、この学習スクリプトでは行わない。
推論時に別モジュールで統合する。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from riceseg_dataset import RiceSEGDataset, RICESEG_NUM_CLASSES


CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "riceseg_unet_best.pth"

# M1 MacではCPU運用前提。
# PyTorchのMPSを使う場合は、動作が不安定なケースもあるため最初はCPUで確認する。
DEVICE = "cpu"

#
# weedは小さい領域として写ることが多いため、256x256へ縮小すると潰れやすい。
# RiceSEGの元画像サイズである512x512のまま学習し、細かい雑草領域を残す。
IMAGE_SIZE = 512

# CPU学習では重くなるが、まずはbatch_size=2のまま試す。
# メモリ不足や極端に遅い場合は BATCH_SIZE = 1 に下げる。
BATCH_SIZE = 2

# 5 epochでは少数クラスのweedを学習しきれない可能性があるため、少し増やす。
EPOCHS = 5

LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# class_4 weed / class_5 duckweed は画素数が少なく、通常のCrossEntropyLossでは無視されやすい。
# そこで少数クラスの損失を重くし、weed/duckweedを予測する方向へ学習を寄せる。
# まずは控えめな手動重みで試す。
CLASS_WEIGHTS = [0.2, 0.3, 1.0, 1.0, 5.0, 5.0]

# Japan/TKO_1 + TKO_2 + TKO_3 を混ぜて 8:2 にランダム分割する。
# TKO_3だけをvalにすると撮影条件差の影響が大きすぎるため、
# まずはモデル自体がRiceSEG Japanを学習できるか確認する。
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
def create_random_japan_train_val_datasets() -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Japan/TKO_1 + TKO_2 + TKO_3 をすべて混ぜて、8:2でランダム分割する。

    目的:
        地域別分割による分布差の影響をいったん弱め、
        U-NetがRiceSEG Japan全体を学習できるか確認する。
    """
    full_dataset = RiceSEGDataset(
        country="Japan",
        regions=["TKO_1", "TKO_2", "TKO_3"],
        image_size=IMAGE_SIZE,
    )

    train_size = int(len(full_dataset) * TRAIN_RATIO)
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset


class DoubleConv(nn.Module):
    """Conv-BN-ReLU を2回繰り返すU-Net基本ブロック。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    軽量U-Net。

    入力:
        RGB画像 3ch

    出力:
        RiceSEG 6クラスのlogits
        shape = (B, 6, H, W)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32) -> None:
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._concat(d1, e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

    @staticmethod
    def _concat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        U-Netのskip connectionを結合する。

        リサイズの丸め誤差でサイズが1pxずれる場合に備え、
        xをskip側のサイズへ合わせる。
        """
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        return torch.cat([skip, x], dim=1)


def calculate_pixel_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    画素単位Accuracyを計算する。

    注意:
        背景画素が多いと高く見えやすい。
        最終評価ではクラス別IoUも見るべき。
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()

    if total == 0:
        return 0.0

    return correct / total


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    count = 0

    pbar = tqdm(dataloader, desc="train", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        acc = calculate_pixel_accuracy(logits.detach(), labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        count += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    return total_loss / count, total_acc / count


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    count = 0

    pbar = tqdm(dataloader, desc="val", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        acc = calculate_pixel_accuracy(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        count += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    return total_loss / count, total_acc / count


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = create_random_japan_train_val_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = UNet(
        in_channels=3,
        num_classes=RICESEG_NUM_CLASSES,
        base_channels=32,
    ).to(DEVICE)

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print("[INFO] RiceSEG U-Net training")
    print(f"device      : {DEVICE}")
    print(f"image_size  : {IMAGE_SIZE}")
    print(f"batch_size  : {BATCH_SIZE}")
    print(f"epochs      : {EPOCHS}")
    print(f"split       : Japan TKO_1+TKO_2+TKO_3 random {TRAIN_RATIO:.1f}:{1.0 - TRAIN_RATIO:.1f}")
    print(f"random_seed : {RANDOM_SEED}")
    print(f"class_weight: {CLASS_WEIGHTS}")
    print(f"train images: {len(train_dataset)}")
    print(f"val images  : {len(val_dataset)}")
    print(f"checkpoint  : {BEST_MODEL_PATH}")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        print(
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": RICESEG_NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "base_channels": 32,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "train_ratio": TRAIN_RATIO,
                    "random_seed": RANDOM_SEED,
                    "class_weights": CLASS_WEIGHTS,
                },
                BEST_MODEL_PATH,
            )

            print(f"[INFO] saved best model: {BEST_MODEL_PATH}")

    print("\n[INFO] training finished")
    print(f"[INFO] best_val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()