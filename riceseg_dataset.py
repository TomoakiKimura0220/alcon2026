"""
RiceSEG を PyTorch の Dataset として読み込むためのモジュール。

現段階では、ALCON画像に近いと考えられる Japan データのみを対象にする。

想定するRiceSEGの構造:
    datasets/RiceSEG/global rice segmentation/Japan/
    ├── TKO_1/
    │   ├── rgb/
    │   └── label/
    ├── TKO_2/
    │   ├── rgb/
    │   └── label/
    └── TKO_3/
        ├── rgb/
        └── label/

確認済み仕様:
    - rgb と label はファイル名の stem が一致する。
    - rgb は 512x512 のカラー画像。
    - label は 512x512 の uint8 グレースケール画像。
    - label の画素値がそのままクラスIDになっている。

RiceSEG 6クラス:
    0: background
    1: green vegetation
    2: senescent vegetation
    3: panicle
    4: weed
    5: duckweed

ALCON用3クラス:
    0: other = background
    1: rice  = green vegetation + senescent vegetation + panicle
    2: weed  = weed + duckweed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


RICESEG_NUM_CLASSES = 6
ALCON_NUM_CLASSES = 3


@dataclass(frozen=True)
class RiceSEGSample:
    """RGB画像とラベル画像のペア情報。"""

    image_path: Path
    label_path: Path
    country: str
    region: str


class RiceSEGDataset(Dataset):
    """
    RiceSEG の rgb/label ペアを返す Dataset。

    Returns:
        image_tensor:
            shape = (3, H, W)
            dtype = torch.float32
            value range = 0.0〜1.0

        label_tensor:
            shape = (H, W)
            dtype = torch.long
            value = クラスID
    """

    def __init__(
        self,
        root_dir: str | Path = "datasets/RiceSEG/global rice segmentation",
        country: str = "Japan",
        regions: list[str] | None = None,
        image_size: int | None = None,
        label_mode: str = "riceseg",
    ) -> None:
        """
        Args:
            root_dir:
                RiceSEG の "global rice segmentation" ディレクトリ。
            country:
                使用する国名。最初は Japan を想定する。
            regions:
                使用する地域名のリスト。
                例: ["TKO_1", "TKO_2"]
                None の場合は country 配下の全地域を使う。
            image_size:
                リサイズ後の1辺のサイズ。
                None の場合は元画像サイズのまま使う。
                RiceSEGのcrop画像は基本512x512なので、通常はNoneでよい。
            label_mode:
                "riceseg" の場合は RiceSEG の6クラスIDをそのまま返す。
                "alcon" の場合は ALCON用の3クラスへ変換して返す。
        """
        self.root_dir = Path(root_dir)
        self.country = country
        self.country_dir = self.root_dir / country
        self.image_size = image_size
        self.label_mode = label_mode

        if self.label_mode not in {"riceseg", "alcon"}:
            raise ValueError(f"label_mode は 'riceseg' または 'alcon' を指定してください: {self.label_mode}")

        if not self.country_dir.exists():
            raise FileNotFoundError(f"country_dir が存在しません: {self.country_dir}")

        if regions is None:
            regions = sorted([p.name for p in self.country_dir.iterdir() if p.is_dir()])

        self.regions = regions
        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(
                f"rgb/label ペアが見つかりません: country={country}, regions={regions}"
            )

    def _collect_samples(self) -> list[RiceSEGSample]:
        samples: list[RiceSEGSample] = []

        for region in self.regions:
            region_dir = self.country_dir / region
            rgb_dir = region_dir / "rgb"
            label_dir = region_dir / "label"

            if not rgb_dir.exists():
                print(f"[WARN] rgb_dir が存在しません: {rgb_dir}")
                continue

            if not label_dir.exists():
                print(f"[WARN] label_dir が存在しません: {label_dir}")
                continue

            rgb_files = sorted(
                [
                    p
                    for p in rgb_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            label_files = sorted(
                [
                    p
                    for p in label_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
            )

            label_by_stem = {p.stem: p for p in label_files}

            for image_path in rgb_files:
                label_path = label_by_stem.get(image_path.stem)
                if label_path is None:
                    print(f"[WARN] 対応するlabelがありません: {image_path}")
                    continue

                samples.append(
                    RiceSEGSample(
                        image_path=image_path,
                        label_path=label_path,
                        country=self.country,
                        region=region,
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"画像を読み込めません: {sample.image_path}")

        label = cv2.imread(str(sample.label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"ラベルを読み込めません: {sample.label_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if image_rgb.shape[:2] != label.shape[:2]:
            raise ValueError(
                "画像とラベルのサイズが一致しません: "
                f"image={sample.image_path} {image_rgb.shape[:2]}, "
                f"label={sample.label_path} {label.shape[:2]}"
            )

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            image_rgb = cv2.resize(image_rgb, size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)

        if self.label_mode == "alcon":
            label_3class = np.zeros_like(label, dtype=np.uint8)
            label_3class[np.isin(label, [1, 2, 3])] = 1
            label_3class[np.isin(label, [4, 5])] = 2
            label = label_3class
            num_classes = ALCON_NUM_CLASSES
        else:
            num_classes = RICESEG_NUM_CLASSES

        if label.min() < 0 or label.max() >= num_classes:
            raise ValueError(
                f"ラベル値が想定範囲外です: {sample.label_path}, "
                f"min={label.min()}, max={label.max()}, label_mode={self.label_mode}"
            )

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.from_numpy(label.astype(np.int64)).long()

        return image_tensor, label_tensor

    def get_sample_info(self, index: int) -> RiceSEGSample:
        """デバッグや可視化用に、index番目のファイルパス情報を返す。"""
        return self.samples[index]


def create_japan_train_val_datasets(
    image_size: int | None = None,
    label_mode: str = "riceseg",
) -> tuple[RiceSEGDataset, RiceSEGDataset]:
    """
    最初の検証用に Japan データだけで train/val を作る。

    train:
        TKO_1 + TKO_2 = 604枚

    val:
        TKO_3 = 100枚
    """
    train_dataset = RiceSEGDataset(
        country="Japan",
        regions=["TKO_1", "TKO_2"],
        image_size=image_size,
        label_mode=label_mode,
    )

    val_dataset = RiceSEGDataset(
        country="Japan",
        regions=["TKO_3"],
        image_size=image_size,
        label_mode=label_mode,
    )

    return train_dataset, val_dataset


def main() -> None:
    """Datasetが正しく読めるかの簡易確認。"""
    for label_mode in ["riceseg", "alcon"]:
        print(f"\n[label_mode={label_mode}]")
        train_dataset, val_dataset = create_japan_train_val_datasets(label_mode=label_mode)

        print(f"train samples: {len(train_dataset)}")
        print(f"val samples  : {len(val_dataset)}")

        image, label = train_dataset[0]
        sample = train_dataset.get_sample_info(0)

        print("\n[first sample]")
        print(f"image path: {sample.image_path}")
        print(f"label path: {sample.label_path}")
        print(f"image tensor: shape={tuple(image.shape)}, dtype={image.dtype}, min={image.min():.3f}, max={image.max():.3f}")
        print(f"label tensor: shape={tuple(label.shape)}, dtype={label.dtype}")
        print(f"label unique: {torch.unique(label).tolist()}")


if __name__ == "__main__":
    main()