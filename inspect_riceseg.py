from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path("datasets/RiceSEG")


def main():
    files = list(DATASET_DIR.rglob("*"))
    image_like = [
        p for p in files
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ]

    print(f"total files: {len(files)}")
    print(f"image-like files: {len(image_like)}")

    for path in image_like[:100]:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        print(path)
        print("  shape:", img.shape, "dtype:", img.dtype)

        if img.ndim == 2:
            vals = np.unique(img)
            print("  unique:", vals[:50], "count:", len(vals))

        elif img.ndim == 3:
            h, w = img.shape[:2]
            sample = img.reshape(-1, img.shape[2])
            if len(sample) > 100000:
                sample = sample[:: len(sample) // 100000]
            colors = np.unique(sample, axis=0)
            print("  sampled unique colors:", len(colors))


if __name__ == "__main__":
    main()