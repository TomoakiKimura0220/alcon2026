from pathlib import Path
import csv


def normalize_path(path_str: str) -> Path:
    """
    input.csv内の 000\\P1060881.JPG のようなWindows風パスを
    macOS/Linuxでも読めるPathに変換する。
    """
    return Path(path_str.strip().replace("\\", "/"))


def to_csv_path(path: Path) -> str:
    """
    output.csvには仕様書に合わせて Windows風パスで書く。
    例: 000/P1060881-output.JPG -> 000\\P1060881-output.JPG
    """
    return str(path).replace("/", "\\")


def make_output_path(input_path: Path) -> Path:
    """
    000/P1060881.JPG -> 000/P1060881-output.JPG
    """
    return input_path.with_name(f"{input_path.stem}-output{input_path.suffix}")


def read_input_csv(csv_path: str = "input.csv") -> list[tuple[int, int, Path]]:
    """
    input.csvを読み込む。

    戻り値:
        [(width, height, image_path), ...]
    """
    rows = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        first = next(reader)
        n = int(first[0])

        for row in reader:
            if not row or len(row) < 3:
                continue

            width = int(row[0].strip())
            height = int(row[1].strip())
            image_path = normalize_path(row[2])

            rows.append((width, height, image_path))

    if len(rows) != n:
        print(f"[WARN] input.csvの枚数 N={n} と実際の行数 {len(rows)} が一致しません")

    return rows


def write_output_csv(
    output_rows: list[list],
    csv_path: str = "output.csv",
) -> None:
    """
    output.csvを書き出す。

    output_rowsの各行:
        [width, height, output_image_path, judge, p, w, r]
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 仕様例に合わせて 1行目は N, , とする
        writer.writerow([len(output_rows), "", ""])

        for row in output_rows:
            writer.writerow(row)