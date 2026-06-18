"""
Номинальные размеры винта: random_specs.csv и оценка по профилю R(Y)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ScrewParams = tuple[float, float, float, float]


def load_specs_tables(root: str | Path) -> dict[str, ScrewParams]:
    root = Path(root)
    tables: dict[str, ScrewParams] = {}
    tp_specs = root / "data" / "traceparts_specs.csv"
    if tp_specs.is_file():
        with open(tp_specs, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("file") or "").strip()
                if not name:
                    continue
                try:
                    r_shank = float(row["r_shank_mm"])
                    r_head = float(row["r_head_mm"])
                    head_h = float(row["head_len_mm"])
                    bbox_len = float(row["length_mm"])
                    tables[name] = (
                        2.0 * r_shank,
                        2.0 * r_head,
                        max(bbox_len - head_h, 1.0),
                        head_h,
                    )
                except (KeyError, ValueError):
                    continue

    for rel in (
        "models/generated_calibrated_1000/random_specs.csv",
        "models/generated_calibrated_1000/manifest.csv",
    ):
        path = root / rel
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("name") or row.get("File Name") or "").strip()
                if not name:
                    continue
                if not name.lower().endswith(".stp"):
                    name = f"{name}.stp"
                try:
                    tables[name] = (
                        float(row["thread_d_mm"]),
                        float(row["head_d_mm"]),
                        float(row["length_mm"]),
                        float(row["head_h_mm"]),
                    )
                except (KeyError, ValueError):
                    continue
    return tables


def lookup_specs(file_name: str, tables: dict[str, ScrewParams]) -> ScrewParams | None:
    return tables.get(file_name) or tables.get(Path(file_name).stem)


def estimate_params_from_points(pts: np.ndarray) -> ScrewParams:
    """
    Оценка по облаку узлов (ориентация TraceParts: головка min Y, резьба max Y).
    length_mm — номинальная длина стержня под головкой (L).
    """
    y = pts[:, 1]
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
    ymin, ymax = float(y.min()), float(y.max())
    length_bbox = ymax - ymin
    if length_bbox < 1e-6:
        return (6.0, 10.0, 20.0, 6.0)

    upper = y > ymax - 0.38 * length_bbox
    lower = y < ymin + 0.38 * length_bbox
    r_shank = float(np.median(r[upper])) if upper.any() else float(np.median(r))
    r_head = float(np.max(r[lower])) if lower.any() else r_shank * 1.4

    wide = lower & (r > r_shank * 1.1)
    if wide.any():
        head_h = float(y[wide].max() - y[wide].min())
    else:
        head_h = 0.18 * length_bbox

    head_h = max(head_h, 0.05 * length_bbox)
    head_h = min(head_h, 0.45 * length_bbox)
    length_mm = max(length_bbox - head_h, 1.0)
    thread_d = max(2.0 * r_shank, 2.0)
    head_d = max(2.0 * r_head, thread_d + 0.5)
    return (thread_d, head_d, length_mm, head_h)


def build_param_scalars(params: ScrewParams) -> np.ndarray:
    thread_d, head_d, length_mm, head_h = params
    d = thread_d + 1e-6
    L = length_mm + 1e-6
    return np.array(
        [
            np.log1p(thread_d),
            np.log1p(head_d),
            np.log1p(length_mm),
            np.log1p(head_h),
            length_mm / d,
            head_h / L,
            head_d / d,
        ],
        dtype=np.float32,
    )
