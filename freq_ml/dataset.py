"""
Датасет: STEP + CSV с частотами
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from freq_ml.constants import (
    FREQ_COLUMNS,
    GENERATED_SAMPLE_WEIGHT,
    N_FREQ,
    TRACEPARTS_SAMPLE_WEIGHT,
)
from freq_ml.geometry import (
    GEOMETRY_FEATURE_VERSION,
    GeometrySample,
    augment_points,
    sample_point_cloud,
)
from freq_ml.screw_params import load_specs_tables, lookup_specs


def _parse_freq_row(row: dict) -> np.ndarray | None:
    values = []
    for col in FREQ_COLUMNS:
        raw = row.get(col, "")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return None
    arr = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        return None
    return arr


def _is_traceparts_path(stp_path: str, models_dir: Path) -> bool:
    try:
        rel = Path(stp_path).resolve().relative_to(models_dir.resolve())
        return len(rel.parts) == 1
    except ValueError:
        return False


def build_combined_manifest(
    root: str | Path,
    extra_sources: list[tuple[str | Path, str | Path]] | None = None,
) -> list[tuple[str, str, np.ndarray, bool]]:
    root = Path(root)
    traceparts_dir = (root / "models").resolve()
    entries: list[tuple[str, str, np.ndarray, bool]] = []
    sources = [
        (root / "results_init2.csv", root / "models"),
        (root / "results_gen_calibrated.csv", root / "models" / "generated_calibrated_1000"),
    ]
    if extra_sources:
        sources.extend(extra_sources)
    for csv_path, models_dir in sources:
        models_dir = models_dir.resolve()
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = row.get("File Name", "").strip()
                freqs = _parse_freq_row(row)
                if not name or freqs is None:
                    continue
                stp = models_dir / name
                if stp.is_file():
                    stp_resolved = str(stp.resolve())
                    is_tp = _is_traceparts_path(stp_resolved, traceparts_dir)
                    entries.append((stp_resolved, name, freqs, is_tp))
    return entries


def split_traceparts_val(
    entries: list[tuple[str, str, np.ndarray, bool]],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    tp_indices = [i for i, e in enumerate(entries) if e[3]]
    gen_indices = [i for i, e in enumerate(entries) if not e[3]]

    rng.shuffle(tp_indices)
    n_val = max(1, int(len(tp_indices) * val_fraction))
    val_idx = tp_indices[:n_val]
    train_idx = tp_indices[n_val:] + gen_indices
    return train_idx, val_idx


class FrequencyDataset(Dataset):
    def __init__(
        self,
        entries: list[tuple[str, str, np.ndarray, bool]],
        cache_dir: str | Path | None,
        n_points: int = 1024,
        log_targets: bool = True,
        build_cache: bool = False,
        augment: bool = False,
        augment_seed: int | None = None,
        specs_root: str | Path | None = None,
    ):
        self.entries = entries
        self.n_points = n_points
        self.log_targets = log_targets
        self.augment = augment
        self.augment_seed = augment_seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.specs_tables = load_specs_tables(specs_root or Path.cwd())
        if self.cache_dir and build_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_cache()

    def _cache_path(self, stp_path: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha1(
            f"{stp_path}|n={self.n_points}|v={GEOMETRY_FEATURE_VERSION}".encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.npz"

    def _build_cache(self) -> None:
        assert self.cache_dir is not None
        for i, (stp_path, name, _, _) in enumerate(self.entries):
            cache = self._cache_path(stp_path)
            assert cache is not None
            if cache.is_file():
                continue
            params = lookup_specs(name, self.specs_tables)
            geom = sample_point_cloud(stp_path, self.n_points, seed=0, screw_params=params)
            np.savez_compressed(cache, points=geom.points, scalars=geom.scalars)
            if (i + 1) % 50 == 0:
                print(f"  кэш: {i + 1}/{len(self.entries)}")

    def _load_geometry(self, stp_path: str, index: int, name: str) -> GeometrySample:
        cache = self._cache_path(stp_path)
        if cache and cache.is_file():
            data = np.load(cache)
            geom = GeometrySample(points=data["points"], scalars=data["scalars"])
        else:
            params = lookup_specs(name, self.specs_tables)
            geom = sample_point_cloud(stp_path, self.n_points, seed=0, screw_params=params)

        if self.augment:
            seed = (self.augment_seed if self.augment_seed is not None else 0) + index * 9973
            rng = np.random.default_rng(seed)
            points = augment_points(geom.points, rng)
            return GeometrySample(points=points, scalars=geom.scalars)
        return geom

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        stp_path, name, freqs, is_traceparts = self.entries[index]
        geom = self._load_geometry(stp_path, index, name)
        target = np.log(freqs) if self.log_targets else freqs.astype(np.float32)
        weight = TRACEPARTS_SAMPLE_WEIGHT if is_traceparts else GENERATED_SAMPLE_WEIGHT
        return {
            "points": torch.from_numpy(geom.points.copy()),
            "scalars": torch.from_numpy(geom.scalars),
            "target": torch.from_numpy(target),
            "freqs_hz": torch.from_numpy(freqs),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "is_traceparts": is_traceparts,
            "name": name,
            "stp_path": stp_path,
        }
