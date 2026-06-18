"""
Профиль R(Y) для STEP: головка (min Y), торец резьбы (max Y)
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from pathlib import Path

import gmsh
import numpy as np

ROOT = Path(__file__).resolve().parent


def profile(stp_path: str, n_bins: int = 150) -> dict:
    gmsh.initialize(["-nopopup"])
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("p")
        gmsh.merge(os.path.abspath(stp_path))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2)
        gmsh.model.mesh.generate(2)
        _, coord, _ = gmsh.model.mesh.getNodes()
        pts = np.asarray(coord, dtype=np.float64).reshape(-1, 3)
    finally:
        gmsh.finalize()

    y = pts[:, 1]
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
    ymin, ymax = float(y.min()), float(y.max())
    length = ymax - ymin
    edges = np.linspace(ymin, ymax, n_bins + 1)
    centers = []
    r_max = []
    for i in range(n_bins):
        m = (y >= edges[i]) & (y < edges[i + 1])
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        r_max.append(float(r[m].max()) if m.any() else 0.0)
    centers = np.array(centers)
    r_max = np.array(r_max)

    upper = y > ymin + 0.55 * length
    lower = y < ymin + 0.35 * length
    r_shank_med = float(np.median(r[upper])) if upper.any() else float("nan")
    r_head_max = float(r[lower].max()) if lower.any() else float("nan")
    wide = (r > r_shank_med * 1.12) & lower if np.isfinite(r_shank_med) else lower
    head_len = float(wide.sum() and (y[wide].max() - y[wide].min())) if wide.any() else float("nan")

    tip = y > ymax - 0.06 * length
    r_tip = float(r[tip].max()) if tip.any() else float("nan")
    band = (y > ymax - 0.14 * length) & (y < ymax - 0.02 * length)
    r_shank_top = float(r[band].max()) if band.any() else float("nan")
    chamfer_axial = float("nan")
    if np.isfinite(r_shank_top) and np.isfinite(r_tip) and r_shank_top > r_tip + 0.05:
        taper = (y > ymax - 0.2 * length) & (r < r_shank_top * 0.97)
        if taper.any():
            chamfer_axial = float(ymax - y[taper].min())

    return {
        "file": Path(stp_path).name,
        "length_mm": length,
        "r_shank_mm": r_shank_med,
        "r_head_mm": r_head_max,
        "head_len_mm": head_len,
        "r_tip_mm": r_tip,
        "r_shank_top_mm": r_shank_top,
        "chamfer_axial_mm": chamfer_axial,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models_dir", type=Path, nargs="?", default=ROOT / "models")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "traceparts_profile.csv")
    args = p.parse_args()
    files = sorted(args.models_dir.glob("*.stp"))
    if args.limit:
        files = files[: args.limit]
    rows = []
    for f in files:
        try:
            rows.append(profile(str(f)))
        except Exception as exc:
            print(f"[WARN] {f.name}: {exc}")
    if not rows:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for key in ("head_len_mm", "chamfer_axial_mm", "r_tip_mm", "r_shank_mm"):
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        if vals:
            print(f"{key}: median={st.median(vals):.2f} mean={st.mean(vals):.2f}")
    print(f"Wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
