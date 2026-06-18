"""
Изучение влияния параметров сетки на итоговые частоты FEM
"""

from __future__ import annotations

import csv
import glob
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import freecad_calc as fc
from freq_ml.screw_params import load_specs_tables, lookup_specs

ROOT = Path(_SCRIPT_DIR)
OUT_CSV = ROOT / "data" / "mesh_study_results.csv"

CASES = [
    ("TP_M12_long", "models/0_100.stp", "Face4,Face7"),
    ("TP_M6_med", "models/0_10.stp", "Face4,Face7"),
    ("TP_M3_short", "models/0_11.stp", "Face4,Face7"),
    ("TP_M8", "models/0_50.stp", "Face4,Face7"),
    ("GEN_M3", "models/generated_calibrated_1000/rand_0001.stp", "Face12"),
    ("GEN_M6", "models/generated_calibrated_1000/rand_0500.stp", "Face12"),
    ("GEN_M20", "models/generated_calibrated_1000/rand_0800.stp", "Face12"),
    ("GEN_M20_long", "models/generated_calibrated_1000/rand_0200.stp", "Face12"),
]


def _mesh_variants(thread_d: float) -> list[tuple[str, float, float, float]]:
    d = max(thread_d, 2.0)
    return [
        ("baseline_auto", 0.0, 0.0, 12),
        ("h08d", 0.08 * d, 0.04 * d, 12),
        ("h10d", 0.10 * d, 0.05 * d, 12),
        ("h12d", 0.12 * d, 0.06 * d, 12),
        ("h15d", 0.15 * d, 0.075 * d, 12),
        ("fixed_0.8mm", 0.8, 0.8, 12),
        ("fixed_1.2mm", 1.2, 1.2, 12),
    ]


def _cleanup():
    for path in glob.glob("/tmp/fcfem_*"):
        shutil.rmtree(path, ignore_errors=True)


def main():
    specs = load_specs_tables(ROOT)
    rows: list[dict] = []
    n_modes = fc.EIGENMODES_COUNT

    for case_label, rel_stp, fix in CASES:
        stp = ROOT / rel_stp
        if not stp.is_file():
            print(f"[WARN] нет файла {stp}")
            continue
        name = stp.name
        params = lookup_specs(name, specs)
        thread_d = params[0] if params else 6.0
        print(f"\n=== {case_label} ({name}) d={thread_d:.2f} mm ===", flush=True)

        for var_label, h_max, h_min, curv in _mesh_variants(thread_d):
            _cleanup()
            freqs, elapsed, stats = fc.process_single_model(
                str(stp),
                mesh_h_max=h_max,
                mesh_h_min=h_min,
                mesh_curvature=curv,
                fixed_face=fix,
                quiet=True,
            )
            row = {
                "case": case_label,
                "file": name,
                "thread_d_mm": round(thread_d, 3),
                "variant": var_label,
                "h_max": round(h_max, 4),
                "h_min": round(h_min, 4),
                "curvature": curv,
                "nodes": stats.get("nodes", ""),
                "tets": stats.get("tets", ""),
                "time_s": round(elapsed, 2),
                "ok": bool(freqs),
            }
            if freqs:
                for i in range(n_modes):
                    row[f"F{i + 1}"] = round(freqs[i], 2) if i < len(freqs) else ""
            rows.append(row)
            f1 = freqs[0] if freqs else None
            f1_str = f"{f1:.0f}" if f1 is not None else "FAIL"
            nodes = stats.get("nodes", "?")
            print(
                f"  {var_label:14s} h_max={h_max:5.2f} nodes={nodes!s:>6} "
                f"F1={f1_str} Hz  {elapsed:.1f}s",
                flush=True,
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "file",
        "thread_d_mm",
        "variant",
        "h_max",
        "h_min",
        "curvature",
        "nodes",
        "tets",
        "time_s",
        "ok",
    ] + [f"F{i}" for i in range(1, n_modes + 1)]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[INFO] Записано {len(rows)} строк → {OUT_CSV}", flush=True)
    _cleanup()


if __name__ in ("__main__", "mesh_study"):
    main()
