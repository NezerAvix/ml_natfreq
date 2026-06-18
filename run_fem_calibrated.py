"""
Скрипт для массового рассчета FEM
"""
import csv
import glob
import os
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import freecad_calc as fc

INPUT_DIR = os.path.join(_SCRIPT_DIR, "models", "generated_calibrated_1000")
OUTPUT_CSV = os.path.join(_SCRIPT_DIR, "results_gen_calibrated.csv")
N = fc.EIGENMODES_COUNT

fc.FIXED_FACE = "Face12"


def _cleanup_fem_tmp():
    for path in glob.glob("/tmp/fcfem_*"):
        shutil.rmtree(path, ignore_errors=True)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Нет каталога {INPUT_DIR}")
        sys.exit(1)

    step_files = sorted(
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".stp", ".step"))
    )
    headers = ["File Name"] + [f"Freq_{i + 1} (Hz)" for i in range(N)] + ["Time (s)"]

    done: set[str] = set()
    if os.path.isfile(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("File Name"):
                    done.add(row["File Name"])

    mode = "a" if done else "w"
    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not done:
            writer.writeheader()

        for i, name in enumerate(step_files, start=1):
            if name in done:
                continue
            path = os.path.join(INPUT_DIR, name)
            freqs, elapsed, _ = fc.process_single_model(path)
            time_str = f"{elapsed:.2f}"
            row = {"File Name": name, "Time (s)": time_str}
            if freqs:
                for j in range(N):
                    row[f"Freq_{j + 1} (Hz)"] = freqs[j] if j < len(freqs) else ""
            else:
                for j in range(N):
                    row[f"Freq_{j + 1} (Hz)"] = "Error/Failed"
            writer.writerow(row)
            f.flush()
            print(f"[INFO] ({i}/{len(step_files)}) {name} ok={bool(freqs)} t={time_str}s", flush=True)
            if i % 25 == 0:
                _cleanup_fem_tmp()

    _cleanup_fem_tmp()
    print(f"[INFO] Готово: {OUTPUT_CSV}", flush=True)


_cleanup_fem_tmp()
main()
