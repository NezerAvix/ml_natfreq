"""
Дозаполнение results_gen_enhanced.csv для строк Error/Failed
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

fc.FIXED_FACE = "auto"

INPUT_DIR = os.path.join(_SCRIPT_DIR, "models", "generated_enhanced_1000")
CSV_PATH = os.path.join(_SCRIPT_DIR, "results_gen_enhanced.csv")
N = fc.EIGENMODES_COUNT


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    headers = fieldnames or (
        ["File Name"] + [f"Freq_{i + 1} (Hz)" for i in range(N)] + ["Time (s)"]
    )
    todo = [r for r in rows if r.get("Freq_1 (Hz)") == "Error/Failed"]
    print(f"[INFO] Дозапуск FEM: {len(todo)} / {len(rows)} моделей")

    for i, row in enumerate(todo, start=1):
        name = row["File Name"]
        path = os.path.join(INPUT_DIR, name)
        freqs, elapsed, _ = fc.process_single_model(path)
        time_str = f"{elapsed:.2f}"
        if freqs:
            for j in range(len(freqs)):
                row[f"Freq_{j + 1} (Hz)"] = freqs[j]
            for j in range(len(freqs), N):
                row[f"Freq_{j + 1} (Hz)"] = ""
        else:
            for j in range(N):
                row[f"Freq_{j + 1} (Hz)"] = "Error/Failed"
        row["Time (s)"] = time_str
        if i % 20 == 0 or i == len(todo):
            print(f"[INFO] {i}/{len(todo)} {name} ok={bool(freqs)}")
        # периодически сохраняем
        if i % 50 == 0:
            _write_csv(CSV_PATH, headers, rows)
            _cleanup_fem_tmp()

    _write_csv(CSV_PATH, headers, rows)
    ok = sum(1 for r in rows if r.get("Freq_1 (Hz)") != "Error/Failed")
    print(f"[INFO] Готово: {ok}/{len(rows)} успешных → {CSV_PATH}")


def _cleanup_fem_tmp():
    for path in glob.glob("/tmp/fcfem_*"):
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


main()
