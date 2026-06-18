# Сравнение предсказаний с FEM
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def mape_report(fem_csv: Path, pred_csv: Path) -> dict:
    with open(fem_csv, newline="", encoding="utf-8") as f:
        fem = {r["File Name"]: r for r in csv.DictReader(f)}
    with open(pred_csv, newline="", encoding="utf-8") as f:
        pred = {r["File Name"]: r for r in csv.DictReader(f)}

    errs: list[float] = []
    by_mode: dict[int, list[float]] = defaultdict(list)
    times: list[float] = []
    n = 0

    for name, row in fem.items():
        if name not in pred:
            continue
        try:
            ff = [float(row[f"Freq_{i} (Hz)"]) for i in range(1, 13)]
            pp = [float(pred[name][f"Freq_{i} (Hz)"]) for i in range(1, 13)]
            times.append(float(pred[name]["Time (s)"]))
        except (ValueError, KeyError):
            continue
        n += 1
        for i in range(12):
            e = abs(ff[i] - pp[i]) / ff[i]
            errs.append(e)
            by_mode[i + 1].append(e)

    return {
        "models": n,
        "mape_pct": 100.0 * sum(errs) / max(len(errs), 1),
        "time_total_s": sum(times),
        "time_mean_s": sum(times) / max(len(times), 1),
        "by_mode": {i: 100.0 * sum(v) / len(v) for i, v in sorted(by_mode.items())},
    }


def main():
    pred = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results-predicted-12.csv"
    fem = ROOT / "results_init2.csv"
    if len(sys.argv) > 2:
        fem = Path(sys.argv[2])
    r = mape_report(fem, pred)
    print(f"FEM: {fem.name}")
    print(f"Предсказания: {pred.name}")
    print(f"Моделей: {r['models']}")
    print(f"Средний MAPE (12 частот): {r['mape_pct']:.2f}%")
    print(f"Время предсказания: сумма {r['time_total_s']:.2f} с, среднее {r['time_mean_s']:.3f} с")
    for i in range(1, 13):
        print(f"  F{i}: {r['by_mode'].get(i, 0):.1f}%")


if __name__ == "__main__":
    main()
