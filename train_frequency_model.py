"""
Обучение нейросети для предсказания первых 12 собственных частот по геометрии STEP.

Признаки v4: нормали, FPS, 2048 точек, param scalars (d, dk, L, k); attention-pool.
Inference: --tta-views 8 (усреднение по поворотам вокруг Y).

Запуск:
  .venv/bin/python train_frequency_model.py train --build-cache
  .venv/bin/python train_frequency_model.py batch --models-dir models
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from freq_ml.constants import FREQ_COLUMNS, N_FREQ
from freq_ml.dataset import FrequencyDataset, build_combined_manifest, split_traceparts_val
from freq_ml.model import FrequencyPredictor

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "geometry_cache_v4"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "freq_predictor_12_best.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relative_errors_hz(pred_log: torch.Tensor, freqs_hz: torch.Tensor) -> torch.Tensor:
    pred_hz = torch.exp(pred_log)
    return torch.abs(pred_hz - freqs_hz) / freqs_hz.clamp_min(1e-6)


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    delta: float = 0.15,
) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quad
    per_elem = 0.5 * quad**2 + delta * linear
    w = weight.unsqueeze(1)
    return (per_elem * w).sum() / w.sum().clamp_min(1e-6) / pred.shape[1]


def weighted_log_mape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    pred_hz = torch.exp(pred)
    target_hz = torch.exp(target)
    rel = (pred_hz - target_hz).abs() / target_hz.clamp_min(1e-6)
    w = weight.unsqueeze(1)
    return (rel * w).sum() / w.sum().clamp_min(1e-6) / pred.shape[1]


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    mape_weight: float = 0.35,
) -> torch.Tensor:
    huber = weighted_huber_loss(pred, target, weight)
    mape = weighted_log_mape_loss(pred, target, weight)
    return (1.0 - mape_weight) * huber + mape_weight * mape


@torch.no_grad()
def evaluate(
    model: FrequencyPredictor,
    loader: DataLoader,
    device: torch.device,
    traceparts_only: bool = False,
) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mape_sum = 0.0
    n_samples = 0
    for batch in loader:
        mask = torch.ones(batch["points"].shape[0], dtype=torch.bool)
        if traceparts_only:
            mask = batch["is_traceparts"]

        if not mask.any():
            continue

        points = batch["points"][mask].to(device)
        scalars = batch["scalars"][mask].to(device)
        target = batch["target"][mask].to(device)
        freqs = batch["freqs_hz"][mask].to(device)

        pred = model(points, scalars)
        mse_sum += nn.functional.mse_loss(pred, target, reduction="sum").item()
        mape_sum += relative_errors_hz(pred, freqs).sum().item()
        n_samples += mask.sum().item() * N_FREQ

    denom = max(n_samples, 1)
    return {"mse_log": mse_sum / denom, "mape": mape_sum / denom}


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    entries = build_combined_manifest(ROOT)
    if args.finetune_traceparts:
        entries = [e for e in entries if e[3]]
        print(f"Режим finetune: только TraceParts, {len(entries)} образцов")
    n_tp = sum(1 for e in entries if e[3])
    n_gen = len(entries) - n_tp
    print(f"Образцов: {len(entries)} (TraceParts: {n_tp}, generated: {n_gen})")

    train_idx, val_idx = split_traceparts_val(entries, args.val_fraction, args.seed)
    print(f"Train: {len(train_idx)}, val (только TraceParts): {len(val_idx)}")

    full_ds = FrequencyDataset(
        entries,
        cache_dir=args.cache_dir,
        n_points=args.n_points,
        build_cache=args.build_cache,
        augment=False,
        specs_root=ROOT,
    )
    train_ds = Subset(
        FrequencyDataset(
            entries,
            cache_dir=args.cache_dir,
            n_points=args.n_points,
            augment=True,
            augment_seed=args.seed,
            specs_root=ROOT,
        ),
        train_idx,
    )
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = FrequencyPredictor(n_freq=N_FREQ).to(device)
    if args.finetune_traceparts and args.checkpoint.is_file():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Загружен чекпоинт: {args.checkpoint}")

    lr = args.lr * (0.25 if args.finetune_traceparts else 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_mape = float("inf")
    stale_epochs = 0
    save_path = args.checkpoint
    if args.finetune_traceparts:
        save_path = args.checkpoint.with_name(args.checkpoint.stem + "_finetuned.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            points = batch["points"].to(device)
            scalars = batch["scalars"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(points, scalars)
            loss = combined_loss(pred, target, weight, mape_weight=args.mape_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, traceparts_only=True)
        train_loss /= max(n_batches, 1)

        if val_metrics["mape"] < best_mape:
            best_mape = val_metrics["mape"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_points": args.n_points,
                    "n_freq": N_FREQ,
                    "log_targets": True,
                    "point_dim": model.point_dim,
                    "n_scalars": model.head[0].in_features - model.width,
                    "best_val_mape_traceparts": best_mape,
                },
                save_path,
            )
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"Эпоха {epoch:4d}/{args.epochs} | "
                f"train loss={train_loss:.4f} | "
                f"val TraceParts MAPE={val_metrics['mape']*100:.2f}%"
            )

        if stale_epochs >= args.patience:
            print(f"Early stop на эпохе {epoch} (patience={args.patience})")
            break

    print(f"Лучший val MAPE (TraceParts): {best_mape*100:.2f}%")
    print(f"Модель: {save_path}")


def _load_model(checkpoint: Path, device: torch.device, n_points: int) -> tuple[FrequencyPredictor, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FrequencyPredictor(
        n_freq=ckpt.get("n_freq", N_FREQ),
        n_scalars=ckpt.get("n_scalars", 15),
        point_dim=ckpt.get("point_dim", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _tta_rotations(n: int) -> list[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rots = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rots.append(np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32))
    return rots


def _apply_rotation_feat(feat: np.ndarray, rot: np.ndarray) -> np.ndarray:
    pos = feat[:, :3] @ rot.T
    nrm = feat[:, 3:6] @ rot.T
    return np.concatenate([pos, nrm], axis=1).astype(np.float32)


@torch.no_grad()
def predict_one(
    model: FrequencyPredictor,
    stp_path: Path,
    device: torch.device,
    n_points: int,
    cache_dir: Path | None = None,
    tta_views: int = 0,
) -> tuple[np.ndarray, float]:
    from freq_ml.geometry import sample_point_cloud

    t0 = time.perf_counter()
    if cache_dir:
        dummy = np.ones(N_FREQ, dtype=np.float32)
        entries = [(str(stp_path.resolve()), stp_path.name, dummy, True)]
        ds = FrequencyDataset(
            entries, cache_dir=cache_dir, n_points=n_points, build_cache=False, specs_root=ROOT
        )
        item = ds[0]
        feat = item["points"].numpy()
        scalars_t = item["scalars"].unsqueeze(0).to(device)
    else:
        geom = sample_point_cloud(str(stp_path), n_points=n_points, seed=0)
        feat = geom.points
        scalars_t = torch.from_numpy(geom.scalars).unsqueeze(0).to(device)

    if tta_views <= 1:
        points = torch.from_numpy(feat).unsqueeze(0).to(device)
        pred_hz = torch.exp(model(points, scalars_t)[0]).cpu().numpy()
    else:
        preds = []
        for rot in _tta_rotations(tta_views):
            rotated = _apply_rotation_feat(feat, rot)
            points = torch.from_numpy(rotated).unsqueeze(0).to(device)
            preds.append(torch.exp(model(points, scalars_t)[0]))
        pred_hz = torch.stack(preds, dim=0).mean(dim=0).cpu().numpy()

    return pred_hz, time.perf_counter() - t0


@torch.no_grad()
def batch_predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(args.checkpoint, device, args.n_points)
    n_points = ckpt.get("n_points", args.n_points)

    models_dir = Path(args.models_dir).resolve()
    stp_files = sorted(models_dir.glob("*.stp"))
    if not stp_files:
        raise SystemExit(f"Нет .stp в {models_dir}")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["File Name", *FREQ_COLUMNS, "Time (s)"]

    print(f"Устройство: {device}, точек: {n_points}")
    print(f"Моделей: {len(stp_files)} → {out_path}")

    rows: list[dict[str, str | float]] = []
    for i, stp in enumerate(stp_files, start=1):
        try:
            pred_hz, elapsed = predict_one(
                model,
                stp,
                device,
                n_points,
                cache_dir=args.cache_dir,
                tta_views=args.tta_views,
            )
            row: dict[str, str | float] = {"File Name": stp.name, "Time (s)": round(elapsed, 3)}
            for j, col in enumerate(FREQ_COLUMNS):
                row[col] = round(float(pred_hz[j]), 2) if j < len(pred_hz) else ""
        except Exception as exc:
            row = {"File Name": stp.name, "Time (s)": 0.0}
            for col in FREQ_COLUMNS:
                row[col] = "Error/Failed"
            print(f"  [ERROR] {stp.name}: {exc}")
        rows.append(row)
        if i % 10 == 0 or i == len(stp_files):
            print(f"  {i}/{len(stp_files)}  {stp.name}  {row['Time (s)']}s")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_time = sum(float(r["Time (s)"]) for r in rows if r.get("Freq_1 (Hz)") != "Error/Failed")
    print(f"Готово. Суммарное время: {total_time:.1f}s")


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(args.checkpoint, device, args.n_points)
    stp = Path(args.stp).resolve()
    pred_hz, elapsed = predict_one(
        model,
        stp,
        device,
        ckpt.get("n_points", args.n_points),
        cache_dir=args.cache_dir,
        tta_views=args.tta_views,
    )
    print(f"Файл: {stp.name}  ({elapsed:.2f}s)")
    for i, f in enumerate(pred_hz, start=1):
        print(f"  Freq_{i}: {f:.2f} Hz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Предсказание собственных частот по STEP")
    sub = p.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="обучить модель")
    train_p.add_argument("--epochs", type=int, default=400)
    train_p.add_argument("--patience", type=int, default=60)
    train_p.add_argument("--batch-size", type=int, default=48)
    train_p.add_argument("--lr", type=float, default=8e-4)
    train_p.add_argument("--val-fraction", type=float, default=0.18)
    train_p.add_argument("--n-points", type=int, default=2048)
    train_p.add_argument("--mape-loss-weight", type=float, default=0.35)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--num-workers", type=int, default=0)
    train_p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    train_p.add_argument("--build-cache", action="store_true")
    train_p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train_p.add_argument(
        "--finetune-traceparts",
        action="store_true",
        help="дообучить только на TraceParts (/models), загрузив --checkpoint",
    )

    pred_p = sub.add_parser("predict", help="предсказать для одного STP")
    pred_p.add_argument("stp", type=Path)
    pred_p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    pred_p.add_argument("--n-points", type=int, default=2048)
    pred_p.add_argument("--tta-views", type=int, default=8, help="0/1 = без TTA")
    pred_p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)

    batch_p = sub.add_parser("batch", help="пакетный прогон по каталогу STP → CSV")
    batch_p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    batch_p.add_argument("--output", type=Path, default=ROOT / "results-predicted-12.csv")
    batch_p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    batch_p.add_argument("--n-points", type=int, default=2048)
    batch_p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    batch_p.add_argument("--tta-views", type=int, default=8, help="0/1 = без TTA")

    p.set_defaults(command="train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "batch":
        batch_predict(args)
    else:
        raise SystemExit("Укажите подкоманду: train, predict или batch")


if __name__ == "__main__":
    main()
