"""
Генерация STEP-моделей винтов (цилиндрическая головка, шестигранное углубление)
с заданными размерами.

Геометрия приближена к TraceParts (models/):
  - скругление верхнего обода головки (аналог Face2/17);
  - сопряжение головка–стержень (Face3/9);
  - фаска 45° на торце резьбы (Face5/6).

Запуск:
  QT_QPA_PLATFORM=offscreen freecadcmd /path/to/generate_screws.py

Вход: screw_variants.csv (или свой файл через --csv), либо --random N.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import FreeCAD as App
import Part
import Import

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "screw_variants.csv")
DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "models", "generated")

DIN912_STANDARD = {
    3: (5.5, 3.0),
    4: (7.0, 4.0),
    5: (8.5, 5.0),
    6: (10.0, 6.0),
    8: (13.0, 8.0),
    10: (16.0, 10.0),
    12: (18.0, 12.0),
    14: (21.0, 14.0),
    16: (24.0, 16.0),
    20: (30.0, 20.0),
}

HEX_SOCKET_BY_M = {
    2: (1.25, 1.0),
    2.5: (2.0, 1.1),
    3: (2.5, 1.5),
    4: (3.0, 2.0),
    5: (4.0, 2.5),
    6: (5.0, 3.0),
    7: (5.5, 3.5),
    8: (6.0, 4.0),
    9: (7.0, 4.5),
    10: (8.0, 5.0),
    12: (10.0, 6.0),
    14: (12.0, 7.0),
    16: (14.0, 8.0),
    18: (14.0, 9.0),
    20: (17.0, 10.0),
    22: (17.0, 11.0),
    24: (19.0, 12.0),
}


def _float_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _resolve_head_dims(thread_d, head_d, head_h):
    m = round(thread_d)
    if head_d is None or head_h is None:
        if m not in DIN912_STANDARD:
            raise ValueError(
                f"Для M{thread_d} нет записи в DIN912_STANDARD; укажите head_d_mm и head_h_mm."
            )
        std_dk, std_k = DIN912_STANDARD[m]
        head_d = std_dk if head_d is None else head_d
        head_h = std_k if head_h is None else head_h
    return head_d, head_h


def _nearest_hex_socket_key(thread_d):
    keys = sorted(HEX_SOCKET_BY_M.keys())
    return min(keys, key=lambda k: abs(k - thread_d))


def _hex_socket_dims(thread_d, head_d, head_h):

    m_key = _nearest_hex_socket_key(thread_d)
    s, t_nom = HEX_SOCKET_BY_M[m_key]

    scale = thread_d / m_key if m_key > 0 else 1.0
    if abs(scale - 1.0) > 0.15:
        s = max(2.0, min(s * scale, head_d * 0.85))

    s_max = head_d * 0.88
    s = min(s, s_max)

    depth = min(t_nom + 0.3, head_h - 0.6, head_d * 0.45)
    depth = max(0.8, depth)

    if s < 1.5:
        return None, None
    return s, depth


def _blend_radius_head_shank(thread_d: float) -> float:

    return max(0.12, min(0.5, 0.04 * thread_d + 0.01))


def _blend_radius_head_top(thread_d: float) -> float:

    return max(0.3, min(1.2, 0.1 * thread_d))


HEAD_H_SCALE = (0.96, 1.02)
HEAD_D_SCALE = (0.98, 1.04)
CHAMFER_AXIAL_MIN_MM = 0.55
CHAMFER_AXIAL_PER_D = 0.095
CHAMFER_TIP_RADIUS_FRAC = 0.82


def _thread_end_chamfer_dims(thread_d: float) -> tuple[float, float, float]:

    r_shank = thread_d / 2.0
    axial = max(CHAMFER_AXIAL_MIN_MM, CHAMFER_AXIAL_PER_D * thread_d)
    axial = min(axial, 0.14 * thread_d + 0.35)
    r_tip = max(r_shank * CHAMFER_TIP_RADIUS_FRAC, r_shank - 0.55)
    r_tip = min(r_tip, r_shank - 0.12)
    return axial, r_shank, r_tip


def _edges_near_y(solid, y_target: float, tol: float = 0.08):
    for edge in solid.Edges:
        ys = [v.Point.y for v in edge.Vertexes]
        if not ys:
            continue
        if max(ys) - min(ys) > tol:
            continue
        if abs(sum(ys) / len(ys) - y_target) <= tol:
            yield edge


def _edges_horizontal_circle(solid, y_target: float, r_expect: float, r_tol: float = 0.15, y_tol: float = 0.08):
    for edge in _edges_near_y(solid, y_target, y_tol):
        rs = [math.hypot(v.Point.x, v.Point.z) for v in edge.Vertexes]
        if not rs:
            continue
        r_mean = sum(rs) / len(rs)
        if abs(r_mean - r_expect) <= r_tol:
            yield edge


def _apply_fillet(solid, radius: float, edges: list, label: str):
    if not edges or radius <= 1e-4:
        return solid
    try:
        return solid.makeFillet(radius, edges)
    except Exception as exc:
        print(f"[WARN] Скругление {label} не применено (R={radius:g}): {exc}")
        return solid


def _apply_head_shank_fillet(solid, r_shank: float, y_junction: float, radius: float):

    edges = list(_edges_horizontal_circle(solid, y_junction, r_shank, r_tol=0.2))
    return _apply_fillet(solid, radius, edges, "головка–стержень")


def _apply_head_top_fillet(solid, r_head: float, y_top: float, radius: float):
    edges = list(_edges_horizontal_circle(solid, y_top, r_head, r_tol=0.25))
    return _apply_fillet(solid, radius, edges, "верх головки")


def _make_shank_with_thread_taper(thread_d: float, length: float, y_base: float):

    axial, r_shank, r_tip = _thread_end_chamfer_dims(thread_d)
    r_tip = max(r_tip, 0.15)
    axial = min(axial, length)
    cyl_len = length - axial
    if cyl_len > 1e-4:
        shank = Part.makeCylinder(
            r_shank,
            cyl_len,
            App.Vector(0, y_base, 0),
            App.Vector(0, 1, 0),
        )
        taper = Part.makeCone(
            r_shank,
            r_tip,
            axial,
            App.Vector(0, y_base + cyl_len, 0),
            App.Vector(0, 1, 0),
        )
        return shank.fuse(taper)
    return Part.makeCone(
        r_shank,
        r_tip,
        length,
        App.Vector(0, y_base, 0),
        App.Vector(0, 1, 0),
    )


def _make_hex_prism_cut(across_flats_mm, depth_mm, y_top):

    s = across_flats_mm
    y0 = y_top - depth_mm
    r_vertex = s / math.sqrt(3.0)
    pts = []
    for i in range(6):
        ang = math.pi / 2.0 + i * math.pi / 3.0
        x = r_vertex * math.cos(ang)
        z = r_vertex * math.sin(ang)
        pts.append(App.Vector(x, y0, z))
    pts.append(pts[0])
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    return face.extrude(App.Vector(0, depth_mm, 0))


def build_screw_shape(thread_d, head_d, length, head_h):

    if thread_d <= 0 or head_d <= 0 or length <= 0 or head_h <= 0:
        raise ValueError("Все размеры должны быть > 0.")
    if head_d < thread_d:
        raise ValueError("Диаметр головки должен быть не меньше диаметра резьбы.")

    r_shank = thread_d / 2.0
    r_head = head_d / 2.0
    y_head_bottom = 0.0
    y_junction = head_h

    r_hs = _blend_radius_head_shank(thread_d)
    r_crown = _blend_radius_head_top(thread_d)

    head = Part.makeCylinder(
        r_head,
        head_h,
        App.Vector(0, y_head_bottom, 0),
        App.Vector(0, 1, 0),
    )
    shank = _make_shank_with_thread_taper(thread_d, length, y_junction)
    solid = head.fuse(shank)

    solid = _apply_head_shank_fillet(solid, r_shank, y_junction, r_hs)
    solid = _apply_head_top_fillet(solid, r_head, y_head_bottom, r_crown)

    s_hex, depth = _hex_socket_dims(thread_d, head_d, head_h)
    if s_hex is not None and depth > 0.5:
        try:
            socket_tool = _make_hex_prism_cut(s_hex, depth, depth)
            solid = solid.cut(socket_tool)
        except Exception:
            pass

    return solid


def find_thread_end_face(geo_obj):

    best_name = None
    best_y = float("-inf")
    for index, face in enumerate(geo_obj.Shape.Faces, start=1):
        center = face.CenterOfMass
        if center.y > best_y:
            best_y = center.y
            best_name = f"Face{index}"
    if best_name is None:
        raise RuntimeError("Не найдена грань торца резьбы для закрепления.")
    return best_name


def export_screw(step_path, thread_d, head_d, length, head_h):
    doc = App.newDocument("ScrewGen")
    try:
        shape = build_screw_shape(thread_d, head_d, length, head_h)
        part = doc.addObject("Part::Feature", "Screw")
        part.Label = f"DIN912_M{thread_d:g}x{length:g}"
        part.Shape = shape
        doc.recompute()

        os.makedirs(os.path.dirname(step_path), exist_ok=True)
        Import.export([part], step_path)

        fixed_face = find_thread_end_face(part)
        bb = part.Shape.BoundBox
        return {
            "fixed_face": fixed_face,
            "bbox_x": bb.XLength,
            "bbox_y": bb.YLength,
            "bbox_z": bb.ZLength,
        }
    finally:
        App.closeDocument(doc.Name)


def load_variants(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "thread_d_mm", "head_d_mm", "length_mm"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV должен содержать столбцы: {', '.join(sorted(required))}")
        for row in reader:
            name = row["name"].strip()
            if not name:
                continue
            rows.append(
                {
                    "name": name,
                    "thread_d": float(row["thread_d_mm"]),
                    "head_d": _float_or_none(row.get("head_d_mm")),
                    "length": float(row["length_mm"]),
                    "head_h": _float_or_none(row.get("head_h_mm")),
                }
            )
    return rows


def build_random_variants(count, seed=None):

    if count < 1:
        return []
    rng = random.Random(seed)
    ms = sorted(DIN912_STANDARD.keys())
    rows = []
    for idx in range(count):
        name = f"rand_{idx + 1:04d}"
        m_base = rng.choice(ms)
        dk0, k0 = DIN912_STANDARD[m_base]
        thread_d = round(m_base + rng.uniform(-0.35, 0.35), 2)
        thread_d = max(2.5, min(22.0, thread_d))
        head_d = round(dk0 * rng.uniform(*HEAD_D_SCALE), 2)
        head_h = round(k0 * rng.uniform(*HEAD_H_SCALE), 2)
        if head_d < thread_d + 0.6:
            head_d = round(thread_d + 0.6, 2)
        if head_h < 2.0:
            head_h = 2.0
        min_len = max(4.0, head_h * 0.45 + 1.0)
        max_len = min(130.0, 4.0 + m_base * rng.uniform(2.0, 14.0))
        if max_len <= min_len:
            max_len = min_len + 5.0
        decimals = rng.choice([0, 1, 2])
        length = round(rng.uniform(min_len, max_len), decimals)
        rows.append(
            {
                "name": name,
                "thread_d": thread_d,
                "head_d": head_d,
                "head_h": head_h,
                "length": length,
            }
        )
    return rows


def _write_random_specs_csv(path, variants, seed):

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "thread_d_mm", "head_d_mm", "length_mm", "head_h_mm", "seed"],
        )
        w.writeheader()
        seed_str = "" if seed is None else str(seed)
        for v in variants:
            w.writerow(
                {
                    "name": v["name"],
                    "thread_d_mm": v["thread_d"],
                    "head_d_mm": v["head_d"],
                    "length_mm": v["length"],
                    "head_h_mm": v["head_h"],
                    "seed": seed_str,
                }
            )


def _script_argv():

    args = sys.argv[1:]
    if args and args[0].endswith(".py"):
        return args[1:]
    return args


def main():
    parser = argparse.ArgumentParser(description="Генерация STEP винтов DIN 912")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Таблица размеров (если не --random)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Папка для .stp и manifest.csv")
    parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        default=0,
        help="Сгенерировать N случайных винтов (если freecadcmd съедает флаги — см. SCREW_RANDOM_COUNT)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed RNG")
    args = parser.parse_args(_script_argv())

    env_n = os.environ.get("SCREW_RANDOM_COUNT", "").strip()
    if env_n.isdigit():
        args.random = int(env_n)
    env_seed = os.environ.get("SCREW_RANDOM_SEED", "").strip()
    if env_seed:
        try:
            args.seed = int(env_seed)
        except ValueError:
            pass
    env_out = os.environ.get("SCREW_OUT", "").strip()
    if env_out:
        args.out = env_out

    if args.random and args.random > 0:
        variants = build_random_variants(args.random, args.seed)
        out_dir = args.out
        if out_dir == DEFAULT_OUT:
            out_dir = os.path.join(_SCRIPT_DIR, "models", f"generated_random_{args.random}")
        args.out = out_dir
    else:
        variants = load_variants(args.csv)

    if not variants:
        print("[ERROR] Нет вариантов для генерации.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    if args.random and args.random > 0:
        specs_path = os.path.join(args.out, "random_specs.csv")
        _write_random_specs_csv(specs_path, variants, args.seed)
        info_path = os.path.join(args.out, "run_info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"count={args.random}\nseed={args.seed!r}\n")

    manifest_path = os.path.join(args.out, "manifest.csv")
    manifest_rows = []

    n_total = len(variants)
    print(f"[INFO] Генерация {n_total} моделей → {args.out}")

    for i, spec in enumerate(variants, start=1):
        name = spec["name"]
        thread_d = spec["thread_d"]
        length = spec["length"]
        head_d, head_h = _resolve_head_dims(thread_d, spec["head_d"], spec["head_h"])

        step_path = os.path.join(args.out, f"{name}.stp")
        hs, hd = _hex_socket_dims(thread_d, head_d, head_h)
        hex_info = f", hex {hs:g}×{hd:g} mm" if hs else ""
        if n_total <= 30 or i == 1 or i == n_total or (n_total > 50 and i % 100 == 0):
            print(f"[INFO] ({i}/{n_total}) {name}: M{thread_d:g}x{length:g}, dk={head_d:g}, k={head_h:g} mm{hex_info}")

        try:
            meta = export_screw(step_path, thread_d, head_d, length, head_h)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            continue

        manifest_rows.append(
            {
                "name": name,
                "file": f"{name}.stp",
                "thread_d_mm": thread_d,
                "head_d_mm": head_d,
                "length_mm": length,
                "head_h_mm": head_h,
                "hex_af_mm": f"{hs:.2f}" if hs else "",
                "hex_depth_mm": f"{hd:.2f}" if hs else "",
                "fixed_face": meta["fixed_face"],
                "bbox_y_mm": f"{meta['bbox_y']:.4f}",
            }
        )
        if n_total <= 30:
            print(f"       → {step_path} ({meta['fixed_face']})")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "file",
                "thread_d_mm",
                "head_d_mm",
                "length_mm",
                "head_h_mm",
                "hex_af_mm",
                "hex_depth_mm",
                "fixed_face",
                "bbox_y_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[INFO] Готово: {len(manifest_rows)}/{n_total} файлов, manifest: {manifest_path}")


# freecadcmd задаёт __name__ = имя файла без .py (не "__main__")
if __name__ in ("__main__", "generate_screws"):
    main()
