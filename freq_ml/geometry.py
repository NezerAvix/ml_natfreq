"""
Извлечение признаков геометрии из STEP (.stp) через Gmsh
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gmsh
import numpy as np

if TYPE_CHECKING:
    from freq_ml.screw_params import ScrewParams

N_POINTS_DEFAULT = 2048
MESH_H_MAX = 1.2
MESH_H_MIN = 0.25
N_BBOX_SCALARS = 8
N_PARAM_SCALARS = 7
N_SCALARS = N_BBOX_SCALARS + N_PARAM_SCALARS
GEOMETRY_FEATURE_VERSION = 5
POINT_DIM = 6


@dataclass(frozen=True)
class GeometrySample:
    points: np.ndarray
    scalars: np.ndarray


def _gmsh_load_surface(
    stp_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float]:
    gmsh.initialize(["-nopopup"])
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("part")
        gmsh.merge(os.path.abspath(stp_path))
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        volume = 0.0
        area = 0.0
        for dim, tag in gmsh.model.getEntities(3):
            volume += gmsh.model.occ.getMass(dim, tag)
        for dim, tag in gmsh.model.getEntities(2):
            area += gmsh.model.occ.getMass(dim, tag)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_H_MAX)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_H_MIN)
        gmsh.model.mesh.generate(2)

        node_tags, coord, _ = gmsh.model.mesh.getNodes()
        pts = np.asarray(coord, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        triangles: list[tuple[int, int, int]] = []
        elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        for flat in elem_node_tags:
            arr = np.asarray(flat, dtype=np.int64)
            nverts = 3 if len(arr) % 3 == 0 else 0
            for i in range(0, nverts, 3):
                a, b, c = int(arr[i]), int(arr[i + 1]), int(arr[i + 2])
                if a in tag_to_idx and b in tag_to_idx and c in tag_to_idx:
                    triangles.append((tag_to_idx[a], tag_to_idx[b], tag_to_idx[c]))
    finally:
        gmsh.finalize()

    tris = np.asarray(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)
    length = ymax - ymin
    dx = xmax - xmin
    dz = zmax - zmin
    d_max = max(dx, dz)
    return pts, tris, length, d_max, dx, dz, volume, area


def _vertex_normals(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = len(pts)
    normals = np.zeros((n, 3), dtype=np.float64)
    if len(tris) == 0:
        return normals
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, tris[:, 0], fn)
    np.add.at(normals, tris[:, 1], fn)
    np.add.at(normals, tris[:, 2], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    mask = lengths.squeeze(axis=1) > 1e-12
    normals[mask] /= lengths[mask]
    return normals


def _farthest_point_sample(
    pts: np.ndarray,
    normals: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(pts)
    if n == 0:
        return np.zeros(n_points, dtype=np.int64)
    if n <= n_points:
        return rng.choice(n, n_points, replace=True)

    indices = np.empty(n_points, dtype=np.int64)
    indices[0] = int(rng.integers(0, n))
    dists = np.full(n, np.inf, dtype=np.float64)
    last = indices[0]
    for i in range(1, n_points):
        diff = pts - pts[last]
        dists = np.minimum(dists, np.einsum("ij,ij->i", diff, diff))
        indices[i] = int(np.argmax(dists))
        last = indices[i]
    return indices


def _build_scalars(length: float, d_max: float, dx: float, dz: float, volume: float, area: float) -> np.ndarray:
    slenderness = length / (d_max + 1e-6)
    compactness = volume / (length * d_max * d_max + 1e-6)
    return np.array(
        [
            np.log1p(length),
            np.log1p(d_max),
            np.log1p(dx),
            np.log1p(dz),
            np.log1p(volume),
            np.log1p(area),
            slenderness,
            compactness * 1e3,
        ],
        dtype=np.float32,
    )


def normalize_points_with_normals(pts: np.ndarray, normals: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float64, copy=True)
    normals = normals.astype(np.float64, copy=True)
    center = pts.mean(axis=0)
    pts -= center
    scale = float(np.linalg.norm(pts, axis=1).max()) + 1e-9
    pts /= scale
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen = np.maximum(nlen, 1e-9)
    normals /= nlen
    feat = np.concatenate([pts.astype(np.float32), normals.astype(np.float32)], axis=1)
    return feat


def augment_points(pts_feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = pts_feat[:, :3].astype(np.float32, copy=True)
    nrm = pts_feat[:, 3:6].astype(np.float32, copy=True)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    pos = pos @ rot.T
    nrm = nrm @ rot.T
    pos += rng.normal(0.0, 0.006, size=pos.shape).astype(np.float32)
    return np.concatenate([pos, nrm], axis=1).astype(np.float32)


def _build_full_scalars(
    length: float,
    d_max: float,
    dx: float,
    dz: float,
    volume: float,
    area: float,
    screw_params: ScrewParams | None,
    mesh_pts: np.ndarray | None = None,
) -> np.ndarray:
    from freq_ml.screw_params import build_param_scalars, estimate_params_from_points

    bbox = _build_scalars(length, d_max, dx, dz, volume, area)
    if screw_params is None and mesh_pts is not None and len(mesh_pts):
        screw_params = estimate_params_from_points(mesh_pts)
    if screw_params is None:
        screw_params = estimate_params_from_points(
            mesh_pts if mesh_pts is not None and len(mesh_pts) else np.zeros((1, 3))
        )
    return np.concatenate([bbox, build_param_scalars(screw_params)])


def sample_point_cloud(
    stp_path: str,
    n_points: int = N_POINTS_DEFAULT,
    seed: int = 0,
    screw_params: ScrewParams | None = None,
) -> GeometrySample:
    pts, tris, length, d_max, dx, dz, volume, area = _gmsh_load_surface(stp_path)
    normals = _vertex_normals(pts, tris)

    rng = np.random.default_rng(seed)
    if len(pts) == 0:
        feat = np.zeros((n_points, POINT_DIM), dtype=np.float32)
    else:
        idx = _farthest_point_sample(pts, normals, n_points, rng)
        feat = normalize_points_with_normals(pts[idx], normals[idx])

    return GeometrySample(
        points=feat,
        scalars=_build_full_scalars(
            length, d_max, dx, dz, volume, area, screw_params, mesh_pts=pts
        ),
    )
