"""Parcellated connectome-harmonic helpers (CHAP-compatible math, Schaefer scale)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent


def load_harmonics_json(path: Path, n_modes: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    modes = np.asarray(data["modes"], dtype=np.float64)
    eigvals = np.asarray(data["eigvals"], dtype=np.float64)
    if n_modes is not None:
        modes = modes[:n_modes]
        eigvals = eigvals[:n_modes]
    # rows = parcels, cols = modes (CHAP uses vecs[:, k])
    vecs = modes.T
    return vecs, eigvals, np.asarray(data["nodes"], dtype=np.float64)


def lap_decomp_from_edges(
    edges: list[list[float]], n_nodes: int, n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols, data = [], [], []
    for i, j, w in edges:
        rows.extend([int(i), int(j)])
        cols.extend([int(j), int(i)])
        data.extend([float(w), float(w)])
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    lap = laplacian(adj, normed=True).astype(np.float64)
    k = min(n_modes + 1, n_nodes - 2)
    vals, vecs = eigsh(lap, k=k, which="SM")
    order = np.argsort(vals)
    vals = vals[order][1 : n_modes + 1]
    vecs = vecs[:, order][:, 1 : n_modes + 1]
    return vals, vecs


def zero_mean_timeseries(ts: np.ndarray) -> np.ndarray:
    """ts shape: (n_parcels, n_timepoints)."""
    return ts - ts.mean(axis=1, keepdims=True)


def mean_power_spectrum(timeseries: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """CHAP mean_power_spectrum — activity projected onto each harmonic."""
    zm = zero_mean_timeseries(timeseries)
    n_modes = vecs.shape[1]
    spectrum = np.zeros(n_modes, dtype=np.float64)
    for k in range(n_modes):
        v = vecs[:, k]
        for tp in range(zm.shape[1]):
            spectrum[k] += abs(float(np.dot(v, zm[:, tp]))) / zm.shape[1]
    return spectrum


def dynamic_power_spectrum(timeseries: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    zm = zero_mean_timeseries(timeseries)
    n_modes = vecs.shape[1]
    spectrum = np.zeros((n_modes, zm.shape[1]), dtype=np.float64)
    for k in range(n_modes):
        v = vecs[:, k]
        for tp in range(zm.shape[1]):
            spectrum[k, tp] = abs(float(np.dot(v, zm[:, tp])))
    return spectrum


def repertoire_entropy(spectrum: np.ndarray, eps: float = 1e-12) -> float:
    p = spectrum / (spectrum.sum() + eps)
    p = p[p > eps]
    return float(-(p * np.log(p)).sum())


def active_mode_count(spectrum: np.ndarray, percentile: float = 75.0) -> int:
    thr = np.percentile(spectrum, percentile)
    return int(np.sum(spectrum >= thr))
