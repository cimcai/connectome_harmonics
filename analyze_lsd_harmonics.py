"""
Connectome-harmonic decomposition of OpenNeuro ds003059 (LSD vs placebo).

Parcellated approximation of Atasoy et al. 2017 / CHAP:
  1. Structural harmonics from HCP Schaefer-400 SC (prebuilt JSON)
  2. Resting-state fMRI from Carhart-Harris LSD study (OpenNeuro)
  3. Project BOLD onto harmonic basis → LSD vs placebo power spectra

Full vertex-level CHAP (Docker): https://github.com/HopkinsPsychedelic/connectome_harmonic_core

Quick demo (2 subjects, ~700 MB download):
    python analyze_lsd_harmonics.py --max-subjects 2

Full cohort (15 subjects, ~10 GB):
    python analyze_lsd_harmonics.py
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chap_compat import (
    active_mode_count,
    load_harmonics_json,
    mean_power_spectrum,
    repertoire_entropy,
)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "ds003059"
OUT = ROOT / "lsd_results"

OPENNEURO_BASE = "https://s3.amazonaws.com/openneuro.org/ds003059"
DEFAULT_HARMONICS = ROOT / "connectome_harmonics_data_hcp.json"

# README: exclude music-problem sessions; motion-heavy subjects often dropped in papers
EXCLUDE_SUBJECTS = {"sub-003", "sub-012", "sub-015"}
ALL_SUBJECTS = [
    "sub-001", "sub-002", "sub-004", "sub-006", "sub-009", "sub-010",
    "sub-011", "sub-013", "sub-017", "sub-018", "sub-019", "sub-020",
]
REST_RUNS = ("run-01", "run-03")  # skip run-02 (music)


def download_openneuro(key: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / key.replace("ds003059/", "")
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        rel = key.split("ds003059/", 1)[-1]
        url = f"{OPENNEURO_BASE}/{rel}"
        print(f"Downloading {key} …")
        urllib.request.urlretrieve(url, local)
    return local


def fetch_schaefer_masker():
    import os
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.maskers import NiftiLabelsMasker

    data_dir = str(ROOT / ".cache" / "nilearn_data")
    os.environ.setdefault("NILEARN_DATA", data_dir)
    atlas = fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=7, resolution_mm=2, data_dir=data_dir
    )
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=False,
        detrend=False,
        high_pass=None,
        low_pass=None,
        t_r=2.0,
        memory="nilearn_cache",
        verbose=0,
    )
    return masker


def extract_parcel_timeseries(bold_path: Path, masker) -> np.ndarray:
    """Return (n_parcels, n_timepoints)."""
    ts = masker.fit_transform(str(bold_path))  # (T, parcels)
    return np.asarray(ts, dtype=np.float64).T


def collect_runs(subjects: list[str]) -> list[dict]:
    runs = []
    for sub in subjects:
        for session, label in (("ses-LSD", "lsd"), ("ses-PLCB", "placebo")):
            for run in REST_RUNS:
                key = f"ds003059/{sub}/{session}/func/{sub}_{session}_task-rest_{run}_bold.nii.gz"
                runs.append({"subject": sub, "condition": label, "run": run, "key": key})
    return runs


def analyze_cohort(
    subjects: list[str],
    harmonics_path: Path,
    n_modes: int,
) -> dict:
    vecs, eigvals, _ = load_harmonics_json(harmonics_path, n_modes=n_modes)
    masker = fetch_schaefer_masker()

    per_run = []
    for spec in collect_runs(subjects):
        path = download_openneuro(spec["key"])
        ts = extract_parcel_timeseries(path, masker)
        if ts.shape[0] != vecs.shape[0]:
            raise SystemExit(
                f"Parcel count {ts.shape[0]} != harmonic nodes {vecs.shape[0]}. "
                "Use Schaefer-400 harmonics."
            )
        spectrum = mean_power_spectrum(ts, vecs)
        per_run.append({**spec, "spectrum": spectrum.tolist(), "entropy": repertoire_entropy(spectrum), "active_modes": active_mode_count(spectrum)})

    by_cond: dict[str, list[np.ndarray]] = {"lsd": [], "placebo": []}
    for row in per_run:
        by_cond[row["condition"]].append(np.array(row["spectrum"]))

    group_lsd = np.mean(by_cond["lsd"], axis=0)
    group_pcb = np.mean(by_cond["placebo"], axis=0)
    ratio = group_lsd / (group_pcb + 1e-12)

    n = len(group_lsd)
    low_band = slice(0, max(1, n // 5))
    high_band = slice(max(0, n - n // 5), n)

    return {
        "meta": {
            "dataset": "OpenNeuro ds003059",
            "doi": "10.18112/openneuro.ds003059.v1.0.0",
            "harmonics_source": str(harmonics_path.name),
            "n_subjects": len(subjects),
            "n_modes": n_modes,
            "method": "Parcellated CHAP-style mean_power_spectrum (Schaefer-400, HCP SC harmonics)",
            "reference": "Atasoy et al. 2017 Sci Rep — connectome-harmonic decomposition under LSD",
            "chap_repo": "https://github.com/HopkinsPsychedelic/connectome_harmonic_core",
        },
        "eigvals": eigvals.tolist(),
        "subjects": subjects,
        "per_run": per_run,
        "group_mean_spectrum": {"lsd": group_lsd.tolist(), "placebo": group_pcb.tolist()},
        "group_ratio_lsd_over_placebo": ratio.tolist(),
        "summary": {
            "low_band_mean_ratio": float(ratio[low_band].mean()),
            "high_band_mean_ratio": float(ratio[high_band].mean()),
            "mean_entropy_lsd": float(np.mean([r["entropy"] for r in per_run if r["condition"] == "lsd"])),
            "mean_entropy_placebo": float(np.mean([r["entropy"] for r in per_run if r["condition"] == "placebo"])),
            "interpretation": (
                "Ratio > 1 means higher harmonic power under LSD. "
                "Atasoy 2017: expect low-mode ratio < 1 and high-mode ratio > 1."
            ),
        },
    }


def plot_results(payload: dict, out_png: Path) -> None:
    lsd = np.array(payload["group_mean_spectrum"]["lsd"])
    pcb = np.array(payload["group_mean_spectrum"]["placebo"])
    ratio = np.array(payload["group_ratio_lsd_over_placebo"])
    modes = np.arange(1, len(lsd) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.patch.set_facecolor("#0b0e14")

    ax = axes[0]
    ax.set_facecolor("#0b0e14")
    ax.plot(modes, pcb, label="placebo", color="#8ec5ff", lw=2)
    ax.plot(modes, lsd, label="LSD", color="#c9a0ff", lw=2)
    ax.set_ylabel("mean harmonic power")
    ax.set_title("Connectome harmonics — LSD vs placebo (ds003059)", color="white")
    ax.tick_params(colors="#9fb0c8")
    ax.legend(facecolor="#111722", edgecolor="#333", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    ax2 = axes[1]
    ax2.set_facecolor("#0b0e14")
    ax2.axhline(1.0, color="#666", ls="--", lw=1)
    ax2.plot(modes, ratio, color="#ffd18e", lw=2)
    ax2.fill_between(modes, 1.0, ratio, where=ratio >= 1.0, alpha=0.25, color="#c9a0ff")
    ax2.fill_between(modes, ratio, 1.0, where=ratio < 1.0, alpha=0.25, color="#8ec5ff")
    ax2.set_xlabel("harmonic mode (low → high spatial frequency)")
    ax2.set_ylabel("LSD / placebo")
    ax2.tick_params(colors="#9fb0c8")
    for spine in ax2.spines.values():
        spine.set_color("#333")

    s = payload["summary"]
    fig.text(
        0.5, 0.01,
        f"low-band ratio={s['low_band_mean_ratio']:.3f}  high-band ratio={s['high_band_mean_ratio']:.3f}  "
        f"entropy LSD={s['mean_entropy_lsd']:.2f} PCB={s['mean_entropy_placebo']:.2f}",
        ha="center", color="#9fb0c8", fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonics", type=Path, default=DEFAULT_HARMONICS)
    parser.add_argument("--modes", type=int, default=40)
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit cohort for quick demo")
    parser.add_argument("--subjects", nargs="*", help="Explicit subject IDs (e.g. sub-001)")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    if args.subjects:
        subjects = list(args.subjects)
    else:
        subjects = [s for s in ALL_SUBJECTS if s not in EXCLUDE_SUBJECTS]
        if args.max_subjects:
            subjects = subjects[: args.max_subjects]

    print(f"Analyzing {len(subjects)} subjects: {', '.join(subjects)}")
    payload = analyze_cohort(subjects, args.harmonics, args.modes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "lsd_harmonic_spectra.json"
    png_path = args.output_dir / "lsd_harmonic_spectra.png"
    json_path.write_text(json.dumps(payload, indent=2))
    plot_results(payload, png_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    print("Summary:", payload["summary"])


if __name__ == "__main__":
    main()
