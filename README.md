# Connectome Harmonics

Educational simulator for **graph Laplacian harmonics** on brain-like connectivity networks — inspired by [Atasoy et al., 2016](https://www.nature.com/articles/ncomms10340).

**Live demo (recommended):** https://fractastical.github.io/connectome_harmonics/

Also mirrored at [cimcai/connectome_harmonics](https://github.com/cimcai/connectome_harmonics) (Pages may conflict with [cimc.ai](https://cimc.ai/) org domain — use fractastical for a reliable public URL).

The graph Laplacian \(L = I - D^{-1/2} W D^{-1/2}\) turns a connectome into a geometry. Its eigenvectors are spatial **harmonics**: low modes are broad, bilateral patterns; high modes are finer, more localized waves.

## What's here

| File | Description |
|------|-------------|
| `connectome_harmonics_simulation.html` | Interactive browser simulator (3 physics models, 40 modes) |
| `connectome_harmonics_sim.py` | Matplotlib animation for a single mode |
| `build_connectome_data.py` | Build harmonics JSON from real HCP Schaefer-400 SC |
| `connectome_harmonics_data.json` | Synthetic toy connectome |
| `connectome_harmonics_data_hcp.json` | Real HCP-YA structural connectome (built) |

## Quick start

**Browser:** open [index.html](index.html) locally, or visit https://fractastical.github.io/connectome_harmonics/. Features a **mode tour**, literature references, and HCP vs toy data toggle.

### Deploy on fractastical GitHub Pages

**Live URL:** https://fractastical.github.io/connectome_harmonics/

1. Create a public empty repo at https://github.com/new → owner **fractastical**, name **connectome_harmonics**
2. Push: `git push fractastical main`
3. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**
4. First deploy: **Actions** tab → approve **github-pages** environment if prompted

The workflow in `.github/workflows/pages.yml` publishes `index.html`, connectome JSON data, and LSD result plots automatically on every push to `main`.

**Python animation:**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python connectome_harmonics_sim.py
.venv/bin/python connectome_harmonics_sim.py --data connectome_harmonics_data_hcp.json --mode 7
```

**Rebuild real connectome data (option 1 — HCP Schaefer-400 SC):**
```bash
.venv/bin/python build_connectome_data.py
```

**LSD harmonic analysis (OpenNeuro ds003059 + CHAP-style projection):**
```bash
python3 -m venv .venv-lsd && .venv-lsd/bin/pip install -r requirements-lsd.txt
.venv-lsd/bin/python analyze_lsd_harmonics.py --max-subjects 2   # quick demo
.venv-lsd/bin/python analyze_lsd_harmonics.py                    # full cohort (~10 GB download)
```

## Data sources

- **Toy graph:** synthetic two-hemisphere network (not medical data)
- **Real graph:** group-average HCP Young Adult structural connectivity, Schaefer-400 parcels, from [GNN_SC_FC](https://github.com/PeiyuChen2023/GNN_SC_FC)

## Harmonic field model

Standing waves on the graph:

\[
x_i(t) = a_k \, v_{k,i} \cos(\sqrt{\lambda_k} \cdot t)
\]

where \(v_k\) is eigenvector \(k\) and \(\lambda_k\) is the corresponding Laplacian eigenvalue (related to spatial frequency on the network).
