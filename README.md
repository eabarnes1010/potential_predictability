# ZIG Precipitation Model & Potential Predictability

A PyTorch implementation of a **Zero-Inflated Gamma (ZIG) neural network** as a
drop-in replacement for the two-stage Seasonally Stationary Weather Model (SSWM)
from Anderson et al. (2016), together with a full replication of their
**Potential Predictable Variance (PPV)** framework (Eqs. 1 and 7–8).

> **Reference:** Anderson, B. T., Gianotti, D. J. S., Salvucci, G., & Furtado, J.
> (2016). Dominant time scales of potentially predictable precipitation variations
> across the continental United States. *Journal of Climate*, 29(24), 8881–8897.
> https://doi.org/10.1175/JCLI-D-15-0635.1

---

## Overview

The classic SSWM uses two separately trained models — a Markov chain for daily
precipitation occurrence and a gamma–gamma mixture for intensity. This repo
replaces both with a **single jointly trained MLP** that simultaneously outputs:

| Head | Output | Constraint |
|------|--------|------------|
| `logit_p` | P(wet \| x) | sigmoid → (0, 1) |
| `alpha` | Gamma shape per component | softplus → > 0 |
| `beta` | Gamma rate per component | softplus → > 0 |
| `pi` | Mixture weights | softmax → sum = 1 |

Set `n_components=1` for a simple ZIG-Gamma; `n_components=2` reproduces the
paper's gamma–gamma intensity model. See `zig_architecture.html` for a
rendered diagram of the network.

The notebook also fully replicates the PPV significance framework:

- **Eq. 1** — raw PPV: fraction of observed inter-annual variance not explained
  by the stochastic baseline.
- **Eq. 7** — null distribution: within-ensemble PPV used to set the p < 0.10
  significance threshold.
- **Normalized PPV** — raw PPV divided by the 90th-percentile threshold;
  values > 1 are statistically significant and correspond to the color scale
  (1–4+) in the paper's maps.

---

## Repository structure

```
.
├── zig_precip.ipynb              # Main notebook (see sections below)
├── zig_architecture.html         # Interactive network diagram
├── data/
│   └── synthetic/
│       ├── make_enso_ar1_data.py # Generator for the ENSO + AR(1) dataset
│       └── enso_ar1_80yr.npz     # 80-year synthetic dataset (see below)
├── docs/
│   └── Anderson_2016_PotPred.pdf # Reference paper
├── figures/
│   ├── ppv_anderson2016_eq1.png          # Raw PPV figure
│   └── ppv_normalized_anderson2016.png   # Normalized PPV figure
└── _archive/
    └── two_stage_precip.py       # Original two-stage model (reference only)
```

---

## Notebook sections

| § | Title | Description |
|---|-------|-------------|
| 1 | Imports | PyTorch, NumPy, matplotlib |
| 2 | Feature engineering | DOY (sin/cos) + lagged occurrence → input tensor |
| 3 | Model | `ZIGammaMLP` — shared trunk + 4 output heads |
| 4 | Joint NLL loss | Single zero-inflated gamma loss function |
| 5 | Training | `train_zig()` with AdamW + cosine annealing |
| 6 | Monte Carlo simulation | 1000-member ensemble forward integration |
| 7 | Calibration | Reliability diagram, intensity Q-Q, annual OCC/SII/TOT |
| 8 | Data loading | File-based loader + inline synthetic fallback |
| 9 | PPV — Eq. 1 | `compute_ppv()`: raw PPV for OCC, SII, TOT |
| 9b | PPV — Eqs. 7–8 | `compute_normalized_ppv()`: null distribution + significance |

### Running end-to-end

1. Set `DATA_FILE` in §8 to your `.npz` data path (or `None` for the inline
   synthetic fallback).
2. Run all cells in order.  §9 / §9b depend on `occ_sim` / `int_sim` produced
   by the Monte Carlo cell in §6.

> **Tip:** Switch `DATA_FILE` to `"data/synthetic/enso_ar1_80yr.npz"` for a
> dataset where the ENSO-driven inter-annual signal is strong enough to produce
> clearly significant normalized PPV (> 1) for all three metrics.

---

## Synthetic data

### `enso_ar1_80yr.npz`

An 80-year daily record (29 200 days) with two explicit variance components:

| Component | What drives it | Captured by ZIG? |
|-----------|---------------|------------------|
| **AR(1) persistence** | β_AR1 = 0.30 on occ[t−1] | ✅ yes (via `occ_lags`) |
| **Seasonal cycle** | Cosine amplitude ± 0.6 logit | ✅ yes (via DOY encoding) |
| **ENSO forcing** | Annual AR(1), φ = 0.70; β_ENSO = 0.80 logit/SD on P(wet); γ_ENSO = 0.25 log-mm/SD on intensity | ❌ no (hidden) |

Because the ZIG model sees only 5-day occurrence lags and DOY, it cannot learn
the multi-year ENSO signal. The residual inter-annual variance drives PPV > 0.

**Arrays saved:**

| Array | Shape | Description |
|-------|-------|-------------|
| `doy` | (N,) int32 | Day-of-year, 1–365 |
| `occ_lags` | (N, 5) float32 | Lagged occurrence (col 0 = oldest, col −1 = t−1) |
| `occ_obs` | (N,) float32 | Binary occurrence (0/1) |
| `int_obs` | (N,) float32 | Intensity in mm (0 on dry days) |
| `enso_daily` | (N,) float32 | Annual ENSO index at daily resolution |
| `enso_annual` | (80,) float32 | Annual ENSO index (for reference / plotting) |

To regenerate the file or adjust parameters:

```bash
python data/synthetic/make_enso_ar1_data.py
```

---

## Input features

```
x  (N, 7)
│
├── sin(2π · doy / 365)   ← smooth seasonal cycle
├── cos(2π · doy / 365)
├── occ[t−5]              ← AR(1) persistence
├── occ[t−4]
├── occ[t−3]
├── occ[t−2]
└── occ[t−1]              ← most recent lag
```

Pass additional predictors (e.g., ENSO index, SST) via the `extra` argument to
`build_features()` — they will be concatenated and `in_features` updated
automatically.

---

## Dependencies

```
python  >= 3.10
torch   >= 2.0
numpy
matplotlib
```

No special installation beyond a standard scientific Python environment.

---

## License

CC BY 4.0 — © 2026 Elizabeth A. Barnes.
Free to use and adapt with attribution.
See [`LICENSE`](LICENSE) for full terms.
