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
replaces both with a **single jointly trained MLP** that simultaneously outputs
all four parameters needed to describe the full daily precipitation distribution.
See `zig_architecture.html` for a rendered diagram of the network.

---

## Model outputs and their roles

All four heads read from the same shared 64-dimensional trunk, so occurrence and
intensity are learned jointly rather than in separate stages.

### Occurrence — `logit_p`

`logit_p` is a scalar raw logit per day. After a sigmoid activation it becomes
**P(wet | x)** — the probability that day *t* is a precipitation event given the
input features. During training, dry days contribute `log(1 − P(wet))` to the
loss and wet days contribute `log P(wet)` plus the intensity log-likelihood.

During the Monte Carlo simulation a Bernoulli draw from P(wet) produces a binary
occurrence sequence `occ_sim`. Summing over a year gives the annual **OCC** (event
count yr⁻¹) used in the PPV calculation.

### Intensity — `alpha`, `beta`, `pi`

On wet days, intensity is modelled as a **K-component gamma mixture**:

```
y | wet  ~  Σ_k  π_k · Gamma(α_k, β_k)
```

where the rate parameterisation gives each component a mean of α_k / β_k.

- **`pi`** (softmax → sums to 1) chooses which gamma component to draw from.
  With K = 2 this reproduces the paper's gamma–gamma mixture.
- **`alpha`** (softplus + ε → strictly positive) sets the shape of each
  component; larger α gives a more symmetric, less heavy-tailed distribution.
- **`beta`** (softplus + ε → strictly positive) sets the rate (= 1/scale);
  the mean of component *k* is α_k / β_k.

During the Monte Carlo simulation a mixture sample gives `int_sim` (mm day⁻¹, 0
on dry days). Annual **TOT** (total accumulation yr⁻¹) is the yearly sum, and
**SII** (Simple Intensity Index, mm event⁻¹) is TOT / OCC.

### Training loss

The joint NLL combines both terms in one pass — no separate training loops:

```
L = −(1/N) Σ_n  [ 𝟙(dry)  · log(1 − P(wet))
                + 𝟙(wet)  · log P(wet)
                + 𝟙(wet)  · log Σ_k π_k · Γ(y_n ; α_k, β_k) ]
```

---

## From model outputs to PPV

The ZIG is a *stochastic weather model* — it captures day-to-day precipitation
variability driven by short-memory meteorological processes (seasonality and AR(1)
persistence in occurrence). Any **low-frequency variance** in the observations
that the model cannot replicate — such as inter-annual forcing from ENSO or PDO
— shows up as a gap between observed and simulated inter-annual variability. PPV
quantifies that gap.

### Step 1 — Monte Carlo ensemble

Run the trained ZIG forward for *S* = 1 000 realisations over the station record
length. Each member is time-stepped one day at a time: both the occurrence lag
buffer and the intensity lag buffer are updated with the simulated values, so the
short-memory structure is self-consistent across both channels. This produces
`occ_sim` (S × T) and `int_sim` (S × T).

### Step 2 — Annual metrics

For each year and each ensemble member compute:

| Metric | Formula | Units |
|--------|---------|-------|
| OCC | Σ occ per year | events yr⁻¹ |
| SII | Σ intensity / Σ occ per year | mm event⁻¹ |
| TOT | Σ intensity per year | mm yr⁻¹ |

Do the same for the observations to get one annual time series per metric.

### Step 3 — Raw PPV (Eq. 1)

For each metric, compute the inter-annual variance of the annual time series —
for each simulation member separately, then take the mean across members
(**variance first, mean second**):

```
PPV(var) = ( σ²_obs  −  mean_n[σ²_sim(n)] )  /  σ²_obs
```

PPV = 0 means all observed variance is explained by short-memory weather noise.
PPV → 1 means the stochastic baseline has near-zero inter-annual variance and
nearly all of the observed variance is potentially predictable. PPV cannot exceed 1.

### Step 4 — Normalized PPV and significance (Eqs. 7–8)

To test whether the observed PPV is significantly above the stochastic baseline,
treat each ensemble member as a pseudo-observation and apply Eq. 1 within the
ensemble:

```
PPV_sim(n) = ( σ²_sim(n)  −  mean_n[σ²_sim(n)] )  /  σ²_sim(n)
```

The 90th percentile of {PPV_sim(n)} is the directional p < 0.10 threshold.
**Normalized PPV = PPV_obs / threshold**; values > 1 are statistically
significant and correspond directly to the 1–4+ color scale in the paper's maps.

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
| 2 | Feature engineering | DOY (sin/cos) + lagged occurrence + lagged intensity → input tensor |
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
2. Run all cells in order. §9 / §9b depend on `occ_sim` / `int_sim` produced
   by the Monte Carlo cell in §6.

> **Tip:** Switch `DATA_FILE` to `"data/synthetic/enso_ar1_80yr.npz"` for a
> dataset where the ENSO-driven inter-annual signal is strong enough to produce
> clearly significant normalized PPV (> 1) for all three metrics.

---

## Synthetic data

### `enso_ar1_80yr.npz`

An 80-year daily record (29 200 days) designed so the ZIG captures some variance
sources but not others, producing non-trivial PPV:

| Component | Mechanism | Captured by ZIG? |
|-----------|-----------|------------------|
| **AR(1) occurrence persistence** | β_AR1 = 0.30 on occ[t−1] in logit P(wet) | ✅ yes — via `occ_lags` features |
| **Short-memory intensity persistence** | Today's intensity correlates with recent wet-day amounts | ✅ yes — via `int_lags` features |
| **Seasonal cycle** | Cosine amplitude ±0.6 logit units, peak DOY 1 | ✅ yes — via DOY sin/cos encoding |
| **ENSO forcing on P(wet)** | Annual AR(1) index (φ = 0.70) adds β_ENSO = 0.07 logit/SD | ❌ no — hidden from ZIG |
| **ENSO forcing on intensity** | Same index scales log-mean intensity by γ_ENSO = 0.03 log-mm/SD | ❌ no — hidden from ZIG |

Because the ZIG sees only 5-day occurrence and intensity lags plus DOY, it
captures short-memory weather persistence but has no access to the multi-year
ENSO signal. The unmodelled ENSO variance inflates σ²_obs relative to σ²_sim
and drives PPV > 0. The ENSO coefficients are intentionally weak so the
resulting normalized PPV sits in the 1.5–2 range — statistically significant
but not overwhelmingly dominant.

**Arrays saved:**

| Array | Shape | Description |
|-------|-------|-------------|
| `doy` | (N,) int32 | Day-of-year, 1–365 |
| `occ_lags` | (N, 5) float32 | Lagged occurrence (col 0 = oldest, col −1 = t−1) |
| `int_lags` | (N, 5) float32 | Lagged intensity in mm, 0 on dry days (same window as `occ_lags`) |
| `occ_obs` | (N,) float32 | Binary occurrence (0/1) |
| `int_obs` | (N,) float32 | Intensity in mm (0 on dry days) |
| `enso_daily` | (N,) float32 | Annual ENSO index at daily resolution |
| `enso_annual` | (80,) float32 | Annual ENSO index (for reference / plotting) |

Passing `enso_daily` as the `extra` argument to `build_features()` gives the ZIG
access to the ENSO signal, which should collapse PPV back toward zero — a useful
sanity check.

To regenerate or adjust parameters:

```bash
python data/synthetic/make_enso_ar1_data.py
```

---

## Input features

```
x  (N, 12)
│
├── sin(2π · doy / 365)     ← smooth seasonal cycle (periodic, no boundary discontinuity)
├── cos(2π · doy / 365)
├── occ[t−5]                ← AR(1) occurrence persistence: oldest lag
├── occ[t−4]
├── occ[t−3]
├── occ[t−2]
├── occ[t−1]                ← most recent occurrence lag (yesterday)
├── log1p(int[t−5])         ← lagged intensity: oldest lag, log1p-transformed
├── log1p(int[t−4])
├── log1p(int[t−3])
├── log1p(int[t−2])
└── log1p(int[t−1])         ← most recent intensity lag (yesterday)
```

Unlike the original SSWM — which models occurrence and intensity as two completely
separate stages — the network can condition on **both** occurrence *and* intensity
history simultaneously. The 5-day intensity lags are log1p-transformed before
concatenation: `log1p(x) = log(1 + x)` maps dry-day zeros cleanly to 0 while
compressing the right-skewed wet-day distribution to a model-friendly range.

Pass additional predictors (e.g., ENSO index, SST anomalies) via the `extra`
argument to `build_features()` — they are concatenated and `in_features` is
updated automatically. Adding a known low-frequency index as a feature is also a
way to quantify how much of the PPV that index explains.

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
