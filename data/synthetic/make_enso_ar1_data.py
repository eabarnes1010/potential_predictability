"""make_enso_ar1_data.py
======================
Generate synthetic daily precipitation with two distinct variance components:

1. **AR(1) short-memory noise** — day-to-day occurrence persistence captured by
   lagged occurrence indicators.  This is the variance a well-fitted ZIG/SSWM
   *can* explain (stochastic baseline for PPV).

2. **ENSO-type low-frequency signal** — an AR(1) process operating at the
   *annual* scale (φ_annual = 0.70 → ~2–3 yr e-folding time) that modulates
   both P(wet) and mean intensity.  Because the ZIG model's feature set
   contains only 5-day occurrence lags and seasonal DOY encoding, it cannot
   detect or replicate this multi-year signal.  The resulting "leftover"
   inter-annual variance is what drives PPV > 0.

Data-generating process
-----------------------
For year y, draw the annual ENSO index:

    E[y] = φ_annual * E[y-1]  +  ε[y],   ε ~ N(0, σ_ε²)
    σ_ε = sqrt(1 - φ_annual²)  → stationary variance ≡ 1

For each day t in year y:

    logit(p_wet[t]) = β0  +  β_seas * cos(2π·doy/365)
                           +  β_ENSO * E[y]
                           +  β_AR1  * occ[t-1]

    intensity[t]    ~ Gamma(shape=α, rate=α / μ[t]),   μ[t] = μ0 * exp(γ_ENSO * E[y])
    (only sampled when occ[t] = 1)

Saved arrays (same schema as other .npz files in this directory)
----------------------------------------------------------------
    doy          : (N,)        int32   day-of-year, 1..365
    occ_lags     : (N, L)      float32 L=5 lagged occurrence indicators
    occ_obs      : (N,)        float32 binary occurrence (0/1)
    int_obs      : (N,)        float32 intensity in mm (0 on dry days)
    enso_daily   : (N,)        float32 ENSO index repeated at daily resolution
    enso_annual  : (N_YEARS,)  float32 annual ENSO index (for plots / analysis)

Usage
-----
    python make_enso_ar1_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Record length ──────────────────────────────────────────────────────────────
N_YEARS: int = 80  # Long enough for ~10–15 ENSO cycles and reliable PPV estimates
DAYS_PER_YEAR: int = 365
N_DAYS: int = N_YEARS * DAYS_PER_YEAR
N_LAGS: int = 5  # Matches default in zig_precip.ipynb
SEED: int = 42

# ── Seasonal baseline for P(wet) ──────────────────────────────────────────────
# expit(β0) ≈ 0.40 base wet fraction away from peak season
BETA0: float = -0.40
# Cosine amplitude; DOY 1 (1 Jan) is the seasonal peak
BETA_SEAS: float = 0.60  # ±0.6 logit units seasonal swing

# ── ENSO parameters ────────────────────────────────────────────────────────────
ENSO_PHI: float = 0.70  # Annual lag-1 autocorrelation → ~2.5 yr e-folding time
# σ chosen so the stationary variance of the ENSO process equals 1.0
ENSO_SIGMA: float = float(np.sqrt(1.0 - ENSO_PHI**2))

# Effect of ENSO (in units of its SD) on logit P(wet)
# +1 SD ENSO → expit(-0.4 + 0.8) ≈ 0.60 wet fraction (vs 0.40 at baseline)
# -1 SD ENSO → expit(-0.4 - 0.8) ≈ 0.23 wet fraction
BETA_ENSO_OCC: float = 0.80

# Effect of ENSO on log(mean intensity): ±25 % per SD
GAMMA_ENSO_INT: float = 0.25

# ── Intensity baseline ─────────────────────────────────────────────────────────
INT_MEAN_BASE: float = 10.0  # mm / event, climatological mean intensity
INT_SHAPE: float = 2.0  # Gamma shape (rate = shape / mean_t)

# ── AR(1) day-to-day occurrence persistence ────────────────────────────────────
# Captured by the ZIG model via occ_lags → this is the stochastic baseline
BETA_AR1: float = 0.30  # contribution of occ[t-1] to logit P(wet)

# ── Wet-day threshold ─────────────────────────────────────────────────────────
WET_THRESHOLD: float = 1.0  # mm, consistent with Anderson et al. (2016)


# ── Helper: fast logistic (scalar-safe) ────────────────────────────────────────
def _expit(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_enso_index(
    n_years: int,
    phi: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a stationary AR(1) annual ENSO index.

    Returns shape (n_years,) with zero mean and unit variance.
    """
    enso = np.zeros(n_years)
    for y in range(1, n_years):
        enso[y] = phi * enso[y - 1] + rng.normal(0.0, sigma)
    # Standardize to exactly unit variance over the record
    enso = (enso - enso.mean()) / enso.std()
    return enso.astype(np.float32)


def generate_daily_precip(
    enso_annual: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate daily precipitation from the causal model described above.

    Parameters
    ----------
    enso_annual : (N_YEARS,) standardized annual ENSO index
    rng         : seeded numpy Generator

    Returns
    -------
    doy_arr      : (N_DAYS,)       int32   day-of-year 1..365
    occ_obs      : (N_DAYS,)       float32 binary occurrence
    int_obs      : (N_DAYS,)       float32 intensity mm (0 on dry days)
    occ_lags_arr : (N_DAYS, L)     float32 lagged occurrence features
    enso_daily   : (N_DAYS,)       float32 annual ENSO index at daily res
    """
    doy_arr = np.tile(np.arange(1, DAYS_PER_YEAR + 1), N_YEARS).astype(np.int32)

    # Pre-compute DOY angle for seasonality term
    doy_angle = 2.0 * np.pi * (doy_arr - 1.0) / DAYS_PER_YEAR  # (N_DAYS,)

    # Expand ENSO to daily (constant within each year)
    enso_daily = np.repeat(enso_annual, DAYS_PER_YEAR).astype(np.float32)

    occ_obs = np.zeros(N_DAYS, dtype=np.float32)
    int_obs = np.zeros(N_DAYS, dtype=np.float32)
    occ_lags_arr = np.zeros((N_DAYS, N_LAGS), dtype=np.float32)

    # Rolling lag buffer: [oldest, ..., most recent] — same convention as notebook
    lag_buf = np.zeros(N_LAGS, dtype=np.float32)

    for t in range(N_DAYS):
        # Record lag state BEFORE updating with today
        occ_lags_arr[t] = lag_buf.copy()

        # Logit P(wet): seasonal + ENSO + AR(1) persistence from yesterday
        logit_p = (
            BETA0
            + BETA_SEAS * np.cos(doy_angle[t])  # seasonal; peak at DOY 1
            + BETA_ENSO_OCC * enso_daily[t]  # ENSO forcing — hidden from ZIG
            + BETA_AR1 * lag_buf[-1]  # AR(1): yesterday's occurrence
        )
        p_wet = _expit(logit_p)

        # Stochastic occurrence draw
        occ_t = float(rng.random() < p_wet)
        occ_obs[t] = occ_t

        if occ_t > 0.0:
            # Log-mean intensity modulated by ENSO
            log_mean = np.log(INT_MEAN_BASE) + GAMMA_ENSO_INT * enso_daily[t]
            mean_int = np.exp(log_mean)
            # Gamma(shape=α, rate=α/mean) so that E[X] = mean_int
            rate = INT_SHAPE / mean_int
            sample = rng.gamma(INT_SHAPE, 1.0 / rate)
            int_obs[t] = max(float(sample), WET_THRESHOLD)

        # Roll lag buffer left; append today's occurrence in last slot
        lag_buf = np.roll(lag_buf, shift=-1)
        lag_buf[-1] = occ_t

    return doy_arr, occ_obs, int_obs, occ_lags_arr, enso_daily


def print_summary(
    occ_obs: np.ndarray,
    int_obs: np.ndarray,
    enso_annual: np.ndarray,
) -> None:
    """Print data-quality diagnostics to verify the generation."""
    wet_mask = occ_obs > 0
    wet_frac = wet_mask.mean()
    mean_int = int_obs[wet_mask].mean() if wet_mask.any() else float("nan")

    # Annual OCC and TOT
    ann_occ = occ_obs.reshape(N_YEARS, DAYS_PER_YEAR).sum(axis=1)
    ann_tot = int_obs.reshape(N_YEARS, DAYS_PER_YEAR).sum(axis=1)
    ann_sii = np.where(ann_occ > 0, ann_tot / ann_occ, 0.0)

    print(f"  Record length : {N_DAYS:,} days ({N_YEARS} years)")
    print(f"  Wet fraction  : {wet_frac:.3f}")
    print(f"  Mean intensity (wet days): {mean_int:.2f} mm")
    print()
    print("  Annual metrics — interannual variability:")
    print(f"    OCC: mean={ann_occ.mean():.1f}  std={ann_occ.std():.1f} events/yr")
    print(f"    SII: mean={ann_sii.mean():.2f}  std={ann_sii.std():.2f} mm/event")
    print(f"    TOT: mean={ann_tot.mean():.1f}  std={ann_tot.std():.1f} mm/yr")
    print()
    print(
        f"  ENSO index: min={enso_annual.min():.2f}  "
        f"max={enso_annual.max():.2f}  "
        f"AR(1)={float(np.corrcoef(enso_annual[:-1], enso_annual[1:])[0, 1]):.3f}"
    )


def verify_ppv(
    occ_obs: np.ndarray,
    int_obs: np.ndarray,
    enso_annual: np.ndarray,
) -> None:
    """Compute PPV using a null simulation (no ENSO) to confirm PPV > 0.

    The null simulation mirrors what the ZIG/SSWM would produce: it uses the
    climatological (DOY-only) P(wet) with AR(1) persistence but *without* any
    ENSO forcing.  This is the stochastic baseline — σ²_sim from Eq. 1.
    """
    rng_null = np.random.default_rng(SEED + 1000)
    S_NULL = 200  # enough members for a stable mean(σ²_sim) estimate

    # Null simulation: same AR(1) persistence, same seasonality, but ENSO = 0
    doy_angle = 2.0 * np.pi * (np.arange(DAYS_PER_YEAR) / DAYS_PER_YEAR)

    def _null_sim_one(seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
        rng_s = np.random.default_rng(SEED + seed_offset)
        occ_s = np.zeros(N_DAYS, dtype=np.float32)
        int_s = np.zeros(N_DAYS, dtype=np.float32)
        lag = 0.0
        for t in range(N_DAYS):
            doy_t = t % DAYS_PER_YEAR
            logit_p = BETA0 + BETA_SEAS * np.cos(doy_angle[doy_t]) + BETA_AR1 * lag
            occ_t = float(rng_s.random() < _expit(logit_p))
            occ_s[t] = occ_t
            if occ_t > 0.0:
                sample = rng_s.gamma(INT_SHAPE, INT_MEAN_BASE / INT_SHAPE)
                int_s[t] = max(float(sample), WET_THRESHOLD)
            lag = occ_t
        return occ_s, int_s

    # Annual metrics for observed record
    def _annual_ts(occ: np.ndarray, inten: np.ndarray):
        ann_occ = occ.reshape(N_YEARS, DAYS_PER_YEAR).sum(axis=1)
        ann_tot = inten.reshape(N_YEARS, DAYS_PER_YEAR).sum(axis=1)
        ann_sii = np.where(ann_occ > 0, ann_tot / ann_occ, 0.0)
        return ann_occ, ann_sii, ann_tot

    ann_occ_obs, ann_sii_obs, ann_tot_obs = _annual_ts(occ_obs, int_obs)
    var_obs = {
        "OCC": float(np.var(ann_occ_obs, ddof=1)),
        "SII": float(np.var(ann_sii_obs, ddof=1)),
        "TOT": float(np.var(ann_tot_obs, ddof=1)),
    }

    # Per-member variances from null simulations (variance first, then mean)
    var_sim_each = {k: np.empty(S_NULL) for k in ("OCC", "SII", "TOT")}
    print(f"\n  Running {S_NULL} null simulations (no ENSO) to verify PPV …")
    for s in range(S_NULL):
        occ_s, int_s = _null_sim_one(s)
        a, b, c = _annual_ts(occ_s, int_s)
        var_sim_each["OCC"][s] = np.var(a, ddof=1)
        var_sim_each["SII"][s] = np.var(b, ddof=1)
        var_sim_each["TOT"][s] = np.var(c, ddof=1)

    print()
    print("  PPV verification (Eq. 1 with null / no-ENSO simulation):")
    print(f"  {'Metric':<6}  {'σ²_obs':>10}  {'mean(σ²_sim)':>12}  {'PPV':>8}")
    print("  " + "-" * 44)
    for key in ("OCC", "SII", "TOT"):
        mean_var_sim = var_sim_each[key].mean()
        v_obs = var_obs[key]
        ppv = (v_obs - mean_var_sim) / v_obs if v_obs > 0 else float("nan")
        flag = " ✓" if ppv > 0.05 else " !"
        print(
            f"  {key:<6}  {v_obs:>10.3f}  {mean_var_sim:>12.3f}  {ppv:>8.3f}{flag}"
        )
    print()
    print("  ✓ = PPV > 0.05  (clearly above stochastic baseline)")


def main() -> None:
    """Generate and save the ENSO + AR(1) precipitation dataset."""
    out_dir = Path(__file__).parent
    out_path = out_dir / "enso_ar1_80yr.npz"

    print("Generating ENSO + AR(1) synthetic precipitation data")
    print("=" * 55)
    print()
    print("Causal structure:")
    print(f"  AR(1) persistence : β_AR1 = {BETA_AR1}  (captured by ZIG/SSWM)")
    print(f"  Seasonal cycle    : β_seas = ±{BETA_SEAS} logit (captured by ZIG/SSWM)")
    print(
        f"  ENSO forcing      : β_ENSO = {BETA_ENSO_OCC} logit/SD  "
        f"(HIDDEN from ZIG → drives PPV)"
    )
    print(
        f"  ENSO intensity    : γ_ENSO = {GAMMA_ENSO_INT} log-mm/SD  "
        f"(HIDDEN from ZIG → drives PPV)"
    )
    print(f"  ENSO AR(1) coeff  : φ = {ENSO_PHI}  (5–7 yr characteristic period)")
    print()

    rng = np.random.default_rng(SEED)

    print("Step 1/3: Generating annual ENSO index …")
    enso_annual = generate_enso_index(N_YEARS, ENSO_PHI, ENSO_SIGMA, rng)

    print("Step 2/3: Generating daily precipitation record …")
    doy_arr, occ_obs, int_obs, occ_lags_arr, enso_daily = generate_daily_precip(
        enso_annual, rng
    )

    print("Step 3/3: Saving data …")
    np.savez(
        out_path,
        doy=doy_arr,
        occ_lags=occ_lags_arr,
        occ_obs=occ_obs,
        int_obs=int_obs,
        enso_daily=enso_daily,
        enso_annual=enso_annual,
    )
    print(f"  Saved → {out_path}")
    print()

    print("Data summary:")
    print_summary(occ_obs, int_obs, enso_annual)

    verify_ppv(occ_obs, int_obs, enso_annual)

    print("Done.")


if __name__ == "__main__":
    main()
