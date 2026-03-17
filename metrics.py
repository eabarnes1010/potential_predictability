"""Calibration, evaluation, and annual metrics functions."""

from typing import Tuple

import numpy as np
import torch

from model import ZIGammaMLP
from simulate import _sample_gamma_mixture


def reliability_diagram(
    prob_pred: np.ndarray,   # (N,)  predicted P(wet)
    y_obs: np.ndarray,       # (N,)  observed 0/1
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bin_centres, observed_freq, bin_counts), each shape (n_bins,).

    A perfectly calibrated model lies on the diagonal.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = np.empty(n_bins)
    freqs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (prob_pred >= edges[i]) & (prob_pred < edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            centres[i] = prob_pred[mask].mean()
            freqs[i] = (y_obs[mask] >= 1.0).mean()
        else:
            centres[i] = (edges[i] + edges[i + 1]) / 2.0
    return centres, freqs, counts


def expected_calibration_error(
    centres: np.ndarray,
    freqs: np.ndarray,
    counts: np.ndarray,
) -> float:
    """Weighted mean |predicted − observed|. ECE=0 is perfect; aim for <0.05."""
    n = counts.sum()
    if n == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(centres - freqs)) / n)


@torch.no_grad()
def intensity_qq_stats(
    model: ZIGammaMLP,
    x_wet: torch.Tensor,    # (M, F)  features on wet days only
    y_obs: np.ndarray,      # (M,)    observed wet-day intensities
    n_quantiles: int = 50,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Q-Q pairs for intensity calibration.

    Draws one predicted sample per observed wet day and compares quantiles.
    Well-calibrated model → points near the 1:1 line.
    Returns (q_obs, q_pred), each shape (n_quantiles,).
    """
    model.eval().to(device)
    _, pi, alpha, beta = model(x_wet.to(device))
    pred_samples = _sample_gamma_mixture(pi, alpha, beta).cpu().numpy()
    q = np.linspace(0.02, 0.98, n_quantiles)
    return np.quantile(y_obs, q), np.quantile(pred_samples, q)


def annual_metrics(
    occ_sim: np.ndarray,   # (S, T_sim)
    int_sim: np.ndarray,   # (S, T_sim)  0 on dry days
    occ_obs: np.ndarray,   # (T_obs,)
    int_obs: np.ndarray,   # (T_obs,)    0 on dry days
    days_per_year: int = 365,
) -> dict:
    """OCC / SII / TOT comparison: obs vs. ensemble mean ± 1σ.

    Replicates the three diagnostics used throughout Anderson et al. (2016).
    Sim and obs are annualized independently, so their lengths need not match.
    """
    n_years_sim = max(1, occ_sim.shape[1] // days_per_year)
    n_years_obs = max(1, len(occ_obs) // days_per_year)

    occ_s = occ_sim.sum(axis=1) / n_years_sim
    acc_s = int_sim.sum(axis=1) / n_years_sim
    occ_o = occ_obs.sum() / n_years_obs
    acc_o = int_obs.sum() / n_years_obs
    sii_s = np.where(occ_s > 0, acc_s / occ_s, 0.0)
    sii_o = acc_o / occ_o if occ_o > 0 else 0.0

    out: dict = {}
    for name, sim_v, obs_v in [
        ("OCC", occ_s, occ_o),
        ("SII", sii_s, sii_o),
        ("TOT", acc_s, acc_o),
    ]:
        out[name] = {
            "obs": float(obs_v),
            "sim_mean": float(sim_v.mean()),
            "sim_std": float(sim_v.std()),
        }
    return out


def compute_annual_metrics_ts(
    occ: np.ndarray,
    intensity: np.ndarray,
    days_per_year: int = 365,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-year OCC, SII, TOT for a single daily time series.

    Parameters
    ----------
    occ : (T,) binary occurrence (0/1)
    intensity : (T,) intensity in mm (0 on dry days)
    days_per_year : length of one annual window

    Returns
    -------
    ann_occ : (n_years,)  occurrence count per year
    ann_sii : (n_years,)  simple intensity index (mm event^-1) per year
    ann_tot : (n_years,)  total accumulation (mm yr^-1) per year
    """
    n_years = len(occ) // days_per_year
    occ = np.asarray(occ, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)

    ann_occ = np.empty(n_years, dtype=np.float64)
    ann_sii = np.empty(n_years, dtype=np.float64)
    ann_tot = np.empty(n_years, dtype=np.float64)

    for y in range(n_years):
        sl = slice(y * days_per_year, (y + 1) * days_per_year)
        o = occ[sl]
        i = intensity[sl]
        ann_occ[y] = o.sum()
        ann_tot[y] = i.sum()
        ann_sii[y] = float(i.sum() / o.sum()) if o.sum() > 0 else 0.0

    return ann_occ, ann_sii, ann_tot


def compute_ppv(
    occ_obs: np.ndarray,
    int_obs: np.ndarray,
    occ_sim: np.ndarray,
    int_sim: np.ndarray,
    days_per_year: int = 365,
) -> dict:
    """Compute PPV for OCC, SII, and TOT per Anderson et al. (2016) Eq. 1.

    The ensemble-mean variance is computed AFTER the per-member variance
    calculation (i.e., var first, then mean across members), matching the
    paper's description of the overbar in Eq. 1.

    Parameters
    ----------
    occ_obs : (T_obs,)    observed binary occurrence
    int_obs : (T_obs,)    observed intensity, mm (0 on dry days)
    occ_sim : (S, T_sim)  simulated binary occurrence, S ensemble members
    int_sim : (S, T_sim)  simulated intensity, mm (0 on dry days)
    days_per_year : days in one annual window

    Returns
    -------
    dict keyed by "OCC", "SII", "TOT"; each entry contains:
        var_obs        : float  observed inter-annual variance
        var_sim_each   : (S,)   per-member inter-annual variance
        mean_var_sim   : float  ensemble-mean variance (mean of per-member vars)
        ppv            : float  PPV scalar
        ann_obs        : (n_years,) observed annual values (for plotting)
        ann_sim        : (S, n_years_sim) simulated annual values (for plotting)
    """
    # ── Observed annual metrics ────────────────────────────────────────────────
    ann_occ_obs, ann_sii_obs, ann_tot_obs = compute_annual_metrics_ts(
        occ_obs, int_obs, days_per_year
    )
    var_obs = {
        "OCC": float(np.var(ann_occ_obs, ddof=1)),
        "SII": float(np.var(ann_sii_obs, ddof=1)),
        "TOT": float(np.var(ann_tot_obs, ddof=1)),
    }
    ann_obs = {"OCC": ann_occ_obs, "SII": ann_sii_obs, "TOT": ann_tot_obs}

    # ── Per-member simulated annual metrics ────────────────────────────────────
    S = occ_sim.shape[0]
    n_years_sim = occ_sim.shape[1] // days_per_year

    var_sim_each = {k: np.empty(S, dtype=np.float64) for k in ("OCC", "SII", "TOT")}
    ann_sim = {
        k: np.empty((S, n_years_sim), dtype=np.float64) for k in ("OCC", "SII", "TOT")
    }

    for s in range(S):
        ann_occ_s, ann_sii_s, ann_tot_s = compute_annual_metrics_ts(
            occ_sim[s], int_sim[s], days_per_year
        )
        # Variance of annual values for this member (mean BEFORE var would be wrong)
        var_sim_each["OCC"][s] = float(np.var(ann_occ_s, ddof=1))
        var_sim_each["SII"][s] = float(np.var(ann_sii_s, ddof=1))
        var_sim_each["TOT"][s] = float(np.var(ann_tot_s, ddof=1))
        ann_sim["OCC"][s] = ann_occ_s
        ann_sim["SII"][s] = ann_sii_s
        ann_sim["TOT"][s] = ann_tot_s

    # ── Mean of per-member variances (Eq. 1 overbar) ──────────────────────────
    results = {}
    for key in ("OCC", "SII", "TOT"):
        mean_var_sim = float(var_sim_each[key].mean())
        v_obs = var_obs[key]
        ppv = (v_obs - mean_var_sim) / v_obs if v_obs > 0 else float("nan")
        results[key] = {
            "var_obs": v_obs,
            "var_sim_each": var_sim_each[key],
            "mean_var_sim": mean_var_sim,
            "ppv": ppv,
            "ann_obs": ann_obs[key],
            "ann_sim": ann_sim[key],
        }

    return results


def compute_normalized_ppv(ppv_results: dict) -> dict:
    """Normalize PPV by the p=0.10 significance threshold (Anderson et al. Eq. 7).

    For each ensemble member n, the within-ensemble PPV is:

        PPV_sim(n) = (sigma2_sim(n) - mean(sigma2_sim)) / sigma2_sim(n)

    The 90th percentile of this null distribution is the directional p < 0.10
    threshold.  Normalized PPV = PPV_obs / threshold; values > 1 are
    statistically significant.

    Parameters
    ----------
    ppv_results : dict returned by compute_ppv (must contain "var_sim_each",
                  "mean_var_sim", and "ppv" for each metric key)

    Returns
    -------
    Augmented dict with three new keys per metric:
        ppv_sim_each   : (S,) null PPV for each ensemble member (Eq. 7)
        threshold_p10  : float  90th-percentile significance threshold
        ppv_normalized : float  normalized PPV (> 1 = significant at p < 0.10)
    """
    results = {}
    for key, d in ppv_results.items():
        var_sim_each = d["var_sim_each"]   # (S,) per-member variances
        mean_var_sim = d["mean_var_sim"]   # scalar ensemble-mean variance

        # Eq. 7: treat each member as the "observation" against the ensemble mean
        # Guard against zero-variance members (rare edge case with small ensembles)
        # by pre-substituting 1.0 in the denominator before masking — avoids
        # RuntimeWarning from numpy evaluating both np.where branches eagerly.
        safe_mask = var_sim_each > 0
        safe_denom = np.where(safe_mask, var_sim_each, 1.0)
        ppv_sim_each = np.where(
            safe_mask,
            (var_sim_each - mean_var_sim) / safe_denom,
            np.nan,
        )

        # One-sided p < 0.10 threshold = 90th percentile of null distribution
        threshold_p10 = float(np.nanpercentile(ppv_sim_each, 90))

        # Normalized PPV; NaN if threshold is non-positive (degenerate ensemble)
        if threshold_p10 > 0:
            ppv_norm = d["ppv"] / threshold_p10
        else:
            ppv_norm = float("nan")

        results[key] = {
            **d,
            "ppv_sim_each": ppv_sim_each,
            "threshold_p10": threshold_p10,
            "ppv_normalized": ppv_norm,
        }

    return results
