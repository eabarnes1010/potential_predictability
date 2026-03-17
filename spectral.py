"""Spectral analysis and frequency decomposition of potential predictability."""

from scipy import signal as scipy_signal
import numpy as np


# Frequency band definitions: (lower_freq_exclusive, upper_freq_inclusive)
# in cycles per year; period = 1/freq years.
_FREQ_BANDS: dict[str, tuple[float, float]] = {
    "2-7yr": (1.0 / 7.0, 1.0 / 2.0),
    "7-20yr": (1.0 / 20.0, 1.0 / 7.0),
    "20-40yr": (1.0 / 40.0, 1.0 / 20.0),
    ">40yr": (0.0, 1.0 / 40.0),
}
# Band order used throughout this section
BAND_ORDER = ["2-7yr", "7-20yr", "20-40yr", ">40yr", "trend"]


def multitaper_psd(
    ts_det: np.ndarray,
    NW: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Multitaper one-sided PSD via the Thomson (1982) method.

    Uses K = 2*NW - 1 DPSS tapers.  The result is normalised so that
    ``psd.sum() == np.var(ts_det, ddof=1)``, making it consistent with
    Anderson et al. (2016) Eq. 2 (σ² = Σ A(ω_k)²/2) where
    psd[k] = A(ω_k)²/2.

    Parameters
    ----------
    ts_det : (N,)
        Linearly detrended annual time series.
    NW : float
        Time-half-bandwidth product (default 4 → 7 tapers).

    Returns
    -------
    freqs : (M,)
        Positive frequencies in cycles per year; DC bin excluded.
    psd : (M,)
        Normalised one-sided PSD; sums to ``np.var(ts_det, ddof=1)``.
    """
    N = len(ts_det)
    K = int(2 * NW) - 1
    tapers, _ = scipy_signal.windows.dpss(N, NW, Kmax=K, return_ratios=True)

    # FFT of all tapered copies simultaneously
    tapered = tapers * ts_det[np.newaxis, :]  # (K, N)
    Xk = np.fft.rfft(tapered, n=N, axis=-1)  # (K, N//2+1)

    # Average squared magnitudes across tapers
    raw = np.mean(np.abs(Xk) ** 2, axis=0)  # (N//2+1,)

    # Two-sided → one-sided: double all non-DC, non-Nyquist bins
    psd = raw.copy()
    if N % 2 == 0:
        psd[1:-1] *= 2.0  # keep DC and Nyquist undoubled
    else:
        psd[1:] *= 2.0  # keep DC only undoubled

    # Drop DC bin (freq = 0; mean already removed by detrending)
    freqs = np.fft.rfftfreq(N, d=1.0)[1:]
    psd = psd[1:]

    # Normalise so psd.sum() == var(ts_det)
    var_det = float(np.var(ts_det, ddof=1))
    total = float(psd.sum())
    if total > 0.0:
        psd = psd * (var_det / total)

    return freqs, psd


def spectral_ppv_decompose(
    ann_obs: np.ndarray,
    ann_sim: np.ndarray,
    var_obs_full: float | None = None,
    NW: float = 4.0,
) -> tuple[dict, dict]:
    """Decompose PPV into frequency-band contributions (Anderson 2016 Eqs. 2–6).

    The total PPV partitions exactly as:
        PPV = Σ_bands PP(band) + PP(trend)            [Eqs. 3 + 6]

    Observed and simulated series **must** have the same number of years;
    an AssertionError is raised otherwise.

    Parameters
    ----------
    ann_obs : (n_years,)
        Observed annual time series (OCC, SII, or TOT).
    ann_sim : (S, n_years)
        Simulated annual time series, S ensemble members.
    var_obs_full : float or None
        Pre-computed σ²_obs (ddof=1); computed here when None.
    NW : float
        Multitaper time-half-bandwidth product passed to ``multitaper_psd``.

    Returns
    -------
    band_results : dict
        Keyed by "2-7yr", "7-20yr", "20-40yr", ">40yr", "trend".  Each entry:
            pp_obs        float  observed band-integrated PP
            pp_null       (S,)   within-ensemble null PP (Eq. 7 analog)
            threshold_p10 float  90th percentile of pp_null
            ppv_normalized float pp_obs / threshold_p10  (>1 → p<0.10)
            n_freqs       int    discrete frequencies in this band (0 for trend)
    spectral_meta : dict
        Diagnostic arrays: freqs, psd_obs, psd_sim_mean, pp_per_freq,
        var_obs_full, var_obs_trend, pp_sum_check.
    """
    n_years = len(ann_obs)
    S = ann_sim.shape[0]

    # ── Dimension check ──────────────────────────────────────────────────────
    assert ann_sim.shape[1] == n_years, (
        f"Observed ({n_years} yr) and simulated ({ann_sim.shape[1]} yr) "
        f"annual series must have the same length."
    )

    if var_obs_full is None:
        var_obs_full = float(np.var(ann_obs, ddof=1))

    # ── Detrend observed series ──────────────────────────────────────────────
    ts_obs_det = scipy_signal.detrend(ann_obs, type="linear")
    var_obs_det = float(np.var(ts_obs_det, ddof=1))
    var_obs_trend = var_obs_full - var_obs_det  # Eq. 5 residual

    # ── Observed PSD ─────────────────────────────────────────────────────────
    freqs, psd_obs = multitaper_psd(ts_obs_det, NW=NW)

    # ── Detrend simulated ensemble (vectorised over members) ─────────────────
    ts_sim_det = scipy_signal.detrend(  # (S, N)
        ann_sim, type="linear", axis=1
    )
    var_sim_full = np.var(ann_sim, axis=1, ddof=1)     # (S,)
    var_sim_det = np.var(ts_sim_det, axis=1, ddof=1)   # (S,)
    var_sim_trend = var_sim_full - var_sim_det          # (S,), Eq. 5 per member

    # ── Simulated PSD (vectorised across members) ────────────────────────────
    K = int(2 * NW) - 1
    tapers, _ = scipy_signal.windows.dpss(
        n_years, NW, Kmax=K, return_ratios=True
    )
    # tapers : (K, N)
    # Broadcast: (S, K, N) = (S, 1, N) * (1, K, N)
    tapered_sim = ts_sim_det[:, np.newaxis, :] * tapers[np.newaxis, :, :]
    Xk_sim = np.fft.rfft(tapered_sim, n=n_years, axis=-1)  # (S, K, N//2+1)
    raw_sim = np.mean(np.abs(Xk_sim) ** 2, axis=1)          # (S, N//2+1)

    # Two-sided → one-sided and drop DC
    if n_years % 2 == 0:
        raw_sim[:, 1:-1] *= 2.0
    else:
        raw_sim[:, 1:] *= 2.0
    raw_sim = raw_sim[:, 1:]  # drop DC; shape (S, len(freqs))

    # Normalise each member's PSD so its row sums to var_sim_det[s]
    row_sums = raw_sim.sum(axis=1, keepdims=True)  # (S, 1)
    safe = row_sums > 0.0
    psd_sim = np.where(
        safe,
        raw_sim * var_sim_det[:, np.newaxis] / np.where(safe, row_sums, 1.0),
        0.0,
    )  # (S, n_freqs)

    # ── Ensemble-mean PSD ────────────────────────────────────────────────────
    mean_psd_sim = psd_sim.mean(axis=0)  # (n_freqs,)

    # ── Per-frequency PP (Eq. 4, using psd[k] = A(ω_k)²/2) ──────────────────
    # PP(var, ω_k) = (psd_obs[k] - mean_psd_sim[k]) / σ²_obs_full
    pp_per_freq = (psd_obs - mean_psd_sim) / var_obs_full  # (n_freqs,)

    # ── Band-integrated PP and within-ensemble null distributions ────────────
    band_results: dict = {}
    for label, (freq_lo, freq_hi) in _FREQ_BANDS.items():
        # >40yr band: take everything above zero up to freq_hi
        if label == ">40yr":
            mask = (freqs > 0.0) & (freqs <= freq_hi)
        else:
            mask = (freqs > freq_lo) & (freqs <= freq_hi)

        pp_obs_band = float(pp_per_freq[mask].sum())

        # Null: treat each member n as its own "observation" (Eq. 7 analog)
        # PP_sim(n, band) = Σ_{k∈band}(psd_sim[n,k] − mean_psd_sim[k]) /
        # σ²_sim_full[n]
        safe_v = var_sim_full > 0.0
        pp_null = np.where(
            safe_v,
            np.sum(psd_sim[:, mask] - mean_psd_sim[mask], axis=1)
            / np.where(safe_v, var_sim_full, 1.0),
            np.nan,
        )

        threshold_p10 = float(np.nanpercentile(pp_null, 90))
        ppv_norm = (
            pp_obs_band / threshold_p10 if threshold_p10 > 0.0 else float("nan")
        )
        band_results[label] = {
            "pp_obs": pp_obs_band,
            "pp_null": pp_null,
            "threshold_p10": threshold_p10,
            "ppv_normalized": ppv_norm,
            "n_freqs": int(mask.sum()),
        }

    # ── Secular trend (Eq. 6) ────────────────────────────────────────────────
    mean_var_sim_trend = float(var_sim_trend.mean())
    pp_obs_trend = (var_obs_trend - mean_var_sim_trend) / var_obs_full

    safe_v = var_sim_full > 0.0
    pp_null_trend = np.where(
        safe_v,
        (var_sim_trend - mean_var_sim_trend)
        / np.where(safe_v, var_sim_full, 1.0),
        np.nan,
    )
    threshold_trend_p10 = float(np.nanpercentile(pp_null_trend, 90))
    ppv_norm_trend = (
        pp_obs_trend / threshold_trend_p10
        if threshold_trend_p10 > 0.0
        else float("nan")
    )
    band_results["trend"] = {
        "pp_obs": float(pp_obs_trend),
        "pp_null": pp_null_trend,
        "threshold_p10": threshold_trend_p10,
        "ppv_normalized": ppv_norm_trend,
        "n_freqs": 0,
    }

    # ── Diagnostic metadata ──────────────────────────────────────────────────
    spectral_meta = {
        "freqs": freqs,
        "psd_obs": psd_obs,
        "psd_sim_mean": mean_psd_sim,
        "pp_per_freq": pp_per_freq,
        "var_obs_full": var_obs_full,
        "var_obs_trend": var_obs_trend,
        "pp_sum_check": sum(v["pp_obs"] for v in band_results.values()),
    }

    return band_results, spectral_meta
