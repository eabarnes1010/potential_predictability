"""Plotting functions for ZIG model evaluation and diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from metrics import reliability_diagram, expected_calibration_error


def plot_loss_curves(
    history: dict,
) -> None:
    """Plot training and validation loss curves.

    Parameters
    ----------
    history : dict
        Training history dict from train_zig() with keys "train" and
        optionally "val" and "best_epoch".
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="Train NLL", color="#4f46e5")

    if "val" in history:
        ax.plot(epochs, history["val"], label="Val NLL", color="#f59e0b")
        best = history.get("best_epoch")
        if best:
            ax.axvline(
                best, color="#ef4444", linestyle="--", linewidth=1,
                label=f"Best epoch ({best})"
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Joint NLL")
    ax.set_title("ZIG training / validation loss")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_single_member_intensity(
    int_sim: np.ndarray,
    member: int = 0,
    max_days: int = 3000,
) -> None:
    """Plot simulated daily precipitation intensity for one ensemble member.

    Parameters
    ----------
    int_sim : (n_simulations, T)
        Simulated intensity array.
    member : int
        Which ensemble member to plot (default 0).
    max_days : int
        Maximum number of days to display (default 3000).
    """
    series = int_sim[member, :]
    xmax = min(max_days, series.shape[0])

    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
    ax.plot(series[:xmax], color="#1f77b4", lw=1.2, alpha=0.9, label=f"Sim member {member}")
    ax.fill_between(np.arange(xmax), series[:xmax], 0, color="#1f77b4", alpha=0.12)

    ax.set_xlim(0, xmax - 1)
    ax.set_title("Simulated Daily Precipitation Intensity (Member 0)", pad=10)
    ax.set_xlabel("Day index")
    ax.set_ylabel("Intensity (mm)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_reliability_diagram(
    prob_pred: np.ndarray,
    y_obs: np.ndarray,
    n_bins: int = 10,
) -> None:
    """Plot occurrence reliability diagram.

    Parameters
    ----------
    prob_pred : (N,)
        Predicted P(wet) from model.
    y_obs : (N,)
        Observed 0/1 binary occurrence.
    n_bins : int
        Number of probability bins (default 10).
    """
    centres, freqs, counts = reliability_diagram(prob_pred, y_obs, n_bins=n_bins)
    ece = expected_calibration_error(centres, freqs, counts)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    ax.scatter(centres, freqs, s=counts / counts.max() * 200, zorder=3)
    ax.set_xlabel("Mean predicted P(wet)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability diagram  (ECE={ece:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_intensity_qq(
    q_obs: np.ndarray,
    q_pred: np.ndarray,
) -> None:
    """Plot intensity Q-Q plot for calibration.

    Parameters
    ----------
    q_obs : (n_quantiles,)
        Observed intensity quantiles.
    q_pred : (n_quantiles,)
        Predicted intensity quantiles.
    """
    qq_rmse = float(np.sqrt(np.mean((q_obs - q_pred) ** 2)))

    lim = max(q_obs.max(), q_pred.max()) * 1.05
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1")
    ax.scatter(q_obs, q_pred, s=20, zorder=3)
    ax.set_xlabel("Observed quantile (mm)")
    ax.set_ylabel("Predicted quantile (mm)")
    ax.set_title(f"Intensity Q-Q  (RMSE={qq_rmse:.2f} mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ppv_anderson(ppv_results: dict) -> None:
    """Plot PPV per Anderson et al. (2016) Eq. 1.

    Parameters
    ----------
    ppv_results : dict
        Results dict from compute_ppv() with keys "OCC", "SII", "TOT".
    """
    LABELS = {
        "OCC": "Occurrence\n(events yr⁻¹)",
        "SII": "SII\n(mm event⁻¹)",
        "TOT": "Total\n(mm yr⁻¹)",
    }
    COLORS = {"OCC": "#2166ac", "SII": "#d6604d", "TOT": "#4dac26"}

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Potential Predictable Variance — Anderson et al. (2016) Eq. 1",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    years_obs = np.arange(1, len(ppv_results["OCC"]["ann_obs"]) + 1)
    years_sim = np.arange(1, ppv_results["OCC"]["ann_sim"].shape[1] + 1)

    for row, key in enumerate(("OCC", "SII", "TOT")):
        d = ppv_results[key]
        color = COLORS[key]

        # ── Left: annual time series ───────────────────────────────────────────
        ax_ts = fig.add_subplot(gs[row, 0])
        sim_lo = d["ann_sim"].min(axis=0)
        sim_hi = d["ann_sim"].max(axis=0)
        sim_med = np.median(d["ann_sim"], axis=0)

        ax_ts.fill_between(
            years_sim, sim_lo, sim_hi, alpha=0.18, color=color, label="Sim range"
        )
        ax_ts.plot(years_sim, sim_med, color=color, lw=1.2, alpha=0.7,
                   label="Sim median")
        ax_ts.plot(years_obs, d["ann_obs"], "k-o", ms=4, lw=1.5,
                   label="Observed")
        ax_ts.set_xlabel("Year of record")
        ax_ts.set_ylabel(LABELS[key], fontsize=8)
        ax_ts.set_title(f"{key} — Annual time series", fontsize=9)
        if row == 0:
            ax_ts.legend(fontsize=7, loc="upper right")

        # ── Middle: σ²_sim distribution vs σ²_obs ─────────────────────────────
        ax_var = fig.add_subplot(gs[row, 1])
        ax_var.hist(
            d["var_sim_each"], bins=12, color=color, alpha=0.65,
            edgecolor="white", label="σ²_sim members"
        )
        ax_var.axvline(
            d["var_obs"], color="black", lw=2.0, ls="--",
            label=f"σ²_obs = {d['var_obs']:.2f}"
        )
        ax_var.axvline(
            d["mean_var_sim"], color=color, lw=1.5, ls=":",
            label=f"mean(σ²_sim) = {d['mean_var_sim']:.2f}"
        )
        ax_var.set_xlabel("Inter-annual variance")
        ax_var.set_ylabel("Count")
        ax_var.set_title(f"{key} — Variance distribution", fontsize=9)
        ax_var.legend(fontsize=7)

        # ── Right: PPV bar ─────────────────────────────────────────────────────
        ax_ppv = fig.add_subplot(gs[row, 2])
        ppv_val = d["ppv"]
        bar_color = color if ppv_val >= 0 else "#999999"
        ax_ppv.bar([key], [ppv_val], color=bar_color, edgecolor="black", width=0.4)
        ax_ppv.axhline(0, color="black", lw=0.8, ls="--")
        ax_ppv.axhline(1, color="black", lw=0.5, ls=":", alpha=0.4)
        ax_ppv.set_ylim(-0.5, 1.2)
        ax_ppv.set_ylabel("PPV")
        ax_ppv.set_title(f"{key} — PPV = {ppv_val:.3f}", fontsize=9)
        ax_ppv.text(
            0, ppv_val + (0.04 if ppv_val >= 0 else -0.08),
            f"{ppv_val:.3f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold",
        )

    plt.show()


def plot_ppv_normalized(ppv_norm_results: dict) -> None:
    """Plot normalized PPV per Anderson et al. (2016) Eqs. 7–8.

    Parameters
    ----------
    ppv_norm_results : dict
        Results dict from compute_normalized_ppv() with keys "OCC", "SII", "TOT".
    """
    COLORS = {"OCC": "#2166ac", "SII": "#d6604d", "TOT": "#4dac26"}

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Normalized Potential Predictable Variance — Anderson et al. (2016) Eqs. 7–8",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    for row, key in enumerate(("OCC", "SII", "TOT")):
        d = ppv_norm_results[key]
        color = COLORS[key]

        # ── Left: null PPV distribution vs observed PPV ────────────────────────
        ax_null = fig.add_subplot(gs[row, 0])
        ppv_sim = d["ppv_sim_each"]
        thresh = d["threshold_p10"]
        ppv_obs = d["ppv"]

        ax_null.hist(
            ppv_sim[~np.isnan(ppv_sim)], bins=14,
            color=color, alpha=0.55, edgecolor="white", label="PPV$_{sim}$(n)",
        )
        ax_null.axvline(
            thresh, color="darkorange", lw=1.8, ls="--",
            label=f"90th pct = {thresh:.3f}",
        )
        ax_null.axvline(
            ppv_obs, color="black", lw=2.0, ls="-",
            label=f"PPV$_{{obs}}$ = {ppv_obs:.3f}",
        )
        ax_null.set_xlabel("PPV$_{sim}$(n)")
        ax_null.set_ylabel("Count")
        ax_null.set_title(f"{key} — null distribution (Eq. 7)", fontsize=9)
        ax_null.legend(fontsize=7)

        # ── Middle: normalized PPV bar with significance line ─────────────────
        ax_norm = fig.add_subplot(gs[row, 1])
        ppv_n = d["ppv_normalized"]
        bar_color = color if ppv_n > 1 else "#aaaaaa"
        ax_norm.bar([key], [ppv_n], color=bar_color, edgecolor="black", width=0.45,
                    label="Norm. PPV")
        ax_norm.axhline(1.0, color="darkorange", lw=1.8, ls="--",
                        label="p=0.10 threshold")
        ax_norm.axhline(0.0, color="black", lw=0.7, ls=":")
        ymax = max(ppv_n * 1.25, 1.5)
        ax_norm.set_ylim(-0.1, ymax)
        ax_norm.set_ylabel(r"Normalized PPV  ($\widetilde{\mathrm{PPV}}$)")
        sig_str = f"✓ sig. (p<0.10)" if ppv_n > 1 else "✗ not sig."
        ax_norm.set_title(
            rf"{key} — $\widetilde{{\mathrm{{PPV}}}}$ = {ppv_n:.2f}  {sig_str}",
            fontsize=9
        )
        ax_norm.text(
            0, ppv_n + 0.04 * ymax, f"{ppv_n:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
        ax_norm.legend(fontsize=7)

        # ── Right: CDF of null vs observed PPV ────────────────────────────────
        ax_cdf = fig.add_subplot(gs[row, 2])
        sorted_null = np.sort(ppv_sim[~np.isnan(ppv_sim)])
        cdf = np.arange(1, len(sorted_null) + 1) / len(sorted_null)
        ax_cdf.step(sorted_null, cdf, color=color, lw=1.5, label="Null CDF")
        ax_cdf.axvline(
            ppv_obs, color="black", lw=2.0, ls="-",
            label=f"PPV$_{{obs}}$={ppv_obs:.3f}"
        )
        ax_cdf.axhline(0.90, color="darkorange", lw=1.2, ls="--",
                       label="90th pct")
        # Mark empirical p-value: fraction of null exceeding ppv_obs
        p_val = float((sorted_null >= ppv_obs).mean())
        ax_cdf.set_xlabel("PPV")
        ax_cdf.set_ylabel("Cumulative probability")
        ax_cdf.set_title(
            f"{key} — CDF  (empirical p = {p_val:.3f})", fontsize=9
        )
        ax_cdf.set_xlim(sorted_null.min() - 0.02, min(sorted_null.max() + 0.05, 1.02))
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.legend(fontsize=7)

    plt.show()


def plot_spectral_ppv(
    ppv_results: dict,
    ppv_spectral: dict,
    ppv_spectral_meta: dict,
) -> None:
    """Plot frequency decomposition of PPV (Anderson 2016 Eqs. 2–6).

    Parameters
    ----------
    ppv_results : dict
        Results from compute_ppv() (needed for record length).
    ppv_spectral : dict
        Band results from spectral_ppv_decompose().
    ppv_spectral_meta : dict
        Metadata from spectral_ppv_decompose().
    """
    from spectral import _FREQ_BANDS, BAND_ORDER

    _SPEC_COLORS = {"OCC": "black", "SII": "#d6604d", "TOT": "#2166ac"}

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle(
        "Frequency Decomposition of Potential Predictability\n"
        "Anderson et al. (2016) Eqs. 2–6",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    # ── Left panel: normalised PPV spectrum (analog of Fig. 4 a–f) ───────────
    ax_spec = axes[0]

    for key, color in _SPEC_COLORS.items():
        meta = ppv_spectral_meta[key]
        freqs = meta["freqs"]          # cycles yr⁻¹, DC excluded
        pp_freq = meta["pp_per_freq"]  # (psd_obs[k] - mean_psd_sim[k]) / σ²_obs

        # Normalise each frequency by the threshold of its enclosing band.
        # Summing within a band then recovers ppv_normalized for that band.
        pp_norm_freq = np.full_like(pp_freq, np.nan)
        for label, (flo, fhi) in _FREQ_BANDS.items():
            if label == ">40yr":
                mask = (freqs > 0.0) & (freqs <= fhi)
            else:
                mask = (freqs > flo) & (freqs <= fhi)
            thresh = ppv_spectral[key][label]["threshold_p10"]
            if thresh > 0.0:
                pp_norm_freq[mask] = pp_freq[mask] / thresh

        # Restrict to periods ≤ N/2 years (half the record length)
        n_years = len(ppv_results[key]["ann_obs"])
        periods = 1.0 / freqs
        plot_mask = (periods >= 2.0) & (periods <= n_years / 2)
        ax_spec.plot(
            periods[plot_mask],
            pp_norm_freq[plot_mask],
            color=color,
            lw=1.5,
            label=key,
        )

    # Significance line at 1.0
    ax_spec.axhline(1.0, color="darkorange", lw=1.3, ls="--", label="p=0.10")
    ax_spec.axhline(0.0, color="black", lw=0.5, ls=":", alpha=0.4)

    # Alternating band shading
    for (p_lo, p_hi), bg in zip(
        [(2, 7), (7, 20), (20, 40), (40, 200)],
        ["#f0f0f0", "#dcdcdc", "#f0f0f0", "#dcdcdc"],
    ):
        ax_spec.axvspan(p_lo, p_hi, alpha=0.4, color=bg, zorder=0)

    ax_spec.set_xscale("log")
    ax_spec.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_spec.set_xticks([2, 5, 10, 20, 40, 80])
    ax_spec.set_xlim(2, n_years / 2)
    ax_spec.set_xlabel("Period (yr)")
    ax_spec.set_ylabel(r"Normalized PPV  ($\widetilde{\mathrm{PPV}}$)")
    ax_spec.set_title("PPV spectrum — per frequency", fontsize=10)
    ax_spec.legend(fontsize=9, loc="upper left")

    # Band period labels along the top x-axis
    ymax = ax_spec.get_ylim()[1]
    for label_txt, mid in [
        ("2–7 yr", 3.7),
        ("7–20 yr", 11.8),
        ("20–40 yr", 28.0),
        (">40 yr", 65.0),
    ]:
        ax_spec.text(
            mid, ymax * 0.97, label_txt,
            ha="center", va="top", fontsize=7, color="0.40",
        )

    # ── Right panel: bar chart by band (analog of Fig. 4 panel f) ────────────
    ax_bar = axes[1]

    x = np.arange(len(BAND_ORDER))
    width = 0.25
    offsets = [-width, 0.0, width]

    for offset, key in zip(offsets, ("OCC", "SII", "TOT")):
        vals = np.array(
            [ppv_spectral[key][b]["ppv_normalized"] for b in BAND_ORDER],
            dtype=float,
        )
        bar_colors = [
            _SPEC_COLORS[key] if (np.isfinite(v) and v > 1.0) else "#cccccc"
            for v in vals
        ]
        ax_bar.bar(
            x + offset,
            vals,
            width=width,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            label=key,
        )

    ax_bar.axhline(1.0, color="darkorange", lw=1.3, ls="--", label="p=0.10")
    ax_bar.axhline(0.0, color="black", lw=0.5, ls=":")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(BAND_ORDER, rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylabel(r"Normalized PPV  ($\widetilde{\mathrm{PPV}}$)")
    ax_bar.set_title("Band-integrated PPV", fontsize=10)
    ax_bar.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.6)
            for c in _SPEC_COLORS.values()
        ],
        labels=list(_SPEC_COLORS.keys()),
        fontsize=8,
        loc="upper right",
    )

    fig.tight_layout()
    plt.show()
