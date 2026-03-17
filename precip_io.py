"""Data I/O utilities for precipitation files."""

import numpy as np


def load_precip_txt(
    path: str,
    wet_threshold: float = 1.0,
    n_lags: int = 5,
) -> dict:
    """Load a station precipitation text file into the standard array dict.

    Expected format
    ---------------
    365 rows (day-of-year) × N_years columns, whitespace-delimited, values in
    mm.  Each column is one calendar year; leap days must have been removed
    upstream so every year has exactly 365 rows.

    The function returns a dict with the same keys as the .npz files produced
    by ``make_enso_ar1_data.py``, so it can be used interchangeably in §8 —
    just point DATA_FILE at the .txt path and the rest of the cell is unchanged.

    Parameters
    ----------
    path          : path to the text file
    wet_threshold : days with intensity below this (mm) are set to dry (0)
    n_lags        : number of lagged occurrence features to build

    Returns
    -------
    dict with keys: doy, occ_lags, occ_obs, int_obs
    """
    mat = np.loadtxt(path, dtype=np.float32)   # (365, N_years)

    if mat.ndim == 1:
        raise ValueError(
            "File parsed as a single row — check that values are "
            "whitespace-separated and the file has 365 rows."
        )
    if mat.shape[0] != 365:
        raise ValueError(
            f"Expected 365 rows (one per DOY), got {mat.shape[0]}.\n"
            "Either transpose the file so rows = DOY and columns = years, "
            "or remove leap days so every year has exactly 365 days."
        )

    n_years = mat.shape[1]
    n_days = 365 * n_years

    # Flatten column-major → chronological daily time series
    # mat.T shape: (N_years, 365)  →  flatten: yr0_d1, yr0_d2, ...,
    # yr0_d365, yr1_d1, ...
    int_obs = mat.T.flatten().astype(np.float32)
    int_obs[int_obs < wet_threshold] = 0.0
    occ_obs = (int_obs > 0).astype(np.float32)

    doy = np.tile(np.arange(1, 366, dtype=np.int32), n_years)   # (N,)

    # Rolling-window lagged occurrence (same convention as .npz files)
    # col 0 = oldest lag (t − n_lags), col −1 = most recent (t − 1)
    occ_lags = np.zeros((n_days, n_lags), dtype=np.float32)
    for j in range(n_lags):
        lag = n_lags - j
        occ_lags[lag:, j] = occ_obs[:n_days - lag]

    return {"doy": doy, "occ_lags": occ_lags, "occ_obs": occ_obs, "int_obs": int_obs}
