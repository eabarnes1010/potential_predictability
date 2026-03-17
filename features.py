"""Feature engineering for ZIG model."""

import math
from typing import Optional

import numpy as np
import torch


def doy_encoding(doy: torch.Tensor) -> torch.Tensor:
    """Integer day-of-year [1..365] → (sin, cos) pair, shape (N, 2)."""
    angle = 2.0 * math.pi * (doy.float() - 1.0) / 365.0
    return torch.stack([angle.sin(), angle.cos()], dim=-1)


def compute_in_features(
    n_lags: int, n_int_lags: int = 0, n_extra: int = 0
) -> int:
    """Return input feature dimension: 2 (doy) + n_lags [+ n_int_lags] [+ n_extra]."""
    return 2 + n_lags + n_int_lags + n_extra


def build_features(
    doy: np.ndarray,                        # (N,)   int, day-of-year 1..365
    occ_lags: np.ndarray,                   # (N, L) float32, lagged occurrence
    int_lags: Optional[np.ndarray] = None,  # (N, L) float32, lagged intensity mm
    extra: Optional[np.ndarray] = None,     # (N, E) optional covariates
) -> torch.Tensor:
    """Assemble float32 input tensor, shape (N, 2 + L [+ L_int] [+ E]).

    Lagged intensities are log1p-transformed before concatenation so that
    dry-day zeros map cleanly to 0 and the right-skewed wet-day values are
    compressed to a model-friendly range.
    """
    doy_enc = doy_encoding(torch.from_numpy(np.asarray(doy, dtype=np.int32)))
    parts = [doy_enc, torch.from_numpy(np.asarray(occ_lags, dtype=np.float32))]
    if int_lags is not None:
        log1p_int = np.log1p(np.asarray(int_lags, dtype=np.float32))
        parts.append(torch.from_numpy(log1p_int))
    if extra is not None:
        parts.append(torch.from_numpy(np.asarray(extra, dtype=np.float32)))
    return torch.cat(parts, dim=-1)
