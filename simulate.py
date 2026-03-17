"""Monte Carlo simulation of ZIG model forward integrations."""

from typing import Optional, Tuple

import numpy as np
import torch

from features import build_features
from model import ZIGammaMLP


def _sample_gamma_mixture(
    pi: torch.Tensor,    # (N, K)
    alpha: torch.Tensor, # (N, K)
    beta: torch.Tensor,  # (N, K)
) -> torch.Tensor:
    """Draw one sample per row from the gamma mixture. Returns shape (N,).

    1. Pick component k ~ Categorical(pi)
    2. Draw x ~ Gamma(alpha_k, rate=beta_k)
    """
    k = torch.multinomial(pi, num_samples=1).squeeze(-1)   # (N,)
    idx = torch.arange(len(pi), device=pi.device)
    return torch.distributions.Gamma(
        concentration=alpha[idx, k], rate=beta[idx, k]
    ).sample()


@torch.no_grad()
def monte_carlo_simulate(
    model: ZIGammaMLP,
    doy_sequence: np.ndarray,                      # (T,) day-of-year
    n_lags: int = 5,
    n_int_lags: int = 0,                           # number of intensity lags
    extra_sequence: Optional[np.ndarray] = None,   # (T, E)
    n_simulations: int = 1000,
    wet_threshold: float = 1.0,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-integrate the ZIG model for ``n_simulations`` realisations.

    Equivalent to the 1000-member MC ensemble in Anderson et al. (2016) Sec. 2c,
    but with a single model call per time step instead of two.

    Returns
    -------
    occ_sim : (n_simulations, T)  binary occurrence
    int_sim : (n_simulations, T)  intensity in mm/event (0 on dry days)
    """
    model.eval().to(device)
    T, S = len(doy_sequence), n_simulations
    occ_sim = np.zeros((S, T), dtype=np.float32)
    int_sim = np.zeros((S, T), dtype=np.float32)

    # Initialise lag buffers to all-zero (dry / zero-intensity start)
    lag_buffer = np.zeros((S, n_lags), dtype=np.float32)
    int_lag_buffer = (
        np.zeros((S, n_int_lags), dtype=np.float32) if n_int_lags > 0 else None
    )

    for t in range(T):
        doy_t = np.full(S, doy_sequence[t], dtype=np.int32)
        extra_t = (
            np.tile(extra_sequence[t], (S, 1)) if extra_sequence is not None else None
        )
        x_t = build_features(doy_t, lag_buffer, int_lag_buffer, extra_t).to(
            device
        )  # (S, F)

        # Single forward pass → both occurrence and intensity parameters
        logit_p, pi, alpha, beta = model(x_t)

        # Stochastic occurrence draw
        wet_draw = torch.bernoulli(torch.sigmoid(logit_p).squeeze(-1)).cpu().numpy()
        occ_sim[:, t] = wet_draw

        # Intensity only for simulated wet days
        wet_mask = wet_draw.astype(bool)
        if wet_mask.any():
            samples = _sample_gamma_mixture(
                pi[wet_mask], alpha[wet_mask], beta[wet_mask]
            ).cpu().numpy()
            int_sim[wet_mask, t] = np.maximum(samples, wet_threshold)

        # Roll occurrence lag buffer: shift left, append today
        lag_buffer = np.roll(lag_buffer, shift=-1, axis=1)
        lag_buffer[:, -1] = wet_draw  # most-recent lag in last column

        # Roll intensity lag buffer with simulated intensity
        # (log1p applied in build_features)
        if int_lag_buffer is not None:
            int_lag_buffer = np.roll(int_lag_buffer, shift=-1, axis=1)
            int_lag_buffer[:, -1] = int_sim[:, t]  # raw mm; log1p in build_features

    return occ_sim, int_sim
