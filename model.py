"""ZIG (Zero-Inflated Gamma) model architecture and loss functions."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZIGammaMLP(nn.Module):
    """Zero-Inflated Gamma (mixture) MLP.

    A single network jointly estimates:
        P(wet | x)  via logit_p head
        p(y | wet, x) via a K-component gamma mixture (alpha, beta, pi heads)

    Parameters
    ----------
    in_features   : input dimension from build_features()
    hidden_sizes  : MLP hidden layer widths
    n_components  : K=1 → simple ZIG-Gamma; K=2 → gamma-gamma (paper form)
    dropout       : applied after each hidden layer
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        n_components: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.K = n_components

        # Shared feature extractor
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.head_logit_p = nn.Linear(prev, 1)            # → logit P(wet)
        self.head_alpha = nn.Linear(prev, n_components)   # → gamma shape
        self.head_beta = nn.Linear(prev, n_components)    # → gamma rate
        self.head_pi = nn.Linear(prev, n_components)      # → mixing weights

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logit_p : (N, 1)   raw logit for P(wet)
        pi      : (N, K)   mixture weights, sum-to-1
        alpha   : (N, K)   gamma shape > 0
        beta    : (N, K)   gamma rate  > 0
        """
        h = self.trunk(x)
        logit_p = self.head_logit_p(h)                          # (N, 1)
        pi = F.softmax(self.head_pi(h), dim=-1)                 # (N, K)
        alpha = F.softplus(self.head_alpha(h)) + 1e-3           # (N, K)
        beta = F.softplus(self.head_beta(h)) + 1e-3             # (N, K)
        return logit_p, pi, alpha, beta

    @torch.no_grad()
    def p_wet(self, x: torch.Tensor) -> torch.Tensor:
        """P(wet | x), shape (N,). Convenience method for calibration."""
        logit_p, *_ = self.forward(x)
        return torch.sigmoid(logit_p).squeeze(-1)


def _gamma_log_prob(
    y: torch.Tensor,      # (M,)    positive values only
    alpha: torch.Tensor,  # (M, K)
    beta: torch.Tensor,   # (M, K)
) -> torch.Tensor:
    """Log-density under each gamma component (rate parameterisation).

    log p(y | α, β) = α·log β − log Γ(α) + (α−1)·log y − β·y

    Returns (M, K).
    """
    y_ = y.unsqueeze(-1)  # (M, 1) — broadcasts over K
    return (
        alpha * beta.log()
        - torch.lgamma(alpha)
        + (alpha - 1.0) * y_.clamp(min=1e-6).log()
        - beta * y_
    )


def zig_nll(
    logit_p: torch.Tensor,  # (N, 1)
    pi: torch.Tensor,       # (N, K)
    alpha: torch.Tensor,    # (N, K)
    beta: torch.Tensor,     # (N, K)
    y: torch.Tensor,        # (N,)   full record (0 on dry days)
    wet_threshold: float = 1.0,
) -> torch.Tensor:
    """Joint zero-inflated gamma NLL — single loss for the whole record.

    Dry days  (y < threshold):  -log(1 − sigmoid(logit_p))
    Wet days  (y ≥ threshold):  -log(sigmoid(logit_p))
                                -log Σ_k π_k · Gamma(y; α_k, β_k)

    Uses F.logsigmoid for numerical stability on both branches.
    """
    is_wet = y >= wet_threshold          # (N,) bool mask

    logit_p = logit_p.squeeze(-1)        # (N,)
    log_p = F.logsigmoid(logit_p)        # log P(wet)
    log_1mp = F.logsigmoid(-logit_p)     # log P(dry) = log(1 − sigmoid)

    # Dry contribution
    loss_dry = -log_1mp[~is_wet].sum()

    # Wet contribution: log P(wet) + log mixture density
    if is_wet.any():
        lp = _gamma_log_prob(y[is_wet], alpha[is_wet], beta[is_wet])  # (M, K)
        log_mix = torch.logsumexp(pi[is_wet].log() + lp, dim=-1)      # (M,)
        loss_wet = -(log_p[is_wet] + log_mix).sum()
    else:
        loss_wet = torch.tensor(0.0, device=y.device)

    return (loss_dry + loss_wet) / len(y)
