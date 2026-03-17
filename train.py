"""Training loop for ZIG model."""

from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ZIGammaMLP, zig_nll


def train_zig(
    model: ZIGammaMLP,
    x: torch.Tensor,                          # (N, F)  training days
    y: torch.Tensor,                          # (N,)    float32, 0 on dry days
    x_val: Optional[torch.Tensor] = None,     # (M, F)  validation days
    y_val: Optional[torch.Tensor] = None,     # (M,)    float32, 0 on dry days
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    wet_threshold: float = 1.0,               # mm, consistent with Anderson et al.
    patience: int = 50,                       # early-stop epochs without improvement
    min_delta: float = 1e-4,                  # minimum drop in val NLL to count
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train the ZIG model, optionally with early stopping on a held-out record.

    If ``x_val`` / ``y_val`` are supplied the function evaluates the validation
    NLL after every epoch.  Training halts when the validation loss has not
    improved by more than ``min_delta`` for ``patience`` consecutive epochs, and
    the best-weight checkpoint is restored before returning.

    Returns
    -------
    history : dict with keys ``"train"`` and (if val data provided) ``"val"``,
              each a list of per-epoch mean NLL values.  ``history["best_epoch"]``
              records the epoch at which the best validation NLL was achieved.
    """
    model.to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    use_val = (x_val is not None) and (y_val is not None)
    train_hist: list[float] = []
    val_hist: list[float] = []

    best_val_nll = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_imp = 0

    for epoch in range(n_epochs):
        # ── Training pass ──────────────────────────────────────────────────────
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logit_p, pi, alpha, beta = model(xb)
            loss = zig_nll(logit_p, pi, alpha, beta, yb, wet_threshold)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(xb)
        scheduler.step()
        train_nll = running / len(x)
        train_hist.append(train_nll)

        # ── Validation pass ────────────────────────────────────────────────────
        if use_val:
            model.eval()
            with torch.no_grad():
                xv = x_val.to(device)
                yv = y_val.to(device)
                lp, pi_v, al, be = model(xv)
                val_nll = zig_nll(lp, pi_v, al, be, yv, wet_threshold).item()
            val_hist.append(val_nll)

            # Check for improvement
            if val_nll < best_val_nll - min_delta:
                best_val_nll = val_nll
                best_epoch = epoch + 1
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1

        # ── Progress print ─────────────────────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            val_str = f"  val NLL={val_hist[-1]:.4f}" if use_val else ""
            print(f"  epoch {epoch + 1:4d}/{n_epochs}  train NLL={train_nll:.4f}{val_str}")

        # ── Early stopping ─────────────────────────────────────────────────────
        if use_val and epochs_no_imp >= patience:
            print(
                f"  Early stop at epoch {epoch + 1} "
                f"(best val NLL={best_val_nll:.4f} at epoch {best_epoch})"
            )
            break

    # Restore best checkpoint if we were tracking validation loss
    if use_val and best_state is not None:
        model.load_state_dict(best_state)

    history: dict[str, list[float] | int] = {"train": train_hist}
    if use_val:
        history["val"] = val_hist
        history["best_epoch"] = best_epoch
    return history
