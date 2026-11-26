"""Early stopping on a VALID metric: QEL on the VALID energy, SDDR on the VALID relative regret. The DFL baselines are
stopped by the OpenPTO loop, which applies the same rule."""

PATIENCE = 50


def early_stopping(history):
    """(best epoch, stop epoch or None) of a VALID history; index 0 is epoch 1 and lower is better.

    A later epoch replaces the best one only when it is strictly lower. The search stops at the first epoch that does
    not improve and lies more than PATIENCE epochs after the best one; later epochs are ignored."""
    best = None
    for epoch, value in enumerate(history, 1):
        if best is None or value < history[best - 1]:
            best = epoch
        elif epoch - best > PATIENCE:
            return best, epoch
    return best, None
