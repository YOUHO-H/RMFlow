import torch
from torch import nn

class EMA:
    """
    Exponential-Moving-Average wrapper for model parameters.
    Call   ema.update()            after each optim.step()
    Call   ema.apply_shadow()       before eval / sampling
    Call   ema.restore()            afterwards
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model                  # <-- keep a handle!
        self.decay  = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup = {}

    # ------------------------------------------------------------
    @torch.no_grad()
    def update(self):
        """Update shadow = decay·shadow + (1-decay)·θ_t."""
        for name, shadow_p in self.shadow.items():
            model_p = self._target_param(name)          # already a Parameter
            shadow_p.mul_(self.decay).add_(
                (1.0 - self.decay) * model_p.data)

    # ------------------------------------------------------------
    def _target_param(self, name: str):
        """Return the *current* model parameter (no iterator involved)."""
        return dict(self.model.named_parameters())[name]

    # ------------------------------------------------------------
    def apply_shadow(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])
        self.backup = {}
