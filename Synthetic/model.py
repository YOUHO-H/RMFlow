import math
import torch
import torch.nn as nn
from torch import Tensor

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

class FourierFeatures(nn.Module):

    def __init__(self, in_dim: int, n_freq: int = 8, logspace: bool = True, base: float = 2.0):
        super().__init__()
        self.in_dim = in_dim
        self.n_freq = n_freq

        if logspace:
            freqs = (base ** torch.arange(n_freq, dtype=torch.float32))
        else:
            freqs = torch.arange(1, n_freq + 1, dtype=torch.float32)

        self.register_buffer("freqs", freqs)

        self.out_dim = in_dim + 2 * (in_dim * n_freq)

    def forward(self, x: Tensor) -> Tensor:
        angles = 2 * math.pi * x[..., None] * self.freqs[None, None, :]   # (B, in_dim, n_freq)
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, in_dim, 2*n_freq)
        sincos = sincos.reshape(x.shape[0], -1)  # (B, in_dim * 2*n_freq)
        return torch.cat([x, sincos], dim=1)     # (B, out_dim)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=1, keepdim=True)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (w.detach() * loss).mean()

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
    def forward(self, x: Tensor) -> Tensor: 
        # return torch.sigmoid(x) * x
        return self.act(x)


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-2)
        self.act1  = Swish()
        self.fc1   = nn.Linear(dim, 2 * dim)

        self.norm2 = nn.LayerNorm(2 * dim, eps=1e-2)
        self.act2  = Swish()
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2   = nn.Linear(2 * dim, dim)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(self.act1(self.norm1(x)))
        h = self.fc2(self.drop(self.act2(self.norm2(h))))
        # h = self.fc1(self.act1(x))
        # h = self.fc2(self.drop(self.act2(h)))
        return x + h 
        # return h 

class MLP(nn.Module):
    def __init__(self, input_dim: int = 32, time_dim: int = 1, hidden_dim: int = 256, num_blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim  = time_dim
        self.hidden_dim = hidden_dim
        self.txt_emb_nn = nn.Sequential(
            nn.Linear(2, 16),
            Swish(),
            nn.Linear(16, 2),
        )

        in_dim = input_dim + time_dim * 1

        self.stem = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-2),
            Swish(),
        )
        self.x_embed = FourierFeatures(in_dim=2, n_freq=8, logspace=False)

        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])

        self.head = nn.Linear(hidden_dim, input_dim)

        nn.init.kaiming_uniform_(self.stem[0].weight, a=1.0)
        nn.init.zeros_(self.stem[0].bias)
        nn.init.zeros_(self.head.weight)  
        nn.init.zeros_(self.head.bias)

    def txt_embed(self, txt_tok, x_0):
        h = txt_tok + 1e-2 * x_0
        h = self.txt_emb_nn(h)
        return h

    def forward(self, x: Tensor, t: Tensor, r: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, 1).float()
        r = r.reshape(-1, 1).float()
        t = t + r

        emb_t = timestep_embedding(t, 30).squeeze(1)
        emb_r = timestep_embedding(r, 30).squeeze(1)

        emb_t = emb_t + emb_r

        x_feat = self.x_embed(x)

        h = torch.cat([x_feat, emb_t], dim=1)
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)

        return out.reshape(*sz)