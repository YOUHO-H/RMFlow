import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertModel
import math
from models.unet import UNetModel, MFUNetModel
from models.dit import MFDiT
# ------------------------------------------------
# Sinusoidal Time Embedding
# ------------------------------------------------
def get_timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings (B, dim)
    """
    device = timesteps.device
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1)
    sinusoid = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
    return emb  # (B, dim)

# ------------------------------------------------
# Time Embedding Block
# ------------------------------------------------
class TimeEmbeddingBlock(nn.Module):
    def __init__(self, time_emb_dim, out_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_dim)
        )

    def forward(self, t):
        t_embed = get_timestep_embedding(t, self.time_mlp[0].in_features)
        return self.time_mlp(t_embed)  # (B, out_dim)

class SpatialTimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int = 128, out_channels: int = 1, height: int = 32, width: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * height * width)
        )
        self.out_channels = out_channels
        self.height = height
        self.width = width

    def forward(self, t):
        t_embed = get_timestep_embedding(t, self.time_mlp[0].in_features)  # (B, time_emb_dim)
        x = self.time_mlp(t_embed)  # (B, C*H*W)
        return x.view(-1, self.out_channels, self.height, self.width)  # (B, C, H, W)



################################################################################
# Positional & temporal embeddings
################################################################################
class SinPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (B,)
        sinusoid_inp = torch.einsum("b,d->b d", t, self.inv_freq)
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return emb

################################################################################
# Transformer backbone
################################################################################
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim * 2)  # SwiGLU
        self.lin2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(F.silu(x1) * x2)
    
def timestep_embedding(t, dim, max_period: float = 10_000.0):
    """
    Convert scalar t ∈ [0, 1] (or [0, T]) to a dim‑D vector using
    sinusoidal Fourier features.  Returned shape: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(half, device=t.device, dtype=t.dtype) / half
    )                               # (half,)
    args = t[:, None] * freqs       # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.time_embedding = SpatialTimeEmbedding()
        self.mlp = MLP(dim, int(dim * 4 / 3))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.time_project = nn.Sequential(
            nn.Linear(16, dim//2),
            nn.SiLU(),
            nn.Linear(dim//2, dim)
        )
        

    def forward(self, x, t):

        e_t = timestep_embedding(t, 16)
        
        tau = self.time_project(e_t).unsqueeze(1)  # (B, 1, dim)
        x = self.norm1(x + self.attn(x, x, x, need_weights=False)[0])
        x = self.norm2(x + self.cross_attn(x, tau, tau, need_weights=False)[0])
        x = x + self.mlp(self.norm3(x))

        return x

# ---------- util: DDPM‑style Fourier embedding ----------


class FlowTokBackbone(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.input_proj = nn.Linear(4, dim)  # project VAE channel dim 4 → dim
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, dim) / dim ** 0.5)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.activation = nn.SiLU()
        self.to_v = OutputBlock(dim=dim, out_channels=4, resolution=32)


    def forward(self, x, t):
        # x: (B, 4, 32, 32)
        # t_emb = self.time_embedding(t)                  # (B, 4, 32, 32)
        # x = torch.cat([x, t_emb], dim=1)  # (B, 6, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)       # → (B, 1024, 6)
        x = self.activation(self.input_proj(x))        # → (B, 1024, dim)
        x = x + self.pos[:, :x.size(1)]                # add positional encoding
        
        for blk in self.blocks:
            x = blk(x, t)

        x = self.to_v(x)                                # (B, 4, 32, 32)
        
        return x  # shape: (B, 1024, dim)

class OutputBlock(nn.Module):
    def __init__(self, dim: int, out_channels: int = 4, resolution: int = 32):
        super().__init__()
        self.linear = nn.Linear(dim, out_channels)
        self.resolution = resolution
        self.out_channels = out_channels
        self.activation = nn.Tanh()  # use tanh for VAE-style bounded outputs

    def forward(self, x):
        """
        x: (B, 1024, dim)
        returns: (B, 4, 32, 32)
        """
        B, L, D = x.shape
        assert L == self.resolution * self.resolution, f"Expected sequence length {self.resolution**2}, got {L}"
        x = self.linear(x)                    # (B, 1024, 4)
        # x = self.activation(x)               # apply activation
        x = x.permute(0, 2, 1)               # (B, 4, 1024)
        x = x.view(B, self.out_channels, self.resolution, self.resolution)
        return x  # (B, 4, 32, 32)

### text encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class FrozenMap768to4x32x32(nn.Module):
    """Map (B,768) → (B,4,32,32) with a *fixed* orthogonal projection."""
    def __init__(self, in_dim=768, out_ch=4, h=32, w=32):
        super().__init__()
        # Orthogonal weight: preserves dot‑products on average
        W = torch.empty(in_dim, out_ch*h*w)
        nn.init.orthogonal_(W)                # shape (768, 4096)
        W *= math.sqrt(out_ch*h*w / in_dim) * 2
        self.register_buffer('W', W)          # not a parameter → no grad

    def forward(self, x):                     # x: (B,768)
        y = x @ self.W                        # (B,4096)
        return y  
    
class LearnableMap768to4x32x32(nn.Module):
    """Learnable map (B,768) → (B,4,32,32) with a linear projection."""
    def __init__(self, in_dim=768, 
                 out_ch=4, h=32, w=32,
                 g_init=2.665, g_min=2.5, g_max=2.8):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_ch * h * w, bias=False)  # no bias for orthogonality
        self.proj = torch.nn.utils.weight_norm(self.proj)     
        # initial scale
        with torch.no_grad():
            self.proj.weight_g.fill_(g_init)
        self.g_min, self.g_max = g_min, g_max
    
    def clamp_gain(self):
        with torch.no_grad():
            self.proj.weight_g.clamp_(self.g_min, self.g_max)

    def forward(self, x):                     # x: (B,768)
        self.clamp_gain() 
        y = self.proj(x)                      # (B,4096)
        return y

################################################################################
# Full model
################################################################################


class FlowTokLite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # self.txt_proj = LearnableMap768to4x32x32(in_dim=768, out_ch=4, h=32, w=32)
        if cfg.frozen_text_proj:
            self.txt_proj = FrozenMap768to4x32x32(in_dim=768, out_ch=4, h=32, w=32)
        else:
            self.txt_proj = LearnableMap768to4x32x32(in_dim=768, out_ch=4, h=32, w=32)
        
        # image tokenizer
        #  self.img_tok = PatchTokenizer(cfg.img_size, 16, cfg.d_model)

        if cfg.model == 'flowtok_lite':
            self.backbone = FlowTokBackbone(dim = cfg.d_model,layers = cfg.n_layers, heads = cfg.n_heads)
        elif cfg.model == 'unet':
            self.backbone = UNetModel(in_channels=4, model_channels=cfg.d_model, 
                                      num_heads = cfg.n_heads,
                                      out_channels=4)
        elif cfg.model == 'dit':
            self.backbone = MFDiT(input_size=32, patch_size=2, in_channels=4, 
                                  dim=cfg.d_model, depth=cfg.n_layers, 
                                  num_heads=cfg.n_heads)
        elif cfg.model == 'mfunet':
            self.backbone = MFUNetModel(in_channels=4, model_channels=cfg.d_model, 
                                      num_heads = cfg.n_heads,
                                      out_channels=4)
    
    def text_to_latent(self, text_emb: torch.Tensor) -> torch.Tensor:
        x = self.txt_proj(text_emb)  # (B, 4 * 32 * 32)
        x = x.view(x.size(0), 4, 32, 32)
        return x  # (B, 4, 32, 32)

    # -------- velocity predictor -------------------------------------------
    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor):  # z_t: (B,N,D) t:(B,)
        if self.cfg.model == 'flowtok_lite' or self.cfg.model == 'unet':
            return self.backbone(z_t, t)
        elif self.cfg.model == 'dit' or self.cfg.model == 'mfunet':
            return self.backbone(z_t, r, t)

