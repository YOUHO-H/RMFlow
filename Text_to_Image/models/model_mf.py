import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from PIL import Image
import math
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
    def __init__(self, time_emb_dim: int = 128, out_channels: int = 2, height: int = 32, width: int = 32):
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
# Tokenizers
################################################################################
class PatchTokenizer(nn.Module):
    """Flatten 16×16 RGB patches → d_model tokens and *vice‑versa*."""

    def __init__(self, img_size: int = 256, patch: int = 16, d_model: int = 384):
        super().__init__()
        assert img_size % patch == 0, "Image size must be divisible by patch"
        self.img_size, self.patch = img_size, patch
        self.n_patches = (img_size // patch) ** 2
        self.in_dim = 3 * patch * patch
        self.proj = nn.Linear(self.in_dim, d_model)

    # -------------------- encoding --------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W)
        B, C, H, W = x.shape
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # B C H' W' p p
        B, C, Hn, Wn, _, _ = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5)        # (B,H',W',C,p,p)
        x = x.reshape(B, Hn * Wn, C * p * p)   # (B,N,C·p²)
        return self.proj(x)  # B N d_model

    # -------------------- decoding --------------------
    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reassemble patch tokens -> image tensor in [0,1].
        tokens: (B, N, d_model)
        """
        B, N, _ = tokens.shape
        p = self.patch
        # --- 1) ensure N == h²
        h = int(math.isqrt(N))
        if h * h != N:
            raise ValueError(f"Token count N={N} is not a perfect square; "
                            "cannot reshape into h×w grid.")
        w = h

        # --- 2) linear projection back to pixel space
        # (B,N,d) @ (d, C·p²)  -> (B,N,C·p²)
        patches = (tokens @ self.proj.weight).contiguous()

        # --- 3) reshape into (B, N, C, p, p) then grid
        patches = patches.reshape(B, N, 3, p, p)             # B N C p p
        patches = patches.reshape(B, h, w, 3, p, p)           # B h w C p p
        patches = patches.permute(0, 3, 1, 4, 2, 5)           # B C h p w p
        img = patches.reshape(B, 3, h * p, w * p)             # B 3 H W

        return img.clamp(0, 1)

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


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mlp = MLP(dim, int(dim * 4 / 3))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x, need_weights=False)[0])
        x = x + self.mlp(self.norm2(x))
        return x

class FlowTokBackbone(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.input_proj = nn.Linear(6, dim)  # project VAE channel dim 4 → dim
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, dim) / dim ** 0.5)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.activation = nn.SiLU()

    def forward(self, x):
        # x: (B, 4, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)       # → (B, 1024, 4)
        x = self.activation(self.input_proj(x))        # → (B, 1024, dim)
        x = x + self.pos[:, :x.size(1)]                # add positional encoding
        for blk in self.blocks:
            x = blk(x)
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

################################################################################
# Full model
################################################################################


class FlowTokLite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.text_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # self.txt_emb = nn.Embedding(self.text_tok.vocab_size, cfg.d_model)
        self.txt_emb = nn.Embedding(self.text_tok.vocab_size, 32)
        self.txt_proj = nn.Linear(self.cfg.seq_len * 32, 4 * 32 * 32)
        self.img_tok = PatchTokenizer(cfg.img_size, 16, cfg.d_model)
        # self.img_tok = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
        self.backbone = FlowTokBackbone(dim = cfg.d_model,layers = cfg.n_layers, heads = cfg.n_heads)
        self.time_embedding = SpatialTimeEmbedding()
        self.to_v = OutputBlock(dim=cfg.d_model, out_channels=4, resolution=32)

    # -------- encoders -------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, captions: list[str], device) -> torch.Tensor:
        ids = self.text_tok(captions, padding="max_length", max_length=self.cfg.seq_len,
                            truncation=True, return_tensors="pt").input_ids
        emb_ = self.txt_emb(ids.to(device))
        emb_ = emb_.view(emb_.size(0), -1)
        return self.txt_proj(emb_)
        # return ids.to(self.to_v.weight.device)

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.img_tok(img)

    # -------- velocity predictor -------------------------------------------
    def forward(self, z_t: torch.Tensor, t: torch.Tensor):  # z_t: (B,N,D) t:(B,)
        t_emb = self.time_embedding(t)
        x = torch.cat([z_t, t_emb], dim=1)
        x = self.backbone(x)  # (B, N, D)
        return self.to_v(x)