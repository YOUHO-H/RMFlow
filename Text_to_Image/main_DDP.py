import argparse
import os
from pathlib import Path
from typing import Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_from_disk
from diffusers.models import AutoencoderKL
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    AutoTokenizer,
    AutoModel,
)

from models.model import FlowTokLite  # local codebase
from models.EMA import EMA           # local codebase
from functools import partial
from PIL import Image, ImageDraw, ImageFont
import wandb
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
################################################################################
# Helper utilities
################################################################################

def setup_distributed(local_rank: int, port: str | None = None):
    """Initialise NCCL process‑group for multi‑GPU training."""
    if dist.is_initialized():
        return  # already set up (e.g. by torchrun)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if port is not None:
        os.environ.setdefault("MASTER_PORT", port)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

################################################################################
# Flow‑matching helper targets (unchanged)
################################################################################

def make_targets(
    txt_tokens: torch.Tensor,
    img_tokens: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_t = (1 - t)[:, None, None, None] * img_tokens + t[:, None, None, None] * txt_tokens
    v = txt_tokens - img_tokens
    return z_t, v, torch.ones_like(txt_tokens)

def adaptive_l2_loss(error: torch.Tensor, gamma: float = 1.0, c: float = 1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (w.detach() * delta_sq).mean()

################################################################################
# Parameter counting (helper)
################################################################################

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# cast the state_dict tensors to bf16 before loading
def cast_sd_to_bf16(sd):
    return {k: (v.to(torch.bfloat16) if torch.is_tensor(v) else v) for k, v in sd.items()}


################################################################################
# Main training loop (DDP)
################################################################################

def train(args):
    local_rank = args.local_rank
    setup_distributed(local_rank, port=args.dist_port)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(123 + rank)

    class CFG:
        n_layers = 8
        d_model = 256
        n_heads = 4
        img_size = args.img_size
        frozen_text_proj = args.frozen_text_proj
        model = args.model

    cfg = CFG()
    model = FlowTokLite(cfg).to(device)
    
    for p in model.parameters():
        p.requires_grad = False
    model.txt_proj.requires_grad = True
    model.backbone.input_blocks[:4].requires_grad_(True)
    model.backbone.output_blocks[-4:].requires_grad_(True)
    model.backbone.out.requires_grad_(True)
    total, trainable = count_parameters(model)
    print(f"Model -- Total parameters: {total}, Trainable parameters: {trainable}")
    model = torch.compile(model, mode="reduce-overhead", backend="inductor")
    ema = EMA(model, decay=0.9995)  # Track original (unwrapped) model
    
    # Wrap in DistributedDataParallel (find_unused_parameters handles jvp path)
    # ---------------------------------------------------------------------
    # Externals: VAE + text encoder (frozen)
    # ---------------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=torch.bfloat16)

    for p in vae.parameters():                  # freeze everything
        p.requires_grad = False
    vae.decoder.conv_in.requires_grad_(True)
    vae.decoder.up_blocks[0].requires_grad_(True)
    vae.decoder.up_blocks[-1].requires_grad_(True)
    vae.decoder.conv_out.requires_grad_(True)
    vae.quant_conv.requires_grad_(True)
    vae.post_quant_conv.requires_grad_(True)
    total, trainable = count_parameters(vae)

    vae = torch.compile(vae, mode="reduce-overhead", backend="inductor")
    

    if args.rcvr_epochs > 0:
        ckpt_path = './MeanFlow_Text2Image/last.pt'
        if os.path.exists(ckpt_path):
            print(f"✓ Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            vae.load_state_dict(ckpt["vae"])
            ema.shadow = ckpt["ema"]
            start_epoch = ckpt["epoch"]
        else:
            print(f"Checkpoint {ckpt_path} not found, starting from scratch.")
            start_epoch = 0

    elif args.use_pretrained:
        pretrained_path = './MeanFlow_Text2Image/pretrained_mf.pt'
        if os.path.exists(pretrained_path):
            print(f"✓ Loading pretrained model from {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(cast_sd_to_bf16(ckpt["model"]))
            vae.load_state_dict(cast_sd_to_bf16(ckpt["vae"]))
        else:
            print(f"Pretrained model {pretrained_path} not found.")
        start_epoch = 0
    else:
        start_epoch = 0

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Param groups (different LR for VAE head is common)
    base_model = model.module
    base_vae   = vae.module
    optim = torch.optim.AdamW(
        [
            {
                "params": base_model.parameters(), "lr": args.lr
            },
            {
                "params": list(base_vae.decoder.up_blocks[-1].parameters()) +
                            list(base_vae.decoder.conv_out.parameters()) +
                            list(base_vae.quant_conv.parameters()) +
                            list(base_vae.post_quant_conv.parameters()),
                "lr": args.lr * 0.5, 
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    ds = load_from_disk(os.path.join(args.dataset, "train_rgb"))
    img_trans = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size), antialias=True),
        transforms.ToTensor(),
    ])

    def collate(batch):
        imgs, captions = zip(*[(img_trans(b["image"]), b["text"]) for b in batch])
        return torch.stack(imgs), list(captions)

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer = optim,
        num_warmup_steps = len(loader) * 2,
        num_training_steps = len(loader) * args.epochs,
        lr_end = 1e-12,
        power = 1,
    )

    if args.rcvr_epochs > 0:
        optim.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    pre_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
    pre_model = AutoModel.from_pretrained("intfloat/e5-base").to(device).eval()

    jvp_fn = partial(torch.autograd.functional.jvp, create_graph=True)
    scale_ = 0.18215
    noise_scale = args.noise_scale

    if rank == 0:
        wandb.init(
            entity="XXX",
            project="XXX",
            mode=args.wandb,
            name='XXX',
            config=vars(args),
            settings=wandb.Settings(_disable_stats=True),
            reinit=True,
        )
        wandb.save("*.txt")

    # ---------------------------------------------------------------------
    # Training epochs
    # ---------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):

        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_embeddings_std = 0.0
        epoch_tix_tok_std = 0.0
        epoch_derror = 0.0
        epoch_img_tok_err = 0.0 
        start_time = time.time()

        for step, (imgs, captions) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True).to(torch.bfloat16)
            # ---------------------------------------------------------
            # Sample (t, r)
            # ---------------------------------------------------------
            if args.t_sample == 'log':
                normal_samples = torch.randn((imgs.size(0), 2), device=device) * 1.0 - 0.4
                samples = 1 / (1 + torch.exp(-normal_samples))  # sigmoid to map to (0,1)
                # t is max
                t = torch.max(samples[:, 0], samples[:, 1])  # ensure t >= r
                # r is min
                r_ = torch.min(samples[:, 0], samples[:, 1])  # ensure r <= t
            elif args.t_sample == 'uniform_1':
                samples = torch.rand((imgs.size(0), 2), device=device)
                t = torch.max(samples[:, 0], samples[:, 1])  # ensure t >= r
                r_ = torch.min(samples[:, 0], samples[:, 1])
            else:
                raise ValueError(f"Unknown t_sample method: {args.t_sample}")
            
            select = torch.rand(imgs.size(0), device=device) < args.flow_ratio
            r_[select] = t[select]

            # unsupervised FT
            t[select] = torch.ones_like(t[select])

            # ---------------------------------------------------------
            # Pre‑process image / text tokens
            # ---------------------------------------------------------
            with torch.no_grad():
                img_tok = vae.module.encode(imgs).latent_dist.mode() * scale_ 
                img_tok = img_tok + 1e-4 * torch.randn_like(img_tok)
                tokens = pre_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
                output = pre_model(**tokens)
                embeddings = output.last_hidden_state[:, 0]  # [CLS]-like 
            if args.frozen_text_proj:
                with torch.no_grad():
                    txt_tok = model.module.text_to_latent(embeddings)
                    reg = 0.0
            else:
                if epoch < 3000:
                    txt_tok = model.module.text_to_latent(embeddings)
                    reg = txt_tok.norm(p=2, dim=(1, 2, 3)).mean() * args.txt_reg
                else:
                    with torch.no_grad():
                        txt_tok = model.module.text_to_latent(embeddings)
                        reg = 0.0
            txt_tok = txt_tok + noise_scale * torch.randn_like(txt_tok)

            # ---------------------------------------------------------
            # Loss computation (unchanged)
            # ---------------------------------------------------------
            def u_fn(x, r_, t):
                return model(x, t - r_, t)

            z_t, v, _ = make_targets(txt_tok, img_tok, t)
            loss = 0.0
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v_pred, dvdt = jvp_fn(u_fn, 
                                    (z_t, r_, t), 
                                    (v.detach(), 
                                    torch.zeros_like(r_), 
                                    torch.ones_like(t)))
        
                dvdt_detach = dvdt.detach()
                v_trgt = v - (t - r_)[:, None, None, None] * dvdt_detach
                error1 = v_pred - v_trgt
                loss += adaptive_l2_loss(error1, gamma=args.gamma, c=1e-3)

                if args.nll > 0.0:
                    img_tok_recon = txt_tok - u_fn(txt_tok, 
                                                   torch.zeros_like(t), t)
                    nll = (img_tok_recon - img_tok).pow(2).mean()
                    loss += args.nll * nll

                loss += reg * args.txt_reg

                if args.vae_refine:
                    x_recon = vae.module.decode(img_tok/scale_).sample
                    recon_loss = F.mse_loss(x_recon, imgs)
                    loss += recon_loss 

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()
            ema.update()

            epoch_loss += loss.item()
            epoch_img_tok_err += error1.item()
            epoch_embeddings_std += embeddings.std().item()
            epoch_tix_tok_std += txt_tok.std().item()
            epoch_derror += torch.mean(torch.square(dvdt_detach))

        # -----------------------------------------------------------------
        # Metrics aggregation (average across all GPUs)
        # -----------------------------------------------------------------
        epoch_loss = torch.tensor(epoch_loss / len(loader), device=device)
        epoch_embeddings_std = torch.tensor(epoch_embeddings_std / len(loader), device=device)
        epoch_tix_tok_std = torch.tensor(epoch_tix_tok_std / len(loader), device=device)
        epoch_derror = torch.tensor(epoch_derror / len(loader), device=device)
        epoch_img_tok_err = torch.tensor(epoch_img_tok_err / len(loader), device=device)

        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_embeddings_std, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_tix_tok_std, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_derror, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_img_tok_err, op=dist.ReduceOp.SUM)

        epoch_loss /= world_size
        epoch_embeddings_std /= world_size
        epoch_tix_tok_std /= world_size
        epoch_derror /= world_size

        elapsed = time.time() - start_time

        if rank == 0:
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                ckpt = {
                    "model": model.module.state_dict(),
                    "ema": ema.shadow,
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "vae": vae.module.state_dict(),
                }
                ckpt_path = f"last.pt"
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    dist.destroy_process_group()

################################################################################
# CLI
################################################################################

def build_parser():
    p = argparse.ArgumentParser(description="FlowTok‑Lite DDP: train or sample")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", type=str, default="flowers_blip_splits", help="Path to the dataset directory (Hugging Face format)")
    p_train.add_argument("--img_size", type=int, default=256)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--epochs", type=int, default=1000)
    p_train.add_argument("--ckpt_out", type=str, default="flowtok_mean_flow_")
    p_train.add_argument("--run_name", type=str, default="FlowTokLite")
    p_train.add_argument("--wandb", type=str, default="disabled")
    p_train.add_argument("--frozen_text_proj", type=bool, default=False)
    p_train.add_argument("--noise_scale", type=float, default=0.01)
    p_train.add_argument("--model", type=str, default="mfunet")
    p_train.add_argument("--alpha", type=float, default=0.0)
    p_train.add_argument("--flow_ratio", type=float, default=0.75)
    p_train.add_argument("--gamma", type=float, default=0.5)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--t_sample", type=str, default='uniform_1',)
    p_train.add_argument("--sob_lambda", type=float, default=1e-1)
    p_train.add_argument("--txt_reg", type=float, default=1e-4)
    p_train.add_argument("--save_every", type=int, default=1)
    p_train.add_argument("--rcvr_epochs", type=int, default=25)
    p_train.add_argument("--sample_epoch", type=int, default=9999,
                         help="Which epoch to load for sampling (0=latest)")
    p_train.add_argument("--use_pretrained", type=bool, default=False,
                         help="Whether to use the pretrained FlowTok‑MeanFlow model")
    p_train.add_argument("--reg_nll", type=float, default=0.0)
    p_train.add_argument("--vae_refine", type=bool, default=False)

    # DDP‑specific
    p_train.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    p_train.add_argument("--dist_port", type=str, default=os.environ.get("PORT", "29500"))

    # ---------------- sample (unchanged) ----------------
    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--ckpt", type=str, required=True)
    p_sample.add_argument("--prompt", type=str, required=True)
    p_sample.add_argument("--out", type=str, default="out.png")
    p_sample.add_argument("--steps", type=int, default=25)
    p_sample.add_argument("--sampler", choices=["euler", "rk38"], default="euler")

    return p

################################################################################
# Entry‑point
################################################################################

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    else:
        raise NotImplementedError("Sampling under DDP is not yet implemented in this refactor. Train the model first, then run a single‑GPU sampling script.")
