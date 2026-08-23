# Rdkit import should be first, do not move it
import os
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion, get_meanflow, get_prompt_context_projection
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
# sample_center_gravity_zero_gaussian_with_mask
import qm9.utils as qm9utils
from equivariant_diffusion import utils as diffusion_utils
from qm9.analyze import check_stability, analyze_stability_for_molecules
from egnn.models import EGNN_dynamics_QM9
from functools import partial
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)
from accelerate import Accelerator
from equivariant_diffusion.EMA import EMA
import numpy as np
from models.model import load_model
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from accelerate.utils import set_seed
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask, prompt_context_x):
    assert len(size) == 3
    x = torch.randn(size, device=device) * 0.1 + prompt_context_x

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected

def sample_gaussian_with_mask(size, device, node_mask, prompt_context_h):
    x = torch.randn(size, device=device) * 0.1 + prompt_context_h
    x_masked = x * node_mask
    return x_masked

def adaptive_l2_loss(error: torch.Tensor, gamma: float = 1.0, c: float = 1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (w.detach() * delta_sq).mean()

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def make_targets(txt_tokens, img_tokens, t):
    # z_t = t[:, None, None, None] * img_tokens + (1 - t)[:, None, None, None] * txt_tokens
    z_t = (1-t)[:, None, None, None] * img_tokens + t[:, None, None, None] * txt_tokens
    v = (txt_tokens - img_tokens)  # Δ = 1
    return z_t, v

def class_balanced_weights(counts, beta=0.9999, device=None):
    """
    counts: Tensor[C] with class counts (float or long)
    returns: Tensor[C] weights normalized to mean=1
    """
    counts = torch.as_tensor(counts, dtype=torch.float32, device=device)
    eff_num = 1.0 - beta ** counts
    w = (1.0 - beta) / eff_num
    w = w / w.mean()  # normalize so average weight ≈ 1
    return w

counts = [120_847, 83_118, 13_297, 18_289, 332]

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true', 
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

# VAE args
parser.add_argument('--latent_nf', type=int, default=4,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb', type=str, default='disbaled', help='wandb mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')

parser.add_argument('--src', type=eval, default=True,
                    help='use source node features')
parser.add_argument('--lambda_v', type=float, default=1.0)
parser.add_argument('--noise_level', type=float, default=1e-2)
parser.add_argument('--t1_always', type=eval, default=False)
parser.add_argument('--reg_nll', type=float, default=0.0)
parser.add_argument('--stable_score', type=eval, default=False,)
# HF Accelerate related
parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], default='bf16')
parser.add_argument('--grad_accum_steps', type=int, default=1)

args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)


# ------------------------
# Accelerator & logging setup
# ------------------------
accelerator = Accelerator(mixed_precision=(None if args.mixed_precision == 'no' else args.mixed_precision),
                          gradient_accumulation_steps=args.grad_accum_steps)
set_seed(42, device_specific=True)

dtype = torch.float32

device = accelerator.device

# Disable W&B on worker processes to avoid duplicate logs
if not accelerator.is_main_process:
    os.environ["WANDB_DISABLED"] = "true"

# Dataset/info
dataset_info = get_dataset_info(args.dataset, args.remove_h)

# W&B (init only on main process)
if accelerator.is_main_process:
    kwargs = {
        'entity': 'XXX',
        'project': 'XXX',
        'mode': 'XXX',
        'name': 'XXX',
        'config': vars(args),
        'settings': wandb.Settings(_disable_stats=True),
        'reinit': True,
    }
    wandb.init(**kwargs)
    wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
train_ds = dataloaders['train'].dataset
valid_ds = dataloaders['valid'].dataset
test_ds = dataloaders['test'].dataset

# conditioning
data_dummy = next(iter(dataloaders['train']))
conditioning = ['homo', 'lumo', 'alpha', 'gap', 'mu', 'Cv']
property_norms = compute_mean_mad(dataloaders, conditioning, args.dataset)
context_dummy = prepare_context(conditioning, data_dummy, property_norms)
context_node_nf = 0
args.context_node_nf = context_node_nf

# Config
class CFG:
    n_layers = 3
    d_model = 256
    n_heads = 4
    seq_len = 20
    model = 'mfunet'
cfg = CFG()


meanflow = load_model(cfg)
count_parameters(meanflow)


train_loader = dataloaders['train']
valid_loader = dataloaders['valid']
test_loader = dataloaders['test']


# optimizer
optim = torch.optim.AdamW(params=meanflow.parameters(), 
                          lr=args.lr, 
                          betas=(0.9, 0.95), weight_decay=0.0)


Len = len(train_loader)
scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer = optim,
        num_warmup_steps = Len * 2,
        num_training_steps = Len * args.n_epochs,
        lr_end = 0,
        power = 1,
    )

meanflow, optim, scheduler, valid_loader, test_loader = accelerator.prepare(
    meanflow, optim, scheduler, valid_loader, test_loader
)

# EMA should track the *unwrapped* model
ema = EMA(accelerator.unwrap_model(meanflow), decay=0.9995)
pretrained_path = 'checkpoints2/epoch_500.pth'
if os.path.exists(pretrained_path):
    print(f"Loading pretrained model from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    meanflow.module.load_state_dict(checkpoint['model_state_dict'])

jvp_fn = partial(torch.autograd.functional.jvp, create_graph=True)

scale_ = 1
trans_ = 1
sigma_min = 1e-4
sample_freq = 20

def main():

    global ema

    for epoch in range(args.start_epoch, args.n_epochs):
        start = time.time()
        loss_epoch = 0.0
        epoch_z_eps_std = 0.0
        epoch_sample_count = 0
        epoch_abs_error = 0.0
        epoch_abs_std = 0.0
        epoch_mol_stable = 0.0
        epoch_atm_stable = 0.0
        epoch_mol_count = 0.0

        dict_cat = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }



        for i, data in enumerate(train_loader):


            x = data['positions'].to(device, dtype)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            prompt_context = qm9utils.prepare_context(conditioning, data, property_norms).to(device, dtype)
            context_tok = prompt_context[:, 0]
            num_atoms = data['num_atoms'].unsqueeze(-1).to(device, dtype) / 100.0
            context_tok = torch.cat([context_tok, num_atoms], dim=-1)

            if not args.src:
                context_tok = 0.1 * torch.randn_like(context_tok)  # dummy context

            xh = torch.cat([x, one_hot], dim=2)

            batch_max_size = xh.size(1)
            if xh.size(1) < 32:
                pad_size = 32 - xh.size(1)
                xh = torch.cat([xh, torch.zeros(xh.size(0), pad_size, xh.size(2), device=device)], dim=1)
                node_mask_pd = torch.cat([node_mask, torch.zeros(xh.size(0), pad_size, 1, device=device)], dim=1).long()

            xh[:,:,:3] = xh[:,:,:3] / scale_ + sigma_min * torch.randn(xh[:,:,:3].size(), device=device)
            phi_txt = meanflow.module.text_to_latent(context_tok)
            if epoch % sample_freq == 0:
                factor = 10
            else:
                factor = 1
            txt_tok = phi_txt + args.noise_level / factor * torch.randn(txt_tok.size(), device=device)  

            xh = xh * node_mask_pd
            txt_tok = txt_tok * node_mask_pd

            xh = xh.unsqueeze(1)
            txt_tok = txt_tok.unsqueeze(1)  #


            samples = torch.rand((xh.size(0), 2), device=device)
            t = torch.max(samples[:, 0], samples[:, 1])  # ensure t >= r
            t_mid = t
            r_ = torch.min(samples[:, 0], samples[:, 1])
            select = torch.rand(xh.size(0), device=device) < 0.75
            r_[select] = t[select]
            r = torch.zeros_like(t)                      # r is always zero in this case
            
            def u_fn(x, r_, t):
                return meanflow(x, t-r_, t)
            
            def pred_fn(x, r, t):
                return x - (t-r)[:, None, None, None] * meanflow(x, t-r, t)

            with accelerator.accumulate(meanflow):
                with accelerator.autocast():
                    z_t = (1-t)[:, None, None, None] * xh + t[:, None, None, None] * txt_tok
                    v = txt_tok - xh  

                    loss = 0.0
                    v_pred, dvdt_detach = jvp_fn(u_fn, 
                                                (z_t, r_, t), 
                                                (v, 
                                                torch.zeros_like(r_), 
                                                torch.ones_like(t)))
                    dvdt_detach = dvdt_detach.detach()
                    v_trgt = v - (t - r_)[:, None, None, None] * dvdt_detach
                    error_v = v_pred - v_trgt
                    error_v = error_v.squeeze(1) * node_mask_pd  # Apply the node mask.
                    loss += adaptive_l2_loss(error_v, gamma=0.5, c=1e-3)

                    # final state loss
                    if args.reg_nll > 0:
                        mu_pred = pred_fn(txt_tok, torch.zeros_like(r), 
                                          torch.ones_like(t))
                        
                        mu_pred = mu_pred[:,:,:,:3]
                        logits = mu_pred[:,:,:,3:]

                        # mse for the first 3 dimensions
                        error = mu_pred - xh[:,:,:,:3]
                        # make label based on one hot xh[:,:,:,3:]
                        target_onehot = xh[:,:,:,3:].long()
                        targets = target_onehot.argmax(dim=-1)
                        cw = class_balanced_weights(counts, beta=0.9998, device=logits.device)
                        loss_ce = F.cross_entropy(
                                    logits.view(-1, logits.size(-1)),    
                                    targets.view(-1),   
                                    weight=cw,                 
                                    reduction='none'
                                )   
                        loss_ce = loss_ce.view_as(targets).unsqueeze(-1)
                        loss_ce = loss_ce.squeeze(1) * node_mask_pd  # Apply the node mask.
                        error = error.squeeze(1) * node_mask_pd  # Apply the node mask.
                        w_sq_error = error ** 2
                        abs_error_mean = error.abs().sum() / (node_mask_pd.sum() * error.size(-1))

                        abs_diff_mean = ((error.abs() - abs_error_mean) ** 2).sum() / (node_mask_pd.sum() * error.size(-1))
                        abs_error_std = torch.sqrt(abs_diff_mean)

                        # final state loss
                        loss_nll_x = w_sq_error.sum() / node_mask_pd.sum()
                        loss_ce = loss_ce.sum() / node_mask_pd.sum()
                        loss_nll = loss_nll_x + loss_ce

                        # total loss
                        loss += loss_nll * args.reg_nll
                    

                    if args.stable_score:
                        with torch.no_grad():
                            x_pred = mu_pred + torch.randn_like(mu_pred) * 1e-3
                            molecules = {'one_hot': [], 'x': [], 'node_mask': []}
                            pred_cat = x_pred.squeeze(1) 
                            one_hot_rec = F.one_hot(pred_cat[:batch_max_size,:,3:], num_classes=5).long()
                            x_rec = x_pred.squeeze(1)
                            for kk in range(xh.size(0)):
                                one_hot_rec_ = one_hot_rec[kk, :batch_max_size, :]
                                x_ = x_rec[kk, :batch_max_size, :] 
                                # x_ = xh_recon[kk, :batch_max_size, :3] + 2e-2 * torch.randn_like(xh_recon[kk, :batch_max_size, :3]).sgn()
                                node_mask_ = node_mask_pd[kk, :batch_max_size,:]
                                molecules['one_hot'].append(one_hot_rec_)
                                molecules['x'].append(x_)
                                molecules['node_mask'].append(node_mask_)
                            validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)
                            mol_stable = validity_dict['mol_stable']
                            atm_stable = validity_dict['atm_stable']
                            epoch_mol_stable += mol_stable
                            epoch_atm_stable += atm_stable
                            epoch_mol_count += 1

                            # reward for stable molecules
                            reward = torch.tensor(mol_stable, device=device, dtype=dtype)
                            loss += (reward * (mu_pred - x_pred).pow(2).mean() + reward * (phi_txt - txt_tok).pow(2).mean()) * 1e-3 


            accelerator.backward(loss)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            # EMA update
            ema.update()

            loss_epoch += loss.item()
            epoch_z_eps_std += torch.std(txt_tok).mean().item()
            epoch_abs_error += abs_error_mean.item()
            epoch_abs_std += abs_error_std.item()
            epoch_sample_count += 1

        epoch_z_eps_std /= epoch_sample_count
        loss_epoch /= epoch_sample_count
        epoch_abs_error /= epoch_sample_count
        epoch_abs_std /= epoch_sample_count
        elapsed = time.time() - start
        
        accelerator.print(f"Epoch {epoch}, Loss: {loss_epoch}, Time: {elapsed}s, ")
        # Only main process logs to W&B
        if accelerator.is_main_process:

            wandb.log({
                'epoch': epoch,
                'loss': loss_epoch,
                'z_eps_std': epoch_z_eps_std,
                'time': elapsed,
                'lr': optim.param_groups[0]['lr'],
                'num_iter': epoch_sample_count
            })

            if epoch % sample_freq == 0:
                epoch_mol_stable /= epoch_mol_count
                epoch_atm_stable /= epoch_mol_count
                wandb.log({
                    'mol_stable': epoch_mol_stable,
                    'atm_stable': epoch_atm_stable
                })

            if (epoch+1) % 500 == 0 and epoch > -1:
                if args.src:
                    add_src = 'withsrc'
                else:
                    add_src = 'nosrc'

                # Unwrap to avoid "module." prefix and ensure compatibility w/ single-GPU load
                unwrapped = accelerator.unwrap_model(meanflow)
                state = {
                    "epoch": epoch,
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "ema": ema.shadow,
                }
                accelerator.save(state, f'checkpoints/epoch_{epoch+1}.pth')

            
    return

if __name__ == "__main__":
    main()