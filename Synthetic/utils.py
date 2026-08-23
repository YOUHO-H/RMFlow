import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def compute_kld(pdf_p, pdf_q):
    return (pdf_p * (pdf_p / pdf_q).log()).sum()

def compute_tvd(pdf_p, pdf_q):
    return 0.5 * (pdf_p - pdf_q).abs().sum()

def compute_density(H):
    counts, xedges, yedges, img = H
    N = counts.sum()
    pdf = counts / (N) + 1e-12
    return pdf

def sample_2d(vf, data_loader, lambda_, step):

    with torch.no_grad():
        for i, (_, _, gd_batch) in enumerate(data_loader):
            # sample prior from N(0, I)
            txt_tok = torch.randn_like(gd_batch)
            if i == 0:
                sol = txt_tok
                gd = gd_batch
            else:
                sol = torch.cat((sol, txt_tok), dim=0)
                gd = torch.cat((gd, gd_batch), dim=0)

        def u_fn(x, t, r):
            return vf(x, t, t-r)

        K = step
        t_1 = torch.ones(sol.shape[0], dtype=torch.float32, device=device)
        for k in range(K, 0, -1):
            sol = sol - u_fn(sol, t_1 * k / K, t_1 * (k-1) / K) * (1 / K)
        sol += torch.randn_like(sol) * 1e-3
        
    sol = sol.cpu().numpy()
    gd = gd.cpu().numpy()
    fig, axs = plt.subplots(1, 1, figsize=(16, 16))
    H_gd = axs.hist2d(gd[:,0], gd[:,1], 128, range=[[-1, 1], [-1, 1]], cmap='copper')
    pdf_gd = compute_density(H_gd) 
    norm = cm.colors.Normalize(vmax=50, vmin=0)
    H = axs.hist2d(sol[:,0], sol[:,1], 128, range=[[-1, 1], [-1, 1]], cmap='copper')
    pdf_gen = compute_density(H)
    
    kl = compute_kld(torch.from_numpy(pdf_gd), torch.from_numpy(pdf_gen)).item()
    tvd = compute_tvd(torch.from_numpy(pdf_gd), torch.from_numpy(pdf_gen)).item()

    cmin = 0.0
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    _ = axs.hist2d(sol[:,0], sol[:,1], 64, range=[[-1, 1], [-1, 1]], norm=norm, cmap='copper')
    axs.set_aspect('equal')
    axs.axis('off')

    plt.tight_layout()
    plt.savefig('cmf_on_checkerboard_{}_{}.png'.format(int(lambda_*1000), step), 
                dpi=300)

    return kl, tvd