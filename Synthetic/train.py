import torch
import os
from functools import partial
from transformers import get_polynomial_decay_schedule_with_warmup
import wandb
from model import EMA
from model import MLP
from model import adaptive_l2_loss
from utils import sample_2d

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

jvp_fn = partial(torch.autograd.functional.jvp, create_graph=True)

def train(task='checkerboard',  
          step = 8,
          lambda_ = 0.0,
          lr = 1e-4, 
          batch_size = 256,
          iterations = 1e5,
          hidden_dim = 256, 
          seed=42,
          path="OT"):

    # velocity field model init
    torch.manual_seed(seed)
    vf = MLP(input_dim=2, 
             time_dim=1, 
             hidden_dim=hidden_dim).to(device) 
    # cont the number of parameters
    num_params = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')
    ema = EMA(vf, decay=0.99)
    # vf = torch.compile(vf, backend="inductor")

    # init optimizer
    optim = torch.optim.Adam(vf.parameters(), lr=lr) 
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9999)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=int(iterations * 0.01),
        num_training_steps=iterations,
        lr_end=1e-12,                # final learning rate
        power=1                  # 1.0 = linear, 2.0 = quadratic, etc.
    )
    
    loss_cum = 0.0

    if task == 'checkerboard':
        if not os.path.exists('checker_2d_data.pt'):
            print('Data not found!')
            exit()
        print('Data found!')

        # load the samples pairs
        data_dict = torch.load('checker_2d_data.pt')
        gaussian_prior = data_dict['gaussian_prior']
        x_1 = data_dict['x_1']
        gd = data_dict['gd']
        
    elif task == 'mixG':
        if not os.path.exists('mixG_1d_data.pt'):
            print('Data not found!')
            exit()
        print('Data found!')

        # load the samples pairs
        data_dict = torch.load('mixG_1d_data.pt')
        gaussian_prior = data_dict['gaussian_prior']
        x_1 = data_dict['x_1']
        gd = data_dict['gd']

    # data loader 
    data_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(gaussian_prior.to(device), x_1.to(device), gd.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )

    global_step = 0
    for epoch in range(int(iterations)//(len(data_loader)) + 1):

        loss_cum = 0.0
        tok_std = 0.0
        dvdt_cum = 0.0
        nll_cum = 0.0
        mf_err_cum = 0.0

        for i, (guassian_batch, x_1_batch, _) in enumerate(data_loader):

            optim.zero_grad() 

            img_tok = x_1_batch 
            txt_tok = guassian_batch
            tok_std += txt_tok.std().item()
            
            samples = torch.rand((img_tok.size(0), 2), device=device)
            t = torch.max(samples[:, 0], samples[:, 1])
            r = torch.min(samples[:, 0], samples[:, 1])
            
            flow_ratio = 0.5
            select = torch.rand(img_tok.size(0), device=device) < flow_ratio
            r[select] = t[select]

            x_t = (1 - t)[:, None] * img_tok + t[:, None] * txt_tok
            v_t = txt_tok - img_tok

            def u_fn(x, t, r):
                return vf(x, t, t-r)
            v_pred, dvdt = jvp_fn(u_fn,
                                (x_t, t, r),
                                (v_t, torch.ones_like(t), torch.zeros_like(t))
                                )
            
            v_tgt = (v_t - (t-r)[:, None] * dvdt).detach()
            dvdt_cum += dvdt.abs().mean().item()
            gamma_r = 0.5
            loss = adaptive_l2_loss(v_pred - v_tgt, gamma=gamma_r, c=1e-6)
            mf_err_cum += (v_pred - v_tgt).pow(2).mean().item()

            if lambda_ > 0:
                mu_pred = txt_tok - u_fn(txt_tok, 
                                        torch.ones_like(t), 
                                        torch.zeros_like(t))
                err = (mu_pred - img_tok)
                loss = loss + err.pow(2).mean() * lambda_ 
                nll_cum += 0.5 * err.pow(2).mean().item() / 1e-3
            else:
                with torch.no_grad():
                    mu_pred = txt_tok - u_fn(txt_tok, 
                                        torch.ones_like(t), 
                                        torch.zeros_like(t))
                    err = (mu_pred - img_tok)
                    nll_cum += 0.5 * err.pow(2).mean().item() / 1e-3

            loss.backward()
            optim.step()
            scheduler.step() 
            ema.update()
            global_step += 1
            loss_cum += loss.item()

            # print(f'epoch: {epoch}, iter: {i}, loss: {loss.item():.6f}')
        
        lr = scheduler.get_last_lr()[0]
        loss_epoch = loss_cum / (i + 1)
        tok_std = tok_std / (i + 1)
        nll_cum = nll_cum / (i + 1)
        mf_err_cum = mf_err_cum / (i + 1)

        print(f'epoch: {epoch}, loss: {loss_epoch:.6f}, lr: {lr:.2e}')
        wandb.log({"train/loss": loss_epoch}, step=global_step)
        wandb.log({"train/tok_std": tok_std}, step=global_step)
        wandb.log({"train/lr": lr}, step=global_step)
        wandb.log({"train/dvdt": dvdt_cum / (i + 1)}, step=global_step)
        wandb.log({"train/nll": nll_cum}, step=global_step)
        wandb.log({"train/mf_err": mf_err_cum}, step=global_step)

        if (epoch) % 10 == 0:
            ema.apply_shadow()
            kl, tv = sample_2d(vf, data_loader, lambda_, step)
            ema.restore()
            print('sampling done')
            wandb.log({"sampling": wandb.Image('cmf_on_checkerboard.png'.format(int(lambda_*1000), step))}, step=global_step)
            wandb.log({"eval/kl": kl}, step=global_step)
            wandb.log({"eval/tv": tv}, step=global_step)
            # save model and ema model as dict
            torch.save({'model': vf.state_dict(),
                        'ema': ema.shadow,
                        'optim': optim.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'global_step': global_step}, 'mf_2d_checkpoint.pth'.format(int(lambda_*1000), step))

            print('model saved')

    return