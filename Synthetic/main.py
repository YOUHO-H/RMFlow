import wandb
from train import train

if __name__ == "__main__":
    task = 'checkerboard' # 'mixG' or 'checkerboard'
    step = 1
    lambda_ = 1e-2
    kwargs = {
        'entity': 'XXX', 
        'project': 'XXX',
        'mode': 'online',
        'name': 'XXX',
        'settings': wandb.Settings(_disable_stats=True), 'reinit': True
        }
    wandb.init(**kwargs)
    wandb.save('*.txt')
    train(task=task,
          step=step, 
          lambda_=lambda_, 
          lr=1e-4, 
          batch_size=256, 
          iterations=1e5, 
          hidden_dim=256, 
          seed=42)