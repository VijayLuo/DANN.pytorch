import wandb
import config


if __name__ == '__main__':
    sweep_id = wandb.sweep(config.SWEEP_CONFIG, entity='vj', project='dann')
    print(f'sweep id: {sweep_id}')
