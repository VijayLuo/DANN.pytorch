import wandb
import config

# 初始化wandb sweep，对超参数进行grid search
if __name__ == '__main__':
    sweep_id = wandb.sweep(config.SWEEP_CONFIG, entity='vj', project='dann')
