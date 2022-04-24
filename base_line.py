import torch
import config
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
from util import weights_init
from dataset import SEEDDataset
from model import FCModel

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    for target_subject in config.SUBJECTS_NAME:
        print(f'target subject: {target_subject}')
        dataset = SEEDDataset(target_subject, train=True)
        dataloader = DataLoader(
            dataset, config.BATCH_SIZE, shuffle=True)

        linear_model = FCModel()
        model = linear_model

        print('Initializing weights...')
        model.apply(weights_init)

        model.to(device)

        loss_label = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = ExponentialLR(optimizer, config.GAMMA)

        model.train()
        for epoch in range(config.EPOCH):
            loss_y_epoch = 0
            for batch, (X, label, subject) in enumerate(dataloader):
                X, label = X.to(device), label.to(
                    device)
                pred_label = model(X)
                loss_y = loss_label(pred_label, label)

                loss_y_epoch += loss_y.item()

                loss = loss_y
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_y_epoch = loss_y_epoch/len(dataloader)
            if(epoch % 10 == 0):
                print(
                    f'loss_y: {loss_y_epoch:>7f} epoch: {epoch}')
            if(epoch in config.LEARNING_STEP):
                scheduler.step()
        torch.save(linear_model.state_dict(),
                   f'weight/fc_{target_subject}.pth')
        print(f'Saved model state to fc_{target_subject}.pth')


if __name__ == '__main__':
    train()
