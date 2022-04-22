import torch
import config
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import init
from torch import nn
from model import DANN
from dataset import SEEDDataset

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"


def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def train():
    for target_subject in config.SUBJECTS_NAME:
        print(f'target subject: {target_subject}')
        dataset = SEEDDataset(target_subject, train=True)
        dataloader = DataLoader(
            dataset, config.BATCH_SIZE, shuffle=True)

        dann = DANN(alpha=config.ALPHA).to(device)
        model = dann

        if(torch.cuda.is_available()):
            model = torch.nn.DataParallel(dann)

        print('Initializing weights...')
        dann.apply(weights_init)

        loss_label = nn.CrossEntropyLoss()
        loss_subject = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

        model.train()
        for epoch in range(config.EPOCH):
            loss_y_epoch = 0
            loss_d_epoch = 0
            for batch, (X, label, subject) in enumerate(dataloader):
                X, label, subject = X.to(device), label.to(
                    device), subject.to(device)
                pred_label, pred_subject = model(X)

                loss_y = loss_label(pred_label, label)
                loss_d = loss_subject(pred_subject, subject)

                loss_y_epoch += loss_y.item()
                loss_d_epoch += loss_d.item()

                loss = loss_y + loss_d
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_y_epoch = loss_y_epoch/len(dataloader)
            loss_d_epoch = loss_d_epoch/len(dataloader)
            print(
                f'loss_y: {loss_y_epoch:>7f}  loss_d: {loss_d_epoch:>7f}  epoch: {epoch}')
        torch.save(model.state_dict(),
                   f'weight/{target_subject}.pth')
        print(f'Saved model state to {target_subject}.pth')


if __name__ == '__main__':
    train()
