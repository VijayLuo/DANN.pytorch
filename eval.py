import torch
from model import DANN
import config
from dataset import SEEDDataset
from torch.utils.data import DataLoader
from torch import nn

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"


def eval():
    for target_subject in config.SUBJECTS_NAME:
        print(f'target subject: {target_subject}')
        dataset = SEEDDataset(target_subject, train=False)
        dataloader = DataLoader(
            dataset, config.BATCH_SIZE, shuffle=True)

        model = DANN(alpha=config.ALPHA).to(device)
        model.load_state_dict(torch.load(f'weight/{target_subject}.pth'))
        
        loss_label = nn.CrossEntropyLoss()
        test_loss, correct = 0, 0

        model.eval()

        with torch.no_grad():
            for X, label in dataloader:
                pred = model(X)
                test_loss += loss_label(pred, label).item()

                print(pred)
                print(label)
                quit()
                correct += (pred.argmax(1) ==
                            label).type(torch.float).sum().item()


if __name__ == '__main__':
    eval()
