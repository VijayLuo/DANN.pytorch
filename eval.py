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
        model.load_state_dict(torch.load(
            f'weight/{target_subject}.pth', map_location='cpu'))

        loss_label = nn.CrossEntropyLoss()
        test_loss, correct = 0, 0

        model.eval()

        with torch.no_grad():
            for X, label in dataloader:
                pred = model(X)
                test_loss += loss_label(pred, label).item()

                correct += (label.argmax(1) == pred.argmax(
                    1)).type(torch.float).sum().item()

        print(correct/config.DATA_NUMBER_OF_SUBJECT)
        print(test_loss/config.DATA_NUMBER_OF_SUBJECT)


if __name__ == '__main__':
    eval()
