import torch
from model import DANN, FCModel
import config
from dataset import SEEDDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import wandb

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"


def eval():
    """对DANN模型与FC模型进行评估，
    输出每个subject作为target subject的准确率及其mean
    """
    accuracy_fc_list = []
    accuracy_dann_list = []
    for target_subject in config.SUBJECTS_NAME:
        print(f'target subject: {target_subject}')
        # 载入SEED数据库
        dataset = SEEDDataset(target_subject, train=False)
        dataloader = DataLoader(
            dataset, config.BATCH_SIZE, shuffle=True)

        # 加载模块weight
        dann = DANN(alpha=config.ALPHA).to(device)
        dann.load_state_dict(torch.load(
            f'weight/dann_{target_subject}.pth'))

        fc = FCModel().to(device)
        fc.load_state_dict(torch.load(
            f'weight/fc_{target_subject}.pth'))

        loss_label = nn.CrossEntropyLoss()
        test_loss_dann, correct_dann = 0, 0
        test_loss_fc, correct_fc = 0, 0

        dann.eval()
        fc.eval()

        with torch.no_grad():
            for X, label in dataloader:
                X, label = X.to(device), label.to(device)
                pred_dann = dann(X)
                test_loss_dann += loss_label(pred_dann, label).item()

                correct_dann += (label.argmax(1) == pred_dann.argmax(
                    1)).type(torch.float).sum().item()
                # 预测结果
                pred_fc = fc(X)

                test_loss_fc += loss_label(pred_fc, label).item()
                correct_fc += (label.argmax(1) == pred_fc.argmax(
                    1)).type(torch.float).sum().item()
        # 计算准确率以及test_loss
        accuracy_dann = correct_dann/config.DATA_NUMBER_OF_SUBJECT
        accuracy_dann_list.append(accuracy_dann)
        test_loss_dann /= len(dataloader)
        print(
            f'DANN Test Error: \n Accuracy: {(100*accuracy_dann):>0.1f}%, Avg loss: {test_loss_dann:>8f}')

        accuracy_fc = correct_fc/config.DATA_NUMBER_OF_SUBJECT
        accuracy_fc_list.append(accuracy_fc)
        test_loss_fc /= len(dataloader)
        print(
            f'Fully Connected Model Test Error: \n Accuracy: {(100*accuracy_fc):>0.1f}%, Avg loss: {test_loss_fc:>8f}')

    print(
        f'DANN Average Accuracy: {(100*np.mean(accuracy_dann_list)):>0.1f}%')

    print(
        f'Fully Connected Model Average Accuracy: {(100*np.mean(accuracy_fc_list)):>0.1f}%')
    wandb.log({'mean_accuracy': np.mean(accuracy_dann_list)})


if __name__ == '__main__':
    eval()
