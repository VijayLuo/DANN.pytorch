from torch import nn
from torch.autograd import Function
import config


class GRL(Function):
    """梯度翻转层
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """正向传播时不改变值"""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播时将梯度反转并乘以λ"""
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    """DANN模型定义
    """

    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha
        # 特征提取模块
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # 标签预测模块
        self.label_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, config.OUTPUT_SIZE),
            nn.Softmax(dim=1),
        )
        # 域预测模块
        self.domain_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, config.SUBJECTS_NUMBER-1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        label = self.label_predictor(feature)
        # 仅在训练时预测该输入所属的subject
        if self.training:
            feature = GRL.apply(feature, self.alpha)
            domain = self.domain_predictor(feature)
            return label, domain
        else:
            return label


class FCModel(nn.Module):
    """base line model
    仅由全连接层组成，相对于DANN模型，直接删除了域预测模块
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.label_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, config.OUTPUT_SIZE),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        label = self.label_predictor(feature)
        return label
