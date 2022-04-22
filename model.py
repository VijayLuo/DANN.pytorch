from torch import nn
from torch.autograd import Function
import config


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha
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

        if self.training:
            feature = GRL.apply(feature, self.alpha)
            domain = self.domain_predictor(feature)
            return label, domain
        else:
            return label
