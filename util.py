from torch import nn
from torch.nn import init


def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
