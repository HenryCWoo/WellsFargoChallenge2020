import torch
from torch import nn
from torch.nn import functional as F


class FCN(nn.Module):
    def __init__(self, bn=True, dropout=True):
        super(FCN, self).__init__()
        self.dropout = dropout
        self.bn = bn

        self.layer_1 = self._hidden_layer(30, 64)
        self.layer_2 = self._hidden_layer(64, 64)
        self.layer_3 = self._hidden_layer(64, 64)

        self.output = nn.Linear(64, 1)

    def _hidden_layer(self, input, output):
        modules = []
        modules.append(nn.Linear(input, output))

        if self.bn:
            modules.append(nn.BatchNorm1d(output))

        modules.append(nn.ReLU(inplace=True))

        if self.dropout:
            modules.append(nn.Dropout(output))

        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.output(x)

        return x
