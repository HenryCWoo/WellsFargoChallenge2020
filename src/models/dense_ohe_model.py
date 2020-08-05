import torch
from torch import nn
from torch.nn import functional as F

from wells_fargo_dataset import XC_COUNT


class DenseOHE(nn.Module):
    ''' This model encodes the 'XC' column as a one hot encoding '''

    def __init__(self, device, bn=True, dropout=True, dropout_rate=0.5):
        super(DenseOHE, self).__init__()
        self.device = device
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.bn = bn

        self.layer_xc = self._hidden_layer(5, 16)
        self.layer_x = self._hidden_layer(30, 64)
        self.layer_1 = self._hidden_layer(64 + 16, 64)
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
            modules.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*modules)

    def forward(self, x, xc):
        xc = xc.squeeze()
        xc = xc.long()  # F.one_hot requires long type
        xc_ohe = F.one_hot(
            xc, num_classes=XC_COUNT).to(self.device, dtype=torch.float32)

        x = self.layer_x(x)
        xc = self.layer_xc(xc_ohe)

        x = torch.cat((x, xc), dim=1)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.output(x)

        return x
