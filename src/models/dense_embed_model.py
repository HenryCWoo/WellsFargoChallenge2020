import torch
from torch import nn
from torch.nn import functional as F

from wells_fargo_dataset import XC_COUNT

EMBED_VEC_SIZE = 16


class DenseEmbed(nn.Module):
    ''' This model encodes the 'XC' column as an embedding '''

    def __init__(self, device, bn=True, dropout=True, dropout_rate=0.5):
        super(DenseEmbed, self).__init__()
        self.device = device
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.bn = bn

        self.embed_xc = nn.Embedding(XC_COUNT, EMBED_VEC_SIZE)
        self.layer_xc = self._embedding_layer(32)

        self.layer_x = self._hidden_layer(30, 64)
        self.layer_1 = self._hidden_layer(64 + 32, 64)
        self.layer_2 = self._hidden_layer(64, 64)

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

    def _embedding_layer(self, output):
        modules = []
        modules.append(nn.Linear(EMBED_VEC_SIZE, output))

        if self.bn:
            modules.append(nn.BatchNorm1d(output))

        modules.append(nn.ReLU(inplace=True))

        if self.dropout:
            modules.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*modules)

    def forward(self, x, xc):
        x = self.layer_x(x)

        xc = self.embed_xc(xc.long()).squeeze()
        xc = self.layer_xc(xc)

        x = torch.cat((x, xc), dim=1)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.output(x)

        return x
