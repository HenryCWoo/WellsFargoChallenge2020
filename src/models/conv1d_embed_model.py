import torch
from torch import nn
from torch.nn import functional as F

from wells_fargo_dataset import XC_COUNT

EMBED_VEC_SIZE = 16


class Conv1DEmbed(nn.Module):
    ''' This model encodes the 'XC' column as an embedding '''

    def __init__(self, device, bn=True, dropout=True, dropout_rate=0.5, hidden_layers=1, hidden_units=128, conv_blocks=1, kernel_size=5, filters=32):
        super(Conv1DEmbed, self).__init__()
        self.device = device
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.bn = bn

        # Conv block parameters
        self.conv_blocks_no = conv_blocks
        self.kernel_size = kernel_size
        self.filters = filters

        # Dense block parameters
        self.hidden_layers_no = hidden_layers
        self.hidden_units_no = hidden_units

        self.embed_xc = nn.Embedding(XC_COUNT, EMBED_VEC_SIZE)
        self.layer_xc = self._embedding_layer(32)

        self.layer_x = self._conv_block(1, filters, kernel_size)
        self.conv_blocks = self._conv_blocks()
        self.dropout = nn.Dropout(self.dropout_rate)

        self.dense_1 = self._hidden_layer(
            filters * (30 - ((kernel_size - 1) * (conv_blocks + 1))) + 32, hidden_units)
        self.hidden_layers = self._hidden_layers()
        self.output = nn.Linear(hidden_units, 1)

    def _conv_block(self, input, output, kernel_size):
        modules = []
        modules.append(nn.Conv1d(input, output, kernel_size=kernel_size))
        modules.append(nn.ReLU(inplace=True))
        return nn.Sequential(*modules)

    def _conv_blocks(self):
        modules = []
        for _ in range(self.conv_blocks_no):
            modules.append(self._conv_block(
                self.filters, self.filters, self.kernel_size))
        return nn.Sequential(*modules)

    def _hidden_layer(self, input, output):
        modules = []
        modules.append(nn.Linear(input, output))

        if self.bn:
            modules.append(nn.BatchNorm1d(output))

        modules.append(nn.ReLU(inplace=True))

        if self.dropout:
            modules.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*modules)

    def _hidden_layers(self):
        modules = []
        for _ in range(self.hidden_layers_no):
            modules.append(self._hidden_layer(
                self.hidden_units_no, self.hidden_units_no))
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
        xc = self.embed_xc(xc.long()).squeeze()
        xc = self.layer_xc(xc)

        x = x.unsqueeze(1)
        x = self.layer_x(x)
        x = self.conv_blocks(x)

        if self.dropout:
            x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = torch.cat((x, xc), dim=1)
        x = self.dense_1(x)
        x = self.hidden_layers(x)
        x = self.output(x)

        return x
