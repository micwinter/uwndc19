"""
SpatioTemporal GRU in PyTorch
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class SGRU_Layer(nn.Module):
    """SGRU Layer based on GRU"""

    def __init__(self, input_size, hidden_size):

        super(SGRU_Layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size #(3 gates)
        # Input Size (batch_size, 4, 3, 64, 64)
        self.conv_u = nn.Conv2d(3, 3*hidden_size, 3, padding=1) #3 input channels
        self.conv_w = nn.Conv2d(hidden_size, 3*hidden_size, 3, padding=1)

    def forward(self, input_, h0):
        ux = self.conv_u(input_) #(batch_size,96,62,62)
        wh = self.conv_w(h0) #(batch_size,96,62,62)
        uxz, uxr, uxh = torch.split(ux, self.hidden_size, dim=1) #(batch_size,32,62,62)
        whz, whr, foo = torch.split(wh, self.hidden_size, dim=1) #(batch_size,32,62,62)
        z = torch.sigmoid(uxz + whz) # update gate (batch_size,32,64,64)
        rx = torch.sigmoid(uxr) # spatial reset gate
        rh = torch.sigmoid(whr) # temporal reset gate
        n1 = torch.tanh((uxh*rx) + self.conv_w(rh*h0)[:,-self.hidden_size:,:,:]) # hadamard products
        h1 = (z*h0) + ((1-z)*n1) #(batch_size,32,64,64)

        return h1
