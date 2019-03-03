"""
SpatioTemporal GRU Model Structure
"""

import torch
import torch.nn as nn
import models
from models.sgru_layer import SGRU_Layer
from torch.autograd import Variable

__all__ = ['sgru', 'SGRU']

class SGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(SGRUModel, self).__init__()

        self.hidden_dim = hidden_dim # hidden dims
        self.layer_dim = layer_dim # num layers

        self.sgru_layer = SGRU_Layer(input_dim, hidden_dim)
        self.selu = nn.SELU()
        self.bnorm = nn.BatchNorm3d(hidden_dim)
        self.maxpool = nn.MaxPool3d((1,2,2), stride=(1,2,2)) # spatial pooling only
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim**3*int(hidden_dim/2), output_dim)
        
    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.shape[0],self.hidden_dim, x.shape[3], x.shape[3]).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.shape[0],self.hidden_dim, x.shape[3],x.shape[3]))

        hidden = h0[0,:]

        embed = []

        for timestep in range(x.shape[1]): # x[:,timestep,:,:,:] -- (batchsize, 3, 64, 64)
            hidden = self.sgru_layer(x[:,timestep,:,:,:], hidden) #(batchsize,32,64,64)
            hidden = self.selu(hidden)
            embed.append(hidden)

        embed = torch.stack(embed, dim=-3) # (batchsize, channel, time, H, W)
        embed = self.bnorm(embed) #(batchsize,32,16,64,64) norm is done over 32 (channel dim)
        embed = self.maxpool(embed) #(batchsize,32,16,32,32)
        #embed = self.fc(embed.view(embed.shape[0],-1)) #(batchsize,16*32*32*32)
        #embed = self.dropout(embed) #(batchsize,32,32,32)
        embed = self.fc(embed.view(embed.shape[0],-1)) #(32**3*16 --> 1)
        return embed
