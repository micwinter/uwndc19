"""
Baseline Model Structure for V4 Neuron Prediction
"""

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

__all__ = ['baseline', 'BASELINE']

class BaseModel(nn.Module):
    def __init__(self, input_dim, layer, output_dim):
        super(BaseModel, self).__init__()

        self.feature_layers = models.vgg16(pretrained=True).features[:layer]
        model_out = self.feature_layers(torch.rand(1,3,80,80))
        self.raw_mask = nn.Parameter(torch.ones(output_dim, model_out.shape[-1], model_out.shape[-1]))
        self.fcs = nn.ModuleList([])
        for ii in range(output_dim):
            self.fcs.append(nn.Linear(model_out.shape[-3], 1))
        self.output_dim = output_dim
        
    def forward(self, x):
        #mask = torch.exp(self.raw_mask)/(torch.exp(self.raw_mask).sum())
        mask = self.raw_mask/self.raw_mask.sum()
        #print('Max %8.2f, min %8.2f of raw_mask' % (torch.max(self.raw_mask), torch.min(self.raw_mask)))
        #print('Max %8.2f, min %8.2f of mask' % (torch.max(mask), torch.min(mask)))
        x = self.feature_layers(x)
        x = (x[:,:, None] * mask).sum(-1).sum(-1)
        all_outputs = torch.zeros(x.shape[0],self.output_dim)
        # Separate output linear layer for each neuron
        for ii in range(self.raw_mask.shape[0]):
            curr_neuron = x[:,:,ii]
            neuron_out = self.fcs[ii](curr_neuron.view(curr_neuron.shape[0],-1))
            all_outputs[:,ii] = torch.squeeze(neuron_out)
        
        return all_outputs
