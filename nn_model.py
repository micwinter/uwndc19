"""
Setup for nn class from UWNDC19 challenge
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as utils
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time


class NNModel:

    def __init__(self):
        self = self
        model = models.alexnet(pretrained = True)
        self.features = self.model.features[0] # first conv2d layer
        self.classifier = nn.Linear(64 * 55 * 55, 18) # 18 neurons
