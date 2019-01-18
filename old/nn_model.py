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
        self.model = models.alexnet(pretrained = True)
        import ipdb; ipdb.set_trace()
        self.model.features = self.model.features[:5] # first conv2d layer
        self.model.classifier = nn.Linear(64 * 55 * 55, 18) # 18 neurons

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x

    def fit(self, dataloaders):

        def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
            since = time.time()

            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:

                    import ipdb; ipdb.set_trace()
                    dataset_sizes = {x: len(dataloaders[x][0]) for x in ['train', 'val']}

                    if phase == 'train':
                        scheduler.step()
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.forward(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights
            model.load_state_dict(best_model_wts)
            return model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.PoissonNLLLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        # model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
        self.model_ft = train_model(self.model, criterion, optimizer_ft, exp_lr_scheduler,
                                    num_epochs=25)

    def predict(self):
        print('1')

def rmse(x, y):
    return np.sqrt(np.mean((x-y) ** 2))

def nnmodel():
    model = NNModel()
    return model
