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

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x

    def rmse(x, y):
        return np.sqrt(np.mean((x-y) ** 2))


    def fit(self, dataloaders):

        # model = models.vgg16(pretrained = True)
        self.model = models.alexnet(pretrained = True)
        self.model.features = self.model.features[0] # first conv2d layer
        self.model.classifier = nn.Linear(64 * 55 * 55, 18) # 18 neurons

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.PoissonNLLLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        # model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
        self.model_ft = train_model(self.model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)





        # Use the model object to select the desired layer
        # layer = model._modules.get('features')

        # Set model to evaluation mode
        # model.eval()

        # scaler = transforms.Scale((224, 224))
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # to_tensor = transforms.ToTensor()


        # def get_vector(image_name):
        #     # 1. Load the image with Pillow library
        #     img = Image.open(image_name)
        #     # 2. Create a PyTorch Variable with the transformed image
        #     t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
        #     # 3. Create a vector of zeros that will hold our feature vector
        #     #    The 'avgpool' layer has an output size of 512
        #     my_embedding = torch.zeros(512)
        #     # 4. Define a function that will copy the output of a layer
        #     def copy_data(m, i, o):
        #         my_embedding.copy_(o.data)
        #     # 5. Attach that function to our selected layer
        #     h = layer.register_forward_hook(copy_data)
        #     # 6. Run the model on our transformed image
        #     model(t_img)
        #     # 7. Detach our copy function from the layer
        #     h.remove()
        #     # 8. Return the feature vector
        #     return my_embedding
    def predict(self, dataloaders):
        model.eval()

        test_data = np.transpose(test_data, (0,3,1,2))
        tensor_x = torch.tensor(test_data)
        tensor_y = torch.tensor(test_target)

        my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
        dataloaders = utils.DataLoader(my_dataset) # create your dataloader

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                import ipdb; ipdb.set_trace()
                _, preds = torch.max(outputs, 1)

                print('Root mean squared error', str(np.mean(rmse(labels, preds))))

                test_preds = np.corrcoef(labels, preds)[0,1] ** 2
                testr2 = r2_score(labels, preds)
                plt.plot(testr2, label='corrcoef^2');plt.xlabel('Recording Index');plt.ylabel(r'Test $R^2$');
                plt.plot(testpredr2, label='pred r2'); # Predictive score
                plt.legend()
                plt.xticks(range(len(testr2)));plt.title('Performance of NN model');
                plt.savefig(str(tag)+'_testr2_NN')
                plt.close()
