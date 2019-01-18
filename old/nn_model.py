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
        def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
            since = time.time()

            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train','val']:
                    if phase == 'train':
                        scheduler.step()
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

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
