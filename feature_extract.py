"""
Extract features from data using conv net for use in linear regression
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import os

model = models.vgg16(pretrained = True)

# Use the model object to select the desired layer
layer = model._modules.get('features')

# Set model to evaluation mode
model.eval()

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

data_dir = '/Users/minter/Documents/all'
data = np.load(os.path.join(data_dir, 'stim.npy'))

for stim_im in enumerate(data.shape[0]):
    stim_vec = get_vector(data[stim_im,:,:,:])
    np.save(str(stim_im)+'features.npy', stim_vec)





# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
