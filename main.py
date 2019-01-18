import torch
from torchvision import models, transforms
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit

def rmse(x, y):
    return np.sqrt(np.mean((x-y) ** 2))


# TODO: btw all the models you make with convnet features, you'll be able to modify them by changing the size of the input image. That will make the network features apply to different scales in the image. Some may be better than others for certain neurons. (but leave that as a last step)
data_root = '/Users/minter/Dropbox/uwndc19_data/'

# Load data
# data = np.load(os.path.join(data_root,'stim.npy'))
# data = data[:-1] # Remove last image, not helpful
# mean=np.array([0.485, 0.456, 0.406])
# std=np.array([0.229, 0.224, 0.225])
# data = data.transpose(0,3,1,2)
# data = (data - mean[:, np.newaxis, np.newaxis] ) / std[:, np.newaxis, np.newaxis]
# data = torch.from_numpy(data.astype('float32'))
#
#
#
# # Load network
# model = models.alexnet(pretrained=True)
#
# # Run data through network
# for conv_layer in [3, 5, 8]:
#     model_ft = model.features[:conv_layer]
#     print(model_ft)
#     features = model_ft(data)
#     features = features.mean(-1).mean(-1)
#     save_name = 'alexnet_l'+str(conv_layer)+'_features.npy'
#     np.save(os.path.join(data_root, save_name), features.detach().numpy())

# Load Responses
df = pd.read_csv(os.path.join(data_root, 'train.csv'))
df.index = df.Id #turn the first col to index
df = df.iloc[:,1:] #get rid of first col that is now index
target = df.values #(551x16) stim x avg sqrt of spikes
target = target[:-1,:] #Remove last image from data and target - wasn't shown to more than 3 neurons, no info
# (TOGGLE) Dataset with all ims
target = np.concatenate([target[:,:2], target[:,9:]],1)
# target = target[:,2:8]

# Load features
features = np.load(os.path.join(data_root, 'alexnet_l8features.npy'))# load features

# Split data into different train/val sets
rs = ShuffleSplit(n_splits=5, test_size=0.1)
splits = [(train , test) for train, test in rs.split(np.arange(features[50:].shape[0]))]

# Fit
for idx, split in enumerate(splits):
    print('Split ', idx)
    import ipdb; ipdb.set_trace()
    reg = LinearRegression().fit(features[split[0]+50], target[split[0]]) # fit for images neurons have seen
    preds = reg.predict(features[split[1]+50])
    print('R2 Score = ', reg.score(features[split[1]+50], target[split[1]]))
    print('RMSE = ', rmse(target[split[1]], preds))
