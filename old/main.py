"""
Extract features from data to predict spiking activity in V4 monkey neurons
"""

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.utils.data as utils
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from base_model import BaseModel
from nn_model import NNModel

def load_data(data_root):
    data = np.load(os.path.join(data_root, 'stim.npy'))
    data = data[50:-1,:,:,:] #remove initial testing images (without label)
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index
    target = df.values #(551x16) stim x avg sqrt of spikes
    target = target[:-1,:]
    # df.head() #show top couple rows of submission

    ## Remove last image from data and target - wasn't shown to more than 3 neurons, no info

    return data, target


# def base_model():


def feature_extract(method, data, target, subdir,
    stim_root='/Users/minter/Dropbox/uwndc19_data/'):
    stim_root = stim_root + subdir
    if method=='base':
        model = BaseModel()
        model.fit(data['train'], target['train'])
        model.predict(data['val'], target['val'])

    elif method=='nn':
        model = NNModel()


        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(stim_root, x),
            data_transforms[x]) for x in ['train', 'val']}

        dataloaders = {x: utils.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4) for x in ['train', 'val']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        model.fit(dataloaders)
        model.predict(dataloaders)

def main(
        data_root='/Users/minter/Dropbox/uwndc19_data/',
        all=True
    ):

    stim, response = load_data(data_root)

    if all:
        subdir = 'stim_full'
        # Dataset with all ims
        response = np.concatenate([response[:,:2], response[:,9:]],1)
    else:
        subdir = 'stim_partial'
        response = response[:,2:8]

    rs = ShuffleSplit(n_splits=5, test_size=0.1)

    splits = [(train , test) for train, test in rs.split(np.arange(stim.shape[0]))]

    data = {}
    target = {}

    for split in splits:
        data['train'] = stim[split[0]]
        data['val'] = stim[split[1]]
        target['train'] = response[split[0]]
        target['val'] = response[split[1]]
        feature_extract('nn', data, target, subdir)
        # features = feature_extract('base_model', train_data, train_target)

    # for train, test in cv.split(data):
    #     model.fit/train/do_the_thing(data[train], target[train])
    #     p = model.predict(data[test])
    #     score = scoring_function(target[test], p)

if __name__ == '__main__':
    main(all=True)





# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
