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
# from base_model import BaseModel
from nn_model import NNModel

def load_data(data_root, all=True, resize=None):
    """
    resize 0 = pad with zeros to 224
    resize 1 = resize to 224 using PIL resize
    """
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    # Stimuli
    stim = np.load(os.path.join(data_root, 'stim.npy'))
    stim = stim[50:-1,:,:,:] #remove initial testing images (without label)
    if resize is not None:
        data = np.empty((stim.shape[0], 224, 224, stim.shape[3]))
        for im in range(stim.shape[0]):
            if resize == 1:
                data[im,:,:,:] = np.array(Image.fromarray((stim[im]*255).astype('uint8')).resize((224,224)))
            elif resize == 0:
                data[im,:,:,:] = np.pad(stim[im], ((72,72), (72,72), (0, 0)), 'constant', constant_values=0)
    else:
        data = stim

    # Responses
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index
    target = df.values #(551x16) stim x avg sqrt of spikes
    target = target[:-1,:] #Remove last image from data and target - wasn't shown to more than 3 neurons, no info

    if all:
        # Dataset with all ims
        target = np.concatenate([target[:,:2], target[:,9:]],1)
    else:
        target = target[:,2:8]

    return data, target

def create_dataloader(data, target):
    dataset = {'train': utils.TensorDataset(torch.Tensor(data['train']),torch.Tensor(target['train'])), 'val': utils.TensorDataset(torch.Tensor(data['val']),torch.Tensor(target['val']))} # create your datset
    dataloaders = {x: utils.DataLoader(dataset[x]) for x in ['train', 'val']}
    return dataloaders

def feature_extract(
        method,
        data,
        target,
        subdir,
        stim_root='/Users/minter/Dropbox/uwndc19_data/'):
    stim_root = stim_root + subdir

    if method=='nn':
        model = NNModel()
        dataloaders = create_dataloader(data, target)
        model.fit(dataloaders)
        model.predict(dataloaders)


def main(
    data_root='/Users/minter/Dropbox/uwndc19_data/',
    all=True
    ):

    # Load stimuli and responses, toggle for full and partial sets
    stim, response = load_data(data_root, all=all, resize=1)

    if all:
        subdir = 'stim_full'
    else:
        subdir = 'stim_partial'

    rs = ShuffleSplit(n_splits=5, test_size=0.1)

    splits = [(train , test) for train, test in rs.split(np.arange(stim.shape[0]))]

    for split in splits:
        data = {'train' : stim[split[0]], 'val': stim[split[1]]}
        target = {'train' : response[split[0]], 'val' : response[split[1]]}
        feature_extract('nn', data, target, subdir)

if __name__ == '__main__':
    main(all=True)
