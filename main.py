"""
Model where you fit Lasso and Ridge regression in 3 parts.
1. Run stimuli through Alexnet and extract features from every layer.
2. Take the features and fit Lasso regression on them to predict the spike rates for each neuron. (Feature selection)
2. Then, take the features with nonzero Lasso coefficients and fit Ridge regression to predict the spike rates.
"""

import torch
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import models

def rmse(x, y):
    return np.sqrt(np.mean((x-y) ** 2))

def load_data(data_root):
    # Load data
    data = np.load(os.path.join(data_root,'stim.npy'))
    data = data[:-1] # Remove last image, not helpful
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    data = data.transpose(0,3,1,2)
    data = (data - mean[:, np.newaxis, np.newaxis] ) / std[:, np.newaxis, np.newaxis]
    data = torch.from_numpy(data.astype('float32'))

    return data


def get_features(data):
    # Load network
    model = models.alexnet(pretrained=True)  # Alexnet model, worked better on these neurons
    # model = models.vgg16(pretrained=True)  # VGG16 model

    # Run data through network
    ext_feat = {}
    for layer in range(len(model.features)):
        model_ft = model.features[:layer]
        features = model_ft(data)
        features = features
        ext_feat[layer] = features.detach().numpy()
        print('layer ', layer, ' of ', len(model.features), ' complete')
    save_name = 'alexnet_features.npy'
    print('Saving file...')
    np.save(os.path.join(data_root, save_name), ext_feat)

def predict(data_root, features_name, alpha=None):
    # Load Responses
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index
    target = df.values #(551x18) stim x avg sqrt of spikes
    target = target[:-1,:] #Remove last image from data and target - wasn't shown to more than 3 neurons, no info

    ## Load features
    net_features = np.load(os.path.join(data_root, features_name)).item()# load features

    # Shuffle indices
    rs = ShuffleSplit(n_splits=5, test_size=0.1)

    ###########  LASSO and then RIDGE ###########
    lasso_idx = {}

    for neuron in range(target.shape[1]):
        print('Neuron: ', neuron)
        perf_r2 = []
        perf_rmse = []

        lasso_idx[neuron] = {}

        for layer in net_features.keys():
            print('Layer: ', layer)

            # Take images in train dataset
            features = net_features[layer][50:]

            # Split data into different train/val sets
            curr_target = target[:,neuron]
            curr_target = curr_target[~np.isnan(curr_target)]
            splits = [(train , test) for train, test in rs.split(np.arange(curr_target.shape[0]))]

            split_r2 = []
            split_rmse = []

            for idx, split in enumerate(splits):
                train_features = features[split[0]].reshape(features[split[0]].shape[0], -1)
                val_features = features[split[1]].reshape(features[split[1]].shape[0], -1)

                # Fit LASSO
                lassoreg = LassoLarsCV(cv=5, n_jobs=2)
                lassoreg.fit(train_features, curr_target[split[0]])

                # Get features with nonzero coefs
                train_features = train_features[:,np.where(lassoreg.coef_)[0]]
                if np.where(lassoreg.coef_)[0].size == 0: #all coefs are zero
                    split_r2.append(0)
                    split_rmse.append(1)
                    continue

                val_features = val_features[:,np.where(lassoreg.coef_)[0]]

                # Save indices to dictionary
                lasso_idx[neuron][layer] = np.where(lassoreg.coef_)[0]

                np.save(os.path.join(data_root, 'neuron_'+str(neuron)+'_layer_'+str(layer)+'.npy'), np.where(lassoreg.coef_)[0])


                # Train linear model on these pixels
                reg = Ridge(alpha)
                # Grid search on alpha
                parameters = {'alpha':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]}
                clf = GridSearchCV(reg, parameters, cv=5)
                clf.fit(train_features, curr_target[split[0]])
                preds = clf.predict(val_features)
                r2_score = clf.score(val_features, curr_target[split[1]])
                rmse_score = rmse(curr_target[split[1]], preds)

                # save scores for split
                split_r2.append(r2_score)
                split_rmse.append(rmse_score)

                print('R2 Score = ', r2_score)
                print('RMSE = ', rmse_score)

                del reg
                del clf
                del lassoreg

            perf_r2.append(np.mean(split_r2))
            perf_rmse.append(np.mean(split_rmse))

        # Save idx dictionary
        np.save(os.path.join(data_root, 'neuron_idx_dict.npy'), lasso_idx)

        plt.plot(perf_r2, label='R2 Score')
        plt.plot(perf_rmse, label='RMSE')
        plt.axhline(y=0.7308948730140196, color='c', linestyle='--', label='RMSE baseline')
        plt.axhline(y=0.0414382636021091, color='m', linestyle='--', label='R2 baseline')
        plt.legend()
        plt.xlabel('Layers')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,-0.1,1))
        model = features_name.split('_')[0]
        plt.title(model+' neuron '+str(neuron))
        plt.savefig(os.path.join(data_root, 'figures',  model+'_LASSO_RIDGE_flat_neuron_'+str(neuron)+'.png'))

        plt.close()


def submission(data_root, features_name, method, alpha=None, layer=6, t=99):
    # Load Responses
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index
    target = df.values #(551x16) stim x avg sqrt of spikes
    target = target[:-1,:] #Remove last image from data and target - wasn't shown to more than 3 neurons, no info

    ## Load features
    net_features = np.load(os.path.join(data_root, features_name)).item()# load features

    neuron_choices = {
    0: 7,
    1: 6,
    2: 4,
    3: 4,
    4: 11,
    5: 7,
    6: 3,
    7: 7,
    8: 8,
    9: 9,
    10: 1,
    11: 6,
    12: 9,
    13: 5,
    14: 5,
    15: 7,
    16: 6,
    17: 1
    }

    # Fit on test images
    preds = np.empty((50,18))
    for neuron in range(target.shape[1]):
        l_choice = neuron_choices[neuron]

        lasso_idx = np.load(os.path.join(data_root, 'neuron_'+str(neuron)+'_layer_'+str(l_choice)+'.npy'))
        #REF: lasso_idx = neuron_idx[neuron][l_choice] # get indices for features

        ####### LASSO -> RIDGE ##########
        # Test features
        test_features = net_features[l_choice][:50]
        test_features = test_features.reshape(test_features.shape[0], -1)
        test_features = test_features[:, lasso_idx]
        train_features = net_features[l_choice][50:]
        train_features = train_features.reshape(train_features.shape[0], -1)
        train_features = train_features[:, lasso_idx]

        rec = df.iloc[:-1, neuron].dropna()
        # Train linear model on these pixels
        # Grid search on alpha
        parameters = {'alpha':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]}
        reg = Ridge()
        clf = GridSearchCV(reg, parameters, cv=5)
        clf.fit(train_features[:len(rec)], rec)
        print('Neuron: ', neuron, 'layer: ', l_choice, clf.best_params_)
        preds[:,neuron] = clf.predict(test_features)

    # Save csv file
    sdf = pd.DataFrame(preds)
    sdf.columns = df.columns#replace the columns with the correct cell ids from training data
    sdf.index.name = 'Id'
    sdf.to_csv(os.path.join(data_root, 'my_sub_'+str(t)+'.csv')) # save to csv
    sdf.head()

if __name__ == '__main__':
    data_root = '/home/jlg/michele/data/uwndc19'

    predict(data_root, 'alexnet_features.npy')

    submission(data_root, 'alexnet_features.npy', 'lassoridge', t=4)
