import torch
from torchvision import models, transforms
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

def rmse(x, y):
    return np.sqrt(np.mean((x-y) ** 2))


# TODO: btw all the models you make with convnet features, you'll be able to modify them by changing the size of the input image. That will make the network features apply to different scales in the image. Some may be better than others for certain neurons. (but leave that as a last step)

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
    # model = models.alexnet(pretrained=True)
    model = models.vgg16(pretrained=True)

    # Run data through network
    ext_feat = {}
    for layer in range(len(model.features)):
        model_ft = model.features[:layer]
        features = model_ft(data)
        features = features
        ext_feat[layer] = features.detach().numpy()
        print('layer ', layer, ' of ', len(model.features), ' complete')
    save_name = 'vgg16_features.npy'
    print('Saving file...')
    np.save(os.path.join(data_root, save_name), ext_feat)

def predict(data_root, features_name, method, alpha=None, subset=False):
    # Load Responses
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index
    target = df.values #(551x16) stim x avg sqrt of spikes
    target = target[:-1,:] #Remove last image from data and target - wasn't shown to more than 3 neurons, no info
    # (TOGGLE) Dataset with all ims
    if subset:
        target = np.concatenate([target[:,:2], target[:,9:]],1)
        import ipdb; ipdb.set_trace()
        features = features[50:]
    else:
        target = target[:,2:8]
        features = features[50:200]

    # Load features
    features = np.load(os.path.join(data_root, features_name)).item()# load features

    # Split data into different train/val sets
    rs = ShuffleSplit(n_splits=5, test_size=0.1)
    splits = [(train , test) for train, test in rs.split(np.arange(features[0].shape[0]))]

    # Fit
    perf_r2 = []
    perf_rmse = []
    for layer in features.keys():
        split_r2 = []
        split_rmse = []
        for idx, split in enumerate(splits):
            # print('Split ', idx)
            if method == 'linear':
                reg = LinearRegression().fit(features[layer][split[0]], target[split[0]]) # fit for images neurons have seen
                preds = reg.predict(features[layer][split[1]])
                r2_score = reg.score(features[layer][split[1]], target[split[1]])
                rmse_score = rmse(target[split[1]], preds)
            elif method == 'ridge':
                # import ipdb; ipdb.set_trace()
                clf = Ridge(alpha)
                clf.fit(features[layer][split[0]+50], target[split[0]])
                preds = clf.predict(features[layer][split[1]])
                r2_score = clf.score(features[layer][split[1]], target[split[1]])
                rmse_score = rmse(target[split[1]], preds)
            # save scores for split
            split_r2.append(r2_score)
            split_rmse.append(rmse_score)
            # print('R2 Score = ', r2_score)
            # print('RMSE = ', rmse_score)
        perf_r2.append(np.mean(split_r2))
        perf_rmse.append(np.mean(split_rmse))

    # pring('')

    plt.plot(perf_r2, label='R2 Score')
    plt.plot(perf_rmse, label='RMSE')
    plt.legend()
    plt.xlabel('Layers')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1))
    model = features_name.split('_')[0]
    plt.title(model)
    if method == 'ridge':
        plt.savefig(os.path.join(data_root, 'figures',  model+'_ridge_reg_alpha'+str(alpha)+'.png'))
    else:
        plt.savefig(os.path.join(data_root, 'figures', model+'_linear_reg.png'))
    plt.close()


if __name__ == '__main__':
    data_root = '/Users/minter/Dropbox/uwndc19_data/'
    get_features(load_data(data_root))
    # predict(data_root, 'alexnet_features.npy', 'linear', subset=True)
    # alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    # for alpha in alphas:
    #   predict(data_root, 'alexnet_features.npy', 'ridge', alpha=alpha)
