import torch
from torchvision import models, transforms
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def rmse(x, y):
    return np.sqrt(np.mean((x-y) ** 2))


# TODO: btw all the models you make with convnet features, you'll be able to modify them by changing the size of the input image. That will make the network features apply to different scales in the image. Some may be better than others for certain neurons. (but leave that as a last step)

#TODO: add baseline R2 and RMSE from RGB average

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
    model = models.alexnet(pretrained=True)
    # model = models.vgg16(pretrained=True)

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

def predict(data_root, features_name, method, alpha=None, subset=False):
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

    # #(TOGGLE) Dataset with all ims
    # if subset:
    #     target = np.concatenate([target[:,:2], target[:,9:]],1)
    #     # Split data into different train/val sets
    #     splits = [(train , test) for train, test in rs.split(np.arange(features[0][50:].shape[0]))]
    #
    # else:
    #     target = target[:,2:8]
    #     # Split data into different train/val sets
    #     splits = [(train , test) for train, test in rs.split(np.arange(features[0][50:200].shape[0]))]
    #
    #
    # # Fit
    # perf_r2 = []
    # perf_rmse = []
    # Color for plots
    # color=iter(cm.rainbow(np.linspace(0,1,len(features.keys()))))

    # for layer in features.keys():
    #     print('Layer: ', layer)
    # #(TOGGLE) Dataset with all ims
    #     if subset:
    #         features[layer] = features[layer][50:]
    #     else:
    #         features[layer] = features[layer][50:200]
    #
    #     split_r2 = []
    #     split_rmse = []
    #     split_scores = np.empty((len(splits), features[layer].shape[2]*features[layer].shape[3]))
    #
    #     for idx, split in enumerate(splits):
    #         print('Split: ', idx)
    #         # print('Split ', idx)
    #
    #         if method == 'linear':
    #             reg = LinearRegression()
    #         elif method == 'ridge':
    #             reg = Ridge(alpha)

            ############# AVG SPACE #####################
        #     # Average over space
        #     train_features = features[layer][split[0]].mean(-1).mean(-1)
        #     val_features = features[layer][split[1]].mean(-1).mean(-1)
        #
        #     # Fit regression & print score
        #     reg.fit(train_features, target[split[0]])
        #     preds = reg.predict(val_features)
        #     r2_score = reg.score(val_features, target[split[1]])
        #     rmse_score = rmse(target[split[1]], preds)
        #
        #     # save scores for split
        #     split_r2.append(r2_score)
        #     split_rmse.append(rmse_score)
        #     print('R2 Score = ', r2_score)
        #     print('RMSE = ', rmse_score)
        #
        # perf_r2.append(np.mean(split_r2))
        # perf_rmse.append(np.mean(split_rmse))
            ############################################

            ########### PER PIXEL ######################
    #
    # neuron_dict = {}
    #
    # # Iterate over 18 neurons
    # for neuron in range(target.shape[1]):
    #     neuron_dict[neuron] = {}
    #     print('Neuron: ', neuron)
    #
    #     for layer in net_features.keys():
    #         print('Layer: ', layer)
    #         neuron_dict[neuron][layer] = {}
    #
    #         # Take images in train dataset
    #         features = net_features[layer][50:]
    #
    #         # Split data into different train/val sets
    #         curr_target = target[:,neuron]
    #         curr_target = curr_target[~np.isnan(curr_target)]
    #         # if neuron == 1 and layer == 0:
    #         #     import ipdb; ipdb.set_trace()
    #         #     print('hi')
    #         splits = [(train , test) for train, test in rs.split(np.arange(curr_target.shape[0]))]
    #
    #         split_r2 = []
    #         split_rmse = []
    #         # Number of splits x number of pixels
    #         split_scores = np.empty((len(splits), features.shape[-2]*features.shape[-1]))
    #         split_rmse = np.empty((len(splits), features.shape[-2]*features.shape[-1]))
    #
    #         for idx, split in enumerate(splits):
    #             if method == 'linear':
    #                 reg = LinearRegression()
    #             elif method == 'ridge':
    #                 reg = Ridge(alpha)
    #             # Regress over all pixels
    #             train_features = features[split[0]].reshape(features[split[0]].shape[0], features[split[0]].shape[1], -1)
    #             val_features = features[split[1]].reshape(features[split[1]].shape[0], features[split[1]].shape[1], -1)
    #
    #             # Iterate over all pixels
    #             for pixel in range(train_features.shape[2]):
    #                 t_feat = train_features[:, :, pixel]
    #                 v_feat = val_features[:, :, pixel]
    #                 reg.fit(t_feat, target[split[0], neuron])
    #                 preds = reg.predict(v_feat)
    #                 split_scores[idx, pixel] = reg.score(v_feat, target[split[1], neuron])
    #                 split_rmse[idx, pixel] = rmse(target[split[1], neuron], preds)
    #
    #         # Average over splits to get per pixel responses for neuron
    #         neuron_dict[neuron][layer]['R2'] = split_scores.mean(0)
    #         neuron_dict[neuron][layer]['RMSE'] = split_rmse.mean(0)
    #
    #         # Plot neuron scores for each pixel for each layer
    #         plt.plot(split_scores.mean(0), label='R2 Score '+str(layer))
    #         plt.plot(split_rmse.mean(0), label='RMSE '+str(layer))
    #         plt.legend()
    #         plt.ylim(-0.1,1)
    #         plt.xlabel('Pixel')
    #         # x1,x2,y1,y2 = plt.axis()
    #         # plt.axis((x1,x2,0,np.max(split_scores.mean(0))))
    #         model = features_name.split('_')[0]
    #         plt.title('Neuron '+str(neuron)+' layer '+str(layer)+' fit per pixel')
    #         if method == 'ridge':
    #             plt.savefig(os.path.join(data_root, 'figures',  model+'_ridge_reg_alpha'+str(alpha)+'_neuron_'+str(neuron)+' layer '+str(layer)+'_fitperpixel.png'))
    #         else:
    #             plt.savefig(os.path.join(data_root, 'figures', model+'_linear_reg_neuron_'+str(neuron)+' layer '+str(layer)+'_fitperpixel.png'))
    #         plt.close()
    #
    # # Save pixel dictionary
    # dict_name = model +'_ridge_reg_alpha'+str(alpha)+'_pixel_dictionary.npy'
    # np.save(os.path.join(data_root, dict_name), neuron_dict)
            ###########################################

            ######## USE PIXEL FOR FLATTENED ##########
            # Regress over all pixels

    # Load in pixel responses
    file = '/Users/minter/Dropbox/uwndc19_data/alexnet_ridge_reg_alpha1000.0_pixel_dictionary.npy'

    neuron_act = np.load(file).item()

    for layer in net_features.keys():
        print('Layer: ', layer)

        # Take images in train dataset
        features = net_features[layer][50:]

        for neuron in range(target.shape[1]):
            print('Neuron: ', neuron)

            # Get 90% best performing pixels
            pixels = neuron_act[neuron][layer]['R2']
            max_response = np.max(pixels)
            max_response = 0.9*max_response
            import ipdb; ipdb.set_trace()
            pixels = pixels[pixels>=max_response]

            train_features = features[layer][split[0]].reshape(features[layer][split[0]].shape[0], features[layer][split[0]].shape[1], -1)
            val_features = features[layer][split[1]].reshape(features[layer][split[1]].shape[0], features[layer][split[1]].shape[1], -1)

            for i in range(train_features.shape[2]):
                t_feat = train_features[:, :, i]
                v_feat = val_features[:, :, i]
                reg.fit(t_feat, target[split[0]])
                split_scores[idx, i] = reg.score(v_feat, target[split[1]])

            # Only take pixels with positive r2scores from flattented features
            scores = split_scores[idx, :]
            pos_scores = scores[scores>0]
            t_pos = train_features[:,:,scores>0].reshape(train_features.shape[0], train_features.shape[1]*pos_scores.shape[0]) # positive scoring pixels
            v_pos = val_features[:,:,scores>0].reshape(val_features.shape[0], val_features.shape[1]*pos_scores.shape[0])

            t_pos = features[layer][split[0]].reshape(features[layer][split[0]].shape[0], -1)
            v_pos = features[layer][split[1]].reshape(features[layer][split[1]].shape[0], -1)


            # Train linear model on these pixels
            reg = Ridge(alpha)
            reg.fit(t_pos, target[split[0]])
            preds = reg.predict(v_pos)
            r2_score = reg.score(v_pos, target[split[1]])
            rmse_score = rmse(target[split[1]], preds)

            # save scores for split
            split_r2.append(r2_score)
            split_rmse.append(rmse_score)
            # print('R2 Score = ', r2_score)
            # print('RMSE = ', rmse_score)

        perf_r2.append(np.mean(split_r2))
        perf_rmse.append(np.mean(split_rmse))

    plt.plot(perf_r2, label='R2 Score')
    plt.plot(perf_rmse, label='RMSE')
    plt.axhline(y=0.7308948730140196, color='c', linestyle='--', label='baseline')
    plt.axhline(y=0.0414382636021091, color='m', linestyle='--', label='R2 baseline')
    plt.legend()
    plt.xlabel('Layers')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1))
    model = features_name.split('_')[0]
    plt.title(model)
    if method == 'ridge':
        plt.savefig(os.path.join(data_root, 'figures',  model+'_ridge_reg_alpha'+str(alpha)+'_pixelwise_flat.png'))
    else:
        plt.savefig(os.path.join(data_root, 'figures', model+'_linear_reg_pixelwise.png'))
    plt.close()
            ###########################################




    # plt.plot(perf_r2, label='R2 Score')
    # plt.plot(perf_rmse, label='RMSE')
    # plt.axhline(y=0.7308948730140196, color='c', linestyle='--', label='RMSE baseline')
    # plt.axhline(y=0.0414382636021091, color='m', linestyle='--', label='R2 baseline')
    # plt.legend()
    # plt.xlabel('Layers')
    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((x1,x2,-0.1,1))
    # model = features_name.split('_')[0]
    # plt.title(model)
    # if method == 'ridge':
    #     plt.savefig(os.path.join(data_root, 'figures',  model+'_ridge_reg_alpha'+str(alpha)+'.png'))
    # else:
    #     plt.savefig(os.path.join(data_root, 'figures', model+'_linear_reg.png'))
    # plt.close()

def submission(data_root, features_name, method, alpha=None, layer=6, t=99):
    # Load Responses
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index
    target = df.values #(551x16) stim x avg sqrt of spikes
    target = target[:-1,:] #Remove last image from data and target - wasn't shown to more than 3 neurons, no info

    ## Load features
    features = np.load(os.path.join(data_root, features_name)).item()# load features

    # Test features
    test_features = features[layer][:50].mean(-1).mean(-1) # first 50 images
    train_features = features[layer][50:].mean(-1).mean(-1)

    # Fit on test images
    preds = np.empty((50,18))
    for unit in range(target.shape[1]):
        rec = df.iloc[:-1, unit].dropna()
        reg = Ridge(alpha)
        reg.fit(train_features[:len(rec)], rec)
        preds[:,unit] = reg.predict(test_features)


    # Save csv file
    sdf = pd.DataFrame(preds)
    sdf.columns = df.columns#replace the columns with the correct cell ids from training data
    sdf.index.name = 'Id'
    sdf.to_csv(os.path.join(data_root, 'my_sub_'+str(t)+'.csv'))#save to csv
    sdf.head()

if __name__ == '__main__':
    data_root = '/Users/minter/Dropbox/uwndc19_data/'#'/home/jlg/michele/data/uwndc19'#'/Users/minter/Dropbox/uwndc19_data/'
    # get_features(load_data(data_root))
    # predict(data_root, 'alexnet_features.npy', 'linear', subset=True)

    # alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    # for alpha in alphas:
    #     predict(data_root, 'alexnet_features.npy', 'ridge', alpha=alpha)

    predict(data_root, 'vgg16_features.npy', 'ridge', alpha=1e3, subset=True)

    # submission(data_root, 'alexnet_features.npy', 'ridge', alpha=1e3, layer=4, t=1)
