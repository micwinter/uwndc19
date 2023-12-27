"""
Pytorch model which takes stimulus videos that each neuron saw, pushes
them through VGG16 to extract features, and then learns a spatial mask
on those extracted features. The spatial mask is then averaged to a
single pixel per channel. Finally, linear regression is done to predict
the spike rate from the vector of feature channels for each neuron.
*Not for a group of neurons.
"""


import os
import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from models import BaseModel



# (For reference)
# def r2(y,yh):
#     ymean = np.mean(y)
#     out = 1 - (np.sum((y-yh)**2)/np.sum((y-ymean)**2))
#     return out


def main(
        batch_size=32,
        layer=None,
        root='../../data/uwndc19/',
        ):
    global save_root, Tensor, file_

    now = datetime.datetime.now()

    save_root = os.path.join(root, now.strftime('%Y_%m_%d_%H_%M'))

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    neuron_loss = []
    for neuron in range(18):
        print('neuron %d' %(neuron))
        # Train separate model for each neuron
        data = load_data(root, neuron=neuron)
        trainset = TensorDataset(data['train_stim'], data['train_label'])
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valset = TensorDataset(data['val_stim'], data['val_label'])
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        # Settings
        input_dim =  64
        layer = layer
        model_name = 'vgg16'
        output_dim = 1
        num_val = data['val_label'].shape[0] # num val in train loop
        num_train = data['train_label'].shape[0]

        # Model
        model = BaseModel(input_dim, layer, output_dim)
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        model.cuda()

        # Logger
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        print('Saving to %s' % (save_root))
        logger = 'neuron'+str(neuron)+'_logger'+now.strftime('%Y_%m_%d_%H_%M')+'.txt'
        file_ = open(os.path.join(save_root,logger), 'w')

        # Write settings to file
        file_.write('neuron: %d\n' % (neuron))
        file_.write('model: %s\n' % (model_name))
        file_.write('layer: %d\n' % (layer))
        file_.write('num stim: %d\n' % (data['train_stim'].shape[0]+data['val_stim'].shape[0]))

        # Train
        print('Training...')
        print('num train: %5d | num train val: %5d' % (num_train,num_val))
        file_.write('num train: %5d | num train val: %5d\n' % (num_train,num_val))

        train_losses = []
        train_r2s = []
        losses = []
        r2s = []
        corrs = []

        running_loss = np.inf

        for epoch in range(0, 2000):
            # Split data into batches
            # Random permutation & then divy into batches
            train_loss, train_r2 = train(train_loader, model, loss_function, optimizer, epoch)

            with torch.no_grad():
                for stim, labels in val_loader:
                    preds = model(stim)
                    loss = torch.sqrt(loss_function(preds, labels.cpu()))
                    preds = preds.detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    loss = loss.cpu().detach().numpy().item()
                    r2 = r2_score(preds, labels)
                    corr = np.corrcoef(np.squeeze(preds), labels)[0][1]

            if np.mod(epoch,500) == 0:
                print('Epoch: %5d | TRAINLOSS: %8.2f | LOSS: %8.2f | CORR: %8.2f | R2: %8.2f' % (epoch,train_loss,loss,corr,r2))
            file_.write('Epoch: %5d | TRAINLOSS: %8.2f | LOSS: %8.2f | CORR: %8.2f | R2: %8.2f\n' % (epoch,train_loss,loss,corr,r2))

            
            train_losses.append(train_loss)
            train_r2s.append(train_r2)
            losses.append(loss)
            corrs.append(corr)
            r2s.append(r2)

            if np.remainder(epoch,100) == 0:
                # Save mask
                save_mask = model.raw_mask.cpu().detach().numpy()
                np.save(os.path.join(save_root, 'neuron'+str(neuron)+'raw_mask_e'+str(epoch)+'.npy'), save_mask)


            # Save/replace checkpoint if loss decreases
            if running_loss > loss:
                running_loss = loss
                save_checkpoint(
                        neuron,
                    {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                    'r2': r2,
                    'corr': corr,
                    'save_root': save_root,
                    'neuron': neuron
                    })
                # best mask save
                ave_mask = model.raw_mask.cpu().detach().numpy()
                np.save(os.path.join(save_root, 'neuron'+str(neuron)+'best_raw_mask.npy'), save_mask)

        # Save numpy arrays & plot performace over time
        np.save(os.path.join(save_root, 'neuron'+str(neuron)+'train_losses.npy'), train_losses)
        np.save(os.path.join(save_root, 'neuron'+str(neuron)+'train_r2s.py'), train_r2s)
        np.save(os.path.join(save_root, 'neuron'+str(neuron)+'losses.npy'), losses)
        np.save(os.path.join(save_root, 'neuron'+str(neuron)+'r2s.npy'),r2s)
        np.save(os.path.join(save_root, 'neuron'+str(neuron)+'corrs.npy'),corrs)

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(train_losses, label='train loss', color='b')
        plt.plot(losses, label='loss', color='r')
        plt.ylim(0,1.2)
        plt.legend()
        plt.subplot(212)
        plt.plot(train_r2s, label='train r2', color='g')
        plt.plot(corrs, label='corrcoef',color='b')
        plt.plot(r2s, label='r2', color='y')
        plt.ylim(-1,1)
        plt.xlabel('Epoch')
        plt.legend()
        fig.suptitle('Performance')
        plt.savefig(os.path.join(save_root,'neuron'+str(neuron)+'plot_perf.png'))
        plt.close()

        neuron_loss.append(loss)


        file_.close()

        # Clear variables
        del model
        del loss_function
        del optimizer
        del train_losses
        del train_r2s
        del losses
        del corrs
        del r2s
        del trainset
        del train_loader
        del valset
        del val_loader

    plt.plot(range(0,18),neuron_loss)
    plt.ylim(0,1.2)
    plt.xlabel('Neuron')
    plt.xticks(np.arange(0,18, step=1))
    plt.title('Best Performance Per Neuron Layer %d' %(layer))
    plt.savefig(os.path.join(save_root,'all_neurons_perf_l'+str(layer)+'.png'))
    plt.close()

    # Plot mask for each neuron
    fig, axs = plt.subplots(3,6)
    fig.subplots_adjust(hspace = .5, wspace=.1)

    axs = axs.ravel()
    for neuron in range(18):
        curr_mask = np.load(os.path.join(save_root, 'neuron'+str(neuron)+'best_raw_mask.npy'))
        im = axs[neuron].imshow(np.squeeze(curr_mask))
        axs[neuron].axis('off')
        axs[neuron].set_title('#'+str(neuron))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('Mask per Neuron Layer %d' %(layer))
    plt.savefig(os.path.join(save_root, 'best_mask_per_neuron_l'+str(layer)+'.png'))
    plt.close()


def train(
        train_loader,
        model,
        loss_function,
        optimizer,
        epoch,
        ):
    """
    Train model for one epoch and evaluate on a subset of train set
    """
    model.zero_grad() # clear gradients

    train_loss = []
    train_r2 = []

    for stim, labels in train_loader: 
        preds = model(stim)
        loss = torch.sqrt(loss_function(preds, labels.cpu()))

        loss.backward()
        optimizer.step()

        train_loss.append(loss.cpu().detach().numpy())
        preds = preds.detach().numpy()
        labels = labels.cpu().detach().numpy()
        train_r2.append(r2_score(preds, labels))

    train_loss = np.mean(train_loss)
    train_r2 = np.mean(train_r2)
    return train_loss, train_r2

def save_checkpoint(neuron, state):
    torch.save(state, os.path.join(save_root, 'neuron'+str(neuron)+'best_checkpoint.t7'))


def load_data(data_root, mode='train', neuron=0):
    # Load stimuli
    stim = np.load(os.path.join(data_root,'stim.npy'))
    stim = stim[:-1] # Remove last image, not helpful
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    stim = stim.transpose(0,3,1,2)
    stim = (stim - mean[:, np.newaxis, np.newaxis] ) / std[:, np.newaxis, np.newaxis]
    stim = stim[50:,:] # only training images

    # Load labels
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index
    label = df.values #(551x18) stim x avg sqrt of spikes
    label = label[:-1,:] #Remove last image from data and target - wasn't shown to more than 3 neurons, no info
    label = label[:,neuron] # only neuron labels
    label = label[~np.isnan(label)]
    label = (label - np.mean(label))/np.std(label) # z normalize

    # Get only stim neuron has seen
    stim = stim[:label.shape[0],:]

    # Split into train and test set
    # 2/3, 1/3 split during training
    rand_indices = np.random.permutation(np.arange(0, len(label)))
    rand_indices = np.array_split(rand_indices,3)
    train_rand = np.concatenate((rand_indices[0],rand_indices[1]))
    val_rand = rand_indices[2]

    data = {}
    data['train_stim'] = Tensor(stim[train_rand])
    data['train_label'] = Tensor(label[train_rand])
    data['val_stim'] = Tensor(stim[val_rand])
    data['val_label'] = Tensor(label[val_rand])
        
    return data


def visualize_mask(group0, group1, root='/home/jlg/michele/gbox/data/uwndc19/', mode=None, layer=None, epoch=None, ident=None):
    if mode == 'numpy':
        group0_mask = np.load(os.path.join(root,group0, 'raw_mask_e'+str(epoch)+'.npy'))
        group1_mask = np.load(os.path.join(root,group1, 'raw_mask_e'+str(epoch)+'.npy'))
        all_masks = np.concatenate((group0_mask[:2,:],group1_mask,group0_mask[2:,:]),axis=0)

    # Plot mask for each neuron
    fig, axs = plt.subplots(3,6)
    fig.subplots_adjust(hspace = .5, wspace=.1)

    axs = axs.ravel()
    for neuron in range(18):
        curr_mask = np.squeeze(all_masks[neuron,:])
        #vmax = np.max(curr_mask)
        im = axs[neuron].imshow(curr_mask)
        axs[neuron].axis('off')
        axs[neuron].set_title('#'+str(neuron))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('Mask per Neuron')
    plt.savefig('/home/jlg/michele/gbox/data/uwndc19/mask_per_neuron_e'+str(epoch)+'_l'+str(layer)+'_'+ident+'.png')
    plt.close()

def predict(group0, group1):
    
    stim = np.load('/home/jlg/michele/gbox/data/uwndc19/stim.npy')
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    stim = stim.transpose(0,3,1,2)
    stim = (stim - mean[:, np.newaxis, np.newaxis] ) / std[:, np.newaxis, np.newaxis]
    stim = torch.cuda.FloatTensor(stim[:50,:]) # only testing images
    
    # Settings ------- Group0
    input_dim =  64
    alexnet_layer = 7
    output_dim = 11

    # Model
    model = BaseModel(input_dim, alexnet_layer, output_dim)
    model = nn.DataParallel(model)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    model.cuda()
    
    # Load models
    print('Loading checkpoint %s' % (group0))
    checkpoint = torch.load(group0)
    loss = checkpoint['loss']
    r2 = checkpoint['r2']
    corr = checkpoint['corr']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    with torch.no_grad():
        group0_preds = model(stim)
    group0_preds = group0_preds.cpu().detach().numpy()

    # Setting --------- Group1
    output_dim = 7

    # Model
    model = BaseModel(input_dim, alexnet_layer, output_dim)
    model = nn.DataParallel(model)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    model.cuda()

    # Load models
    print('Loading checkpoint %s' % (group1))
    checkpoint = torch.load(group1)
    loss = checkpoint['loss']
    r2 = checkpoint['r2']
    corr = checkpoint['corr']
    best_checkpoint.t7model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    with torch.no_grad():
        group1_preds = model(stim)
    group1_preds = group1_preds.cpu().detach().numpy()
    predictions = np.concatenate((group0_preds[:,:2],group1_preds,group0_preds[:,2:]),axis=1)
    
    df = pd.read_csv('/home/jlg/michele/gbox/data/uwndc19/train.csv')
    df.index = df.Id #turn the first col to index
    df = df.iloc[:,1:] #get rid of first col that is now index

    sub = pd.DataFrame(predictions)
    sub.columns = df.columns#replace the columns with the correct cell ids from training data
    sub.index.name = 'Id'
    sub.to_csv('/home/jlg/michele/gbox/data/uwndc19/my_sub_6.csv')#save to csv
    sub.head()


 



if __name__ == '__main__':
    for layer in [5,7,10,12,14,17]:
        main(layer=layer)    

#main(layer=2) # 2,5,7,10,12,14,17
    #visualize_mask('2019_03_01_15_49', '2019_03_01_15_57', mode='numpy', layer=5, epoch=200, ident='batch32')
    #visualize_mask('/home/jlg/michele/gbox/data/uwndc19/2019_02_28_15_30/best_checkpoint.t7','/home/jlg/michele/gbox/data/uwndc19/2019_02_28_15_44/best_checkpoint.t7')
    #predict('/home/jlg/michele/gbox/data/uwndc19/2019_02_26_15_30/mask_checkpoint_e611.t7','/home/jlg/michele/gbox/data/uwndc19/2019_02_26_15_39/mask_checkpoint_e89.t7')
    #predict('/home/jlg/michele/gbox/data/uwndc19/2019_02_26_14_15/mask_checkpoint_e685.t7',
    #'/home/jlg/michele/gbox/data/uwndc19/2019_02_26_14_20/mask_checkpoint_e380.t7')
