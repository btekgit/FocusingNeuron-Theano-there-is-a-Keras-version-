#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:04:59 2018

@author: btek
"""
import os
import numpy as np
import _pickle as cPickle
from urllib.request import urlretrieve
from sklearn.datasets.samples_generator import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def to_categorical(ar, num_classes=None):
    if not num_classes:
        num_classes= np.unique(ar)
    n= ar.shape[0]
    br = np.zeros((n,num_classes),dtype=type(ar))
    br[np.arange(n),ar]=1
    
    return br

def standarize_columns(trn, val=None, tst=None):
    
    scaler = MinMaxScaler()
    scaler.fit(trn)
    trn = scaler.transform(trn)
    if val is not None:
        val = scaler.transform(val)  
    if tst is not None:
        tst = scaler.transform(tst)
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate((trn,val),axis=0))
    trn = scaler.transform(trn)
    if val is not None:
        val = scaler.transform(val)  
    if tst is not None:
        tst = scaler.transform(tst)
    
    
    return trn,val,tst


def standarize_whole(trn, val=None, tst=None):
 
    mn_trn = np.mean(trn)
    std_trn = np.std(trn)
    trn -= mn_trn
    trn /= std_trn
    if val is not None:
        val -= mn_trn
        val /= std_trn
    if tst is not None:
        tst -= mn_trn
        tst /= std_trn
    
    return trn, val, tst


def standarize_image_025(trn, val=None, tst=None):

    K = 4.0 # 2.0 is very good with MNIST 99.20-99.19
    M = 256.0
    trn /= M
    trn *=K
    trn -= K/2.0
    if val is not None:
        val /= M
        val *= K
        val -= K/2
    if tst is not None:
        tst /= M
        tst *= K
        tst -= K/2
    
    return trn, val, tst

def reshape_and_standardize(trn, tst, standardize_by_stats=False,columns=True, verbose=False):
    if trn.ndim==4:
        n_s, w, h, n_c = np.shape(trn)
    elif trn.ndim==3:
        n_s, w, h = np.shape(trn)
        n_c = 1
    else:
        w, h = np.shape(trn)
        n_c = 1
        n_s = 1
    
    if verbose:
        print_x_i_mean_variance(trn)
        print_x_i_mean_variance(tst)
        
    trn= np.reshape(trn, [-1, n_c * w * h]).astype('float32')
    tst= np.reshape(tst, [-1, n_c * w * h]).astype('float32')
   
    if standardize_by_stats:
        if (columns):
            trn, tst,_ = standarize_columns(trn, tst)# this normalizes each data column
        else:
            trn, tst,_ = standarize_whole(trn,tst) # this one is not really a normalization
            
    else:
        # input is image 
        # this does not take mean and variance into account
        # but works good with images. 
        trn, tst,_ = standarize_image_025(trn, tst)
        
    if verbose:
        print_x_i_mean_variance(trn)
        print_x_i_mean_variance(tst)
    trn = np.reshape(trn, [-1, n_c, w, h]).astype('float32')
    tst = np.reshape(tst, [-1, n_c, w, h]).astype('float32')
    

    return trn,tst
    
def print_x_i_mean_variance(X): # incomplete code
    # assume 4-d array came . N,1,w,h
    print("Data shape:",X.shape)
    #Xcol = np.reshape(X,(-1,X.shape[-1]*X.shape[-2]))
    mns =np.mean(X,axis=0)
    vrs = np.var(X,axis=0)
    print("means mx,mean, min:",np.max(mns), np.mean(mns), np.min(mns))
    print("vars mx,meann, min:",np.max(vrs), np.mean(vrs), np.min(vrs))

#def x_i_zero_mean_unit_variance(X1, X2, verbose=False):   
#    
#    colmns= np.mean(X1,axis=0)
#    colvrs= np.std(X1,axis=0)
#    colvrs[colvrs==0]=np.float32(1.0)
#    X1 = (X1-colmns)/(colvrs+1E-10)
#    X2 = (X2-colmns)/(colvrs+1E-10)
#    if X1.ndim==4:
#        X1 = np.reshape(X,(-1,n_c,w,h))
#    if X1.ndim==3:
#        X1 = np.reshape(X,(-1,w,h))
#    return X
 
def load_dataset_cifar10(folder="", verbose=False):
    data = np.load(folder+"cifar_10.npz")
    print("Dataset Loaded")
    X_train, y_train = data['x_train'], data['y_train']
    
    X_test, y_test = data['x_test'], data['y_test']

    X_train, X_test  = reshape_and_standardize(X_train, X_test,
                                               standardize_by_stats=False,
                                               verbose=True)
    
    print("Data shape:",X_train.shape)
    
    random_i = np.random.permutation(X_train.shape[0])
    var_len=10000
    X_train, X_valid = X_train[random_i[:-var_len]], X_train[random_i[-var_len:]]
    y_train, y_valid = y_train[random_i[:-var_len]], y_train[random_i[-var_len:]]
    print("Train shape:",X_train.shape)
    print("Val shape:",X_valid.shape)
    plt_figure=False
    if plt_figure:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(y_train)
        plt.hist(y_valid)
        plt.show()
    
    y_train = np.squeeze(y_train).astype('int32')
    y_valid = np.squeeze(y_valid).astype('int32')
    y_test = np.squeeze(y_test).astype('int32')
    print("Test shape:",X_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_dataset_fashion(folder=""):
    data = np.load(folder+"fashion.npz")
    print("Dataset Loaded")
    X_train, y_train = data['x_train'], data['y_train']
    
    X_test, y_test = data['x_test'], data['y_test']

    X_train, X_test  = reshape_and_standardize(X_train, X_test,standardize_by_stats=False)
    
    print("Data shape:",X_train.shape)
    
    random_i = np.random.permutation(X_train.shape[0])
    var_len=10000
    X_train, X_valid = X_train[random_i[:-var_len]], X_train[random_i[-var_len:]]
    y_train, y_valid = y_train[random_i[:-var_len]], y_train[random_i[-var_len:]]
    print("Train shape:",X_train.shape)
    print("Val shape:",X_valid.shape)
    plt_figure=False
    if plt_figure:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(y_train)
        plt.hist(y_valid)
        plt.show()
    
    y_train = np.squeeze(y_train).astype('int32')
    y_valid = np.squeeze(y_valid).astype('int32')
    y_test = np.squeeze(y_test).astype('int32')
    print("Test shape:",X_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
def load_dataset_mnist_cluttered(folder=""):
    data = np.load(folder+"mnist_cluttered_60x60_6distortions.npz")
    
    X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
    X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
    X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)
    print("Dataset Loaded")
    # reshape for convolutions
    
    DIM = 60
    print("Train shape:",X_train.shape)
    print("Val shape:",X_valid.shape)
    # -------------------------------------------------------------------------
    # DO NOT standardize MNIST_CLUTTERED, it is standard
    #X_train,X_valid,X_test = standarize_whole(X_train,val=X_valid,tst=X_test)
    #--------------------------------------------------------------------------

    X_train = X_train.reshape((X_train.shape[0], 1, DIM, DIM))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, DIM, DIM))
    X_test = X_test.reshape((X_test.shape[0], 1, DIM, DIM))
    print("Dataset reshaped")    

    return X_train, y_train.astype('int32'), X_valid, y_valid.astype('int32'), X_test, y_test.astype('int32')

def load_dataset_mnist(folder=""):
    # We first define a download function, supporting both Python 2 and 3.
    
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data
        
    fname = folder+"mnist.npz"
    if not os.path.exists(fname):
        download(fname, folder=folder)
    else:
        data = np.load(fname)
        print("Dataset Loaded")
    
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    
    #X_t = np.concatenate((X_train,X_test),axis=0)
    #print_x_i_mean_variance(X_t)
    #X_t=x_i_zero_mean_unit_variance(X_t)
    #print_x_i_mean_variance(X_t)
    #X_train, X_test =X_t[:-10000,:,:], X_t[-10000:,:,:]
    # do not standardize data with mean variance of columns as it blows the joint image 
    X_train, X_test  = reshape_and_standardize(X_train, X_test,standardize_by_stats=False, columns=True)
    
    
    X_train, X_valid = X_train[:-10000], X_train[-10000:]
    y_train, y_valid = y_train[:-10000], y_train[-10000:]
    pltFigures=False
    if pltFigures:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(y_train)
        plt.hist(y_valid)
        plt.show()
    
    y_train = np.squeeze(y_train).astype('int32')
    y_valid = np.squeeze(y_valid).astype('int32')
    y_test = np.squeeze(y_test).astype('int32')
    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test



def load_dataset_boston(folder=""):
    data = np.load(folder+"boston_housing.npz")
    print("Dataset Loaded")
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    
    X_train, X_test,_  = standarize_columns(X_train, X_test)
    X_train, X_valid = X_train[:-50].astype('float32'), X_train[-50:].astype('float32')
    y_train, y_valid = y_train[:-50].astype('float32'), y_train[-50:].astype('float32')
    X_test, y_test = X_test.astype('float32'), y_test.astype('float32')
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
    
def load_dataset_reuters(folder=""):
    data = np.load(folder+"reuters_preprocessed.npz")
    print("Dataset Loaded")
    X_train, y_train = data['X_train'], data['Y_train']
    tr_len = X_train.shape[0]
    print("Data shape:",X_train.shape)
    ix = np.arange(tr_len)
    np.random.shuffle(ix)
    #print("ix: ",ix.shape)
    X_train = X_train[ix]
    y_train = y_train[ix]
    X_train, X_valid = X_train[:-982].astype('float32'), X_train[-982:].astype('float32')
    y_train, y_valid = y_train[:-982].astype('float32'), y_train[-982:].astype('float32')
    X_test, y_test = data['X_test'].astype('float32'), data['Y_test'].astype('float32')
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def add_noisy_dims(x, noise_dims, noise_scale, noise_pattern):
    """
    Given the data matrix x, it adds noise_dims redundant random columns
    

    Parameters
    ----------
    x : data matrix, columns=features
    noise_dims : number of random redundant dims to add
    noise_scale: scale multiplier for the noise, not important if the 
    data will be standardized
    noise_pattern: select the location of noise columns with respect to data
    

    Returns
    -------
    Noise added dataset
    """
    
    n_features = x.shape[1]
    if noise_pattern=="sides":
        noise_x = np.random.rand(x.shape[0], noise_dims) * noise_scale
        noise_y = np.random.rand(x.shape[0], noise_dims) * noise_scale
        x = np.concatenate((noise_x, x, noise_y), axis=1)
        #print("noise added to sides")
    elif noise_pattern=="left":
        
        noise_x = np.random.rand(x.shape[0], noise_dims) * noise_scale
        print("noise scale ", noise_scale, "mx: ",np.max(noise_x))
        x = np.concatenate((noise_x, x), axis=1)
        #print("noise added to left")
    elif noise_pattern=="right":
        noise_x = np.random.rand(x.shape[0], noise_dims) * noise_scale
        x = np.concatenate((x, noise_x), axis=1)
        #print("noise added to right")
    elif noise_pattern=="interleaved":
        noise_x = np.random.rand(x.shape[0], noise_dims) * noise_scale
        noise_y = np.random.rand(x.shape[0], noise_dims) * noise_scale
        noise_z = np.random.rand(x.shape[0], noise_dims) * noise_scale
        half_x = int(n_features/2)
        #print(half_x)
        ts = np.concatenate((noise_x, x[:,0:half_x], noise_y), axis=1)
        x = np.concatenate((ts, x[:,half_x:n_features], noise_z), axis=1)
    
    elif noise_pattern=="left-constant":
        noise_x = np.ones((x.shape[0], noise_dims)) * noise_scale
        x = np.concatenate((noise_x, x), axis=1)
        
    elif noise_pattern=="right-constant":
        noise_x = np.ones((x.shape[0], noise_dims)) * noise_scale
        x = np.concatenate((noise_x, x), axis=1) 
        #print("noise added interleaved to both sides and center")
        
    return x
        
def load_blob(classes=3, features=10, samples=10, random_state=0, 
              noise_dims=0, noise_scale=1.0, noise_pattern=None, clusters=1):
    """
    Creates a Gaussian blob classification dataset of given parameters   

    Parameters
    ----------
    classes : num classes
    features :  num columns
    samples:  num rows
    random_state:  random seed
    noise_dims: parameters for adding redundant dims
    noise_scale: //
    noise_pattern: //
    clusters: num blobs for each class
    Returns
    -------
    Noise added standardized/normalized datasets dict
    separating train test samples and labels
    """
    samples = int(samples / .7)  # 30% for test
    xs, ys = make_classification(n_samples=samples, n_features=features, 
                                 n_informative=features, n_redundant=0, 
                                 n_classes=classes, n_clusters_per_class=clusters)
    ys = np.float32(ys)
    
    xs = add_noisy_dims(xs, noise_dims, noise_scale, noise_pattern)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(xs, ys.squeeze(), test_size=0.3, random_state=42)
    
    # now normalize all
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return dict(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    num_examples_train=X_train.shape[0],
    num_examples_test=X_test.shape[0])
    
    
def stich_datasets(X1, X2, y1, y2):
    # this method is not complete 
    # thought this dataset can model multiproblem domains.
    Xres = np.concatenate((X1,X2),axis=3)
    # how to configure output labels. multi-dim or joint?
    Yres = np.zeros(shape=(y1.shape(0),y1.shape(1)+y1.shape(2)))