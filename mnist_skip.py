#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from __future__ import unicode_literals
"""
original code taken from lasagne mnist tutorial.

adapted on Thu Apr 19 18:36:12 2018

@author: btek
"""

import os
if 'THEANO_FLAGS' not in os.environ.keys():
    os.environ['THEANO_FLAGS']='device=cuda1, floatX=float32, gpuarray.preallocate=.1'
os.environ['MKL_THREADING_LAYER']='GNU'
import time
import lasagne
import theano
import theano.tensor as T
import numpy as np
theano.config.exception_verbosity = 'high'
import sys
from focusing import FocusedLayer1D, U_numeric
from lasagne_utils import sgdWithLrsClip,categorical_focal_loss,\
get_shared_by_pattern,sgdWithLrs,iterate_minibatches,set_params_value,\
print_param_stats, get_params_values_wkey, sgdWithWeightSupress


from data_utils import load_dataset_mnist, load_dataset_mnist_cluttered,\
load_dataset_fashion, load_dataset_cifar10


def build_custom_mlp(input_var=None, input_shape=(None,1,28,28), 
                     depth=2, width=800, drop_input=.2, drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    nonlin = lasagne.nonlinearities.rectify
    lin = lasagne.nonlinearities.linear
    ini = lasagne.init.HeUniform()
    softmax = lasagne.nonlinearities.softmax
    
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    
    #network = lin(lasagne.layers.BatchNormLayer(network))
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    
    # Hidden layers and dropout:
    
    net_layers=[]
    k=0
    net_layers.append(network)
    k+=1
    for lay in range(depth):
        l_hidden = lasagne.layers.DenseLayer(net_layers[k-1], num_units=width, 
                                             nonlinearity=lin, W=ini,
                                             name='dense-'+str(lay))
                
        net_layers.append(l_hidden)
        k+=1
        l_bn = lasagne.layers.NonlinearityLayer(
            lasagne.layers.BatchNormLayer(net_layers[k-1]), nonlinearity=nonlin)
        
        net_layers.append(l_bn)
        k+=1
        if drop_hidden:
            l_drp = lasagne.layers.dropout(net_layers[k-1], p=drop_hidden)
            net_layers.append(l_drp)
            k+=1
    # Output layer:
    
    l_out = lasagne.layers.DenseLayer(
            net_layers[k-1], num_units=10, W=ini,
            nonlinearity=softmax, name='dense'+str(depth+1))
   
    return l_out

def build_focused_mlp(input_var=None, input_shape=(None,1,28,28), 
                      depth=2, width=800, drop_input=.2,
                     drop_hidden=.5, update_mu=True, update_si=True,
                     init_mu='spread', initsigma=0.1, batch_norm=True,
                     skip_connections=True):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    nonlin = lasagne.nonlinearities.rectify
    lin = lasagne.nonlinearities.linear
    ini = lasagne.init.HeUniform()
    softmax = lasagne.nonlinearities.softmax
    #batchnorm = lasagne.layers.batch_norm
    
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                         input_var=input_var)
    
        
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    
    net_layers=[]
    k=0
    net_layers.append(network)
    k+=1
    for lay in range(depth):
        l_focused = FocusedLayer1D(
                net_layers[k-1], num_units=width, nonlinearity=lin, name='focus-'+str(lay),
                              trainMus=update_mu, 
                              trainSis=update_si, 
                              initMu=init_mu, 
                              W=lasagne.init.Constant(0.0), withWeights=True, 
                              bias=lasagne.init.Constant(0.0), 
                              initSigma=initsigma, 
                             scaler=1.0, weight_gain=1.0,
                             trainScaler=False, trainWs=True)
        
        if skip_connections:
            print("Merging", lasagne.layers.get_output_shape(l_focused))
            if k == 1:
                random_layers_ix  = 0
            else:
                random_layers_ix = np.random.randint(k-1)
            l_prev = net_layers[k-1]
            
            
            #if k > 1:
            #    l_prev  = lasagne.layers.FeaturePoolLayer(l_prev, pool_size=2, axis=1)
            l_prev = lasagne.layers.FlattenLayer(l_prev)
           

            l_focused = lasagne.layers.ConcatLayer((l_focused,l_prev), axis=1)
            print(" With ", lasagne.layers.get_output_shape(l_prev))
        
        net_layers.append(l_focused)
        k+=1
        if batch_norm:
            l_bn = lasagne.layers.NonlinearityLayer(
                    lasagne.layers.BatchNormLayer(net_layers[k-1]), nonlinearity=nonlin)
        else:
            l_bn = lasagne.layers.NonlinearityLayer(net_layers[k-1], nonlinearity=nonlin)
        
        net_layers.append(l_bn)
        k+=1
        if drop_hidden:
            l_drp = lasagne.layers.dropout(net_layers[k-1], p=drop_hidden)
            net_layers.append(l_drp)
            k+=1
    # Uncomment below if output layer will be also focusing:   
#    network = FocusedLayer1D(
#                net_layers[k-1], num_units=10, nonlinearity=lasagne.nonlinearities.softmax, 
#                              name='focus-'+str(depth+1),
#                              trainMus=update_mu, 
#                              trainSis=update_si, 
#                              initMu=init_mu, 
#                              W=lasagne.init.Constant(0.0), withWeights=True, 
#                              bias=lasagne.init.Constant(0.0), 
#                              initSigma=initsigma, 
#                             scaler=1.0, weight_gain=1.0,
#                             trainScaler=False, trainWs=True)
    
    l_out = lasagne.layers.DenseLayer(
            net_layers[k-1], num_units=10, W=ini,
            nonlinearity=softmax, name='dense'+str(depth+1))
   
    return l_out


def build_cnn_simple(input_var=None,input_shape=(None,1,28,28)):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.
    
    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def build_cnn_simple2(input_var,input_shape, oned=False, params_for_test={},
                      replace_dense_w_focus = False, num_f=32, fsize=(5,5), 
                      psize=(2,2), num_dense=256, nonlin=lasagne.nonlinearities.rectify):
    ''' As a third model, we'll create a CNN of two convolution + pooling stages
     and a hidden layer in front of the output layer.

    if oned=True, convolutions are 1D.
     if replace_dense_w_focus=True, dense layers are replaced by Focusing Layers
     default 
     drop out is 0.25, 
     filter size is fsize 5x5
     pool size psize(2,2)
     drop out prob drop_input=0.25
     num_f = 32
     params_for_test is a dictionary for receiving configuration specific params.
     like init sigma for focusing neuron. 
    '''
    drop_input=0.25
    convlayer = lasagne.layers.Conv2DLayer
    poollayer= lasagne.layers.MaxPool2DLayer
    lin =lasagne.nonlinearities.linear
    if "cnn_num_filters" in params_for_test.keys():
        num_f = params_for_test["cnn_num_filters"]    

    #lasagne.layers.BatchNormLayer(net_layers[k-1]), nonlinearity=nonlin)
    if oned:
        convlayer=lasagne.layers.Conv1DLayer
        fsize = fsize[0]
        psize = psize[0]
        num_dense=64
        poollayer= lasagne.layers.MaxPool1DLayer
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    
    
    #network = lasagne.layers.dropout(network, p=drop_input)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = convlayer(
            network, num_filters=num_f, filter_size=fsize,
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform())
    
     # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = convlayer(
            network, num_filters=num_f, filter_size=fsize,
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform())
    
    network = poollayer(network, pool_size=psize)

#    A fully-connected layer of 256 units with 50% dropout on its inputs:
    if replace_dense_w_focus:
       
##=============================================================================
#        network = lasagne.layers.DenseLayer(
#                 lasagne.layers.dropout(network, p=.5),
#                 num_units=num_dense,
#                 nonlinearity=lasagne.nonlinearities.rectify)
#         
##=============================================================================
#=============================================================================
        
        network = lasagne.layers.dropout(network, p=.5)
        network = FocusedLayer1D(network, num_units=num_dense, 
                                 nonlinearity=nonlin,
                                 name='focus-0',
                                 trainMus=True, 
                                 trainSis=True, 
                                 initMu='spread', 
                                 W=lasagne.init.Constant(0.0), withWeights=True, 
                                 bias=lasagne.init.Constant(0.0), 
                                 initSigma=0.15, 
                                 scaler=1.0, weight_gain=1.0,
                                 trainScaler=False, trainWs=True)
        
        #network = lasagne.layers.NonlinearityLayer(
        #        lasagne.layers.BatchNormLayer(network), nonlinearity=nonlin)
        network = lasagne.layers.NonlinearityLayer(network, nonlinearity=nonlin)
#=============================================================================
        network = lasagne.layers.dropout(network, p=.5)
        network = FocusedLayer1D(
                network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax, name='focus-1',
                              trainMus=True, 
                              trainSis=True, 
                              initMu='spread', 
                              W=lasagne.init.Constant(0.0), withWeights=True, 
                              bias=lasagne.init.Constant(0.0), 
                              initSigma=0.1, 
                              scaler=1.0, weight_gain=1.0,
                              trainScaler=False, trainWs=True)
    else:
            
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=num_dense,
                nonlinearity=lasagne.nonlinearities.rectify)
    
        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)

    return network

def build_fcnn_simple(input_var=None,input_shape=(None,1,28,28),
                      update_mu=True, update_si=True,
                     init_mu='spread', initsigma=0.1, batch_norm=True):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = FocusedLayer1D(
                network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax, 
                              name='focus-1',
                              trainMus=update_mu,
                              trainSis=update_si,
                              initMu=init_mu,
                              W=lasagne.init.Constant(0.0), withWeights=True,
                              bias=lasagne.init.Constant(0.0),
                              initSigma=initsigma,
                             scaler=1.0, weight_gain=1.0,
                             trainScaler=False, trainWs=True)
                

    return network



def build_network_model(model, input_var, input_shape, params_for_test, verbose=True):
     # Create neural network model (depending on first command line parameter)
    if verbose:
        print("Building model and compiling functions...")
    if model.startswith('mlp'):
        extraparams = model.split(':', 1)
        if len(extraparams)==1:
            network = build_custom_mlp(input_var,input_shape=input_shape, 
                                       depth=int(2), width=int(800))
        else:
            depth, width, drop_in, drop_hid = extraparams[1].split(',')
            network = build_custom_mlp(input_var,input_shape=input_shape, 
                                       depth=int(depth), width=int(width))
        
    elif model.startswith('focused_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        depth, width, drop_in, drop_hid = int(depth), int(width), float(drop_in), float(drop_hid)
        network = build_focused_mlp(input_var,input_shape=input_shape, depth=depth,
                                    width=width, drop_input=drop_in, drop_hidden=drop_hid,
                                    update_mu=True, update_si=True,
                                    init_mu='spread',initsigma=0.1,
                                    batch_norm=True)
        
    elif model.startswith('fixed_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        depth, width, drop_in, drop_hid = int(depth), int(width), float(drop_in), float(drop_hid)
        print(depth, width, drop_in, drop_hid)
        network = build_focused_mlp(input_var,input_shape=input_shape,
                                    depth=depth, width=width, drop_input=drop_in, 
                                    drop_hidden=drop_hid, init_mu='spread',
                                    update_mu=False, update_si=False,initsigma=.1)
        
    elif model == 'cnn' or model=='cnn1d' or model=='cnn2d':
        oned=False
        if model=='cnn1d': 
            oned=True
            
        network = build_cnn_simple2(input_var,input_shape=input_shape, oned=oned, 
                                    params_for_test=params_for_test)
        
    elif model.startswith('focused_cnn'):
        network = build_cnn_simple2(input_var,input_shape=input_shape, oned=False, 
                                    params_for_test=params_for_test, 
                                    replace_dense_w_focus= True)
        
    elif model.startswith('fixed_cnn'):
        #network = build_focused_cnn(input_var)
        network = build_fcnn_simple(input_var,input_shape=input_shape, 
                                    update_mu=False, update_si=False,initsigma=.10)
    else:
        print("Unrecognized model type %r." % model)
        return
    
    return network

def build_functions(network, input_var, target_var):
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
    #all_layers = lasagne.layers.get_all_layers(network)
    #l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.00001
    #loss = loss + l2_penalty
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    print("Params",params)
    print("Param count:", lasagne.layers.count_params(network))
    
    LR_rate = theano.shared(np.float32(0.0),name='lr_all')
    LR_MU = theano.shared(np.float32(0.0),name='lr_mu')
    LR_SI = theano.shared(np.float32(0.0),name='lr_si')
    LR_FW = theano.shared(np.float32(0.0),name='lr_fw')
    LR_params = [LR_rate, LR_MU, LR_SI, LR_FW]
        
    updates = sgdWithLrsClip(loss, params, learning_rate=LR_rate, 
                         mu_lr=LR_MU, si_lr=LR_SI, 
                         focused_w_lr=LR_FW, momentum=.90)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    
    train_fn = theano.function([input_var, target_var],
                               loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn, LR_params

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
    
def main(model='mlp', num_epochs=500, dataset='mnist', folder="", exp_start_time=None, verbose=False):
    # Load the dataset
    print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    params_for_test ={}
    
    data_root = '../datasets/image/'
    if not os.path.exists(data_root):
        print('This code searches for data in \n \'data_root\'='+data_root+'\n folder. '+
              'Data must be packed in .npz (keras packed) archive. E.g. mnist.npz, cifar10.npz')
        print('To test on different data download or provide npz files.')
        print('You can change \'data_root\' folder ')
        data_root =''
            
    if dataset == 'mnist':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_mnist(folder=data_root)
    elif dataset == 'mnist_cluttered':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_mnist_cluttered(folder=data_root)
    elif dataset == 'cifar10':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_cifar10(folder=data_root)
        params_for_test["cnn_num_filters"]=32
    elif dataset == 'fashion':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_fashion(folder=data_root)
    
    if verbose: 
        print("Data mean:",np.mean(X_train))
        print("Data max:",np.max(X_train))
        print("Data var:",np.var(X_train))
        print("Data shape:", np.shape(X_train))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')    
    input_shape= (None, X_train.shape[1],X_train.shape[2],X_train.shape[3])
    
    
    if model == 'cnn1d':
        input_var = T.tensor3('inputs')
        input_shape= (None, X_train.shape[1],X_train.shape[2]*X_train.shape[3])
        X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[2]*X_train.shape[3]))
        X_val = np.reshape(X_val,(X_val.shape[0],X_val.shape[1],X_val.shape[2]*X_val.shape[3]))
        X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2]*X_test.shape[3]))
    
    network = build_network_model(model, input_var, input_shape, params_for_test)
    
    train_fn, eval_fn, LR_params = build_functions(network, input_var, target_var)

   
    # Finally, launch the training loop.
    if verbose:
        print("Starting training...")
    # Prepare the lists to store performance
    val_acc_list =[]
    tst_acc_list =[] 
    val_err_list = []
    trn_err_list = []
    param_list = []
    
    # learning rates 
    lr_all = 0.1 # this affect all non-focus variables
    lr_all_decay = .9    
    lr_mu = 0.01  # focus center
    lr_mu_decay = 0.9
    lr_si = 0.001 # focus aperture
    lr_si_decay = 0.9
    lr_fw = 0.1 # focus weights
    lr_fw_decay = .9
    decay_epoch = 100
    
    # this is the learning rate decay function.
    decay_check = lambda x: x==decay_epoch
#   
    if dataset == 'fashion':
        lr_mu = 0.1 
        lr_si = 0.01
        lr_all = 0.0005
    
    if dataset =='cifar10':
        decay_epoch = 10
        lr_mu = 0.01
        lr_all = 0.01
        lr_fw = 0.01
        decay_epoch = 10
        decay_check = lambda x: x>decay_epoch and x%decay_epoch==1
        
     
    if model.find('cnn')>=0:
        lr_all = 0.01
        lr_all_decay = .9
        
        lr_mu = 0.01
        lr_mu_decay = 0.9
        lr_si = 0.001
        lr_si_decay = 0.9
        lr_fw = 0.01
        lr_fw_decay = .9
        decay_epoch = 40
        
        if dataset =='cifar10':
            decay_epoch = 10
            lr_mu = 0.01
            lr_mu_decay = 0.75
            lr_si = 0.01
            lr_si_decay = 0.75
            lr_fw = 0.01
            lr_fw_decay = .75
            decay_check = lambda x: x>decay_epoch and x%decay_epoch==1
        '''
        lr_all  new value: 0.01
        lr_mu  new value: 0.01
        lr_si  new value: 0.0001
        lr_fw  new value: 0.01
        this achives 99.56 in repeatition 2 for focusing neuron
        same for dense network. 
        
        Be careful that the variance of the results is high!
        you may need to repeat the same experiment few times
        to compare two cases
        ''' 

    all_params = lasagne.layers.get_all_params(network, trainable=True)
    set_params_value(LR_params,[lr_all,lr_mu,lr_si,lr_fw])
    batch_num = 512
    #batch_num = 16
    print_int = 20
    debug_params_save = False
    debug_print_param_stats = False
  
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if debug_params_save:
            record_params =  ['focus-0.mu','focus-0.si','focus-0.W','focus-1.mu',
                              'focus-1.si','focus-1.W']
            param_list.append(get_params_values_wkey(all_params,record_params ))
        
        if decay_check(epoch):
   
            lr_all = lr_all*lr_all_decay
            lr_fw= lr_fw*lr_fw_decay
            lr_mu= lr_mu*lr_mu_decay
            lr_si= lr_si*lr_si_decay        
            set_params_value(LR_params,[lr_all,lr_mu,lr_si,lr_fw])
        
        for batch in iterate_minibatches(X_train, y_train, batch_num, shuffle=True):
            inputs, targets = batch
            if epoch==0:
                err, acc = eval_fn(inputs, targets)
                train_err +=err
            else:                
                train_err += train_fn(inputs, targets)
            train_batches += 1
        
        trn_err_list.append(train_err/train_batches)
        
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, y_val.shape[0]//4, shuffle=False):
            inputs, targets = batch
            err, acc = eval_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        if (epoch%print_int==0 or epoch==1):
            print("Model {} Epoch {} of {} took {:.3f}s".format(model, epoch, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
            if debug_print_param_stats:
                print_param_stats(network)

        
        val_acc_list.append(val_acc / val_batches * 100)
        val_err_list.append(val_err / val_batches)
        if np.isnan(train_err):
            break
         
        tst_err  = 0
        tst_acc = 0
        tst_batches = 0
        for batch in iterate_minibatches(X_test, y_test, y_test.shape[0]//4, shuffle=False):
             inputs, targets = batch
             err, acc = eval_fn(inputs, targets)
             tst_err += err
             tst_acc += acc
             tst_batches += 1
        
        tst_acc_list.append(tst_acc /tst_batches* 100) # to pick the tst error at best val accuracy. 
        tst_err_fin = tst_err / tst_batches
        tst_acc_fin = tst_acc / tst_batches * 100
    
    # After training, we compute and print the test error:    
    #print(val_acc_list)
    val_ac_np = np.asarray(val_acc_list)
    best_val = np.argmax(val_ac_np)
    if np.isnan(train_err):
        return
        
    
    print("\nFinal results:")
    print("  test loss:\t\t\t{:.6f}".format(tst_err_fin))
    print("  test accuracy:\t\t{:.2f} %".format(tst_acc_fin))
    
    print("\nTest result at best val epoch: ", best_val)
    #print(tst_acc_list)
    print("  test accuracy:\t\t{:.2f} %".format(tst_acc_list[best_val]))
    best_test_early_stop = tst_acc_list[best_val]
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    print("_result_change")
    print(start_time, timestr)
    if os.path.exists(folder):
        filename= str(folder+dataset+"_result_"+model+"_"+exp_start_time+"_"+timestr)
    else:
        filename= str(dataset+"_result_"+model+"_"+exp_start_time+"_"+timestr)
    
    np.savez(filename,(trn_err_list, val_err_list, val_acc_list, tst_err_fin, 
                       tst_acc_fin*100, tst_acc_list, best_test_early_stop))

    # save model and code 
    if os.path.exists(folder):
        filename= str(folder+dataset+"_model_"+model+"_"+timestr)
    else:
        filename= str('outputs/'+dataset+"_model_"+model+"_"+timestr)
    
    fixed_params = lasagne.layers.get_all_params(network, trainable=False)
    fixed_params =[t.name for t in fixed_params]
    trn_params = lasagne.layers.get_all_params(network, trainable=True)
    trn_params =[t.name for t in trn_params]
    fixed_param_values = lasagne.layers.get_all_param_values(network, trainable=False)
    trn_param_values = lasagne.layers.get_all_param_values(network, trainable=True)
    
    np.savez(filename, trn_params, trn_param_values, fixed_params, fixed_param_values)
    if debug_params_save:
        filename= str(folder+dataset+"_debug_params_"+model)
        np.savez_compressed(filename, param_list, record_params)
    
        
    plt_figures = False
    if plt_figures:
        import matplotlib.pyplot as plt
        plt.plot(trn_err_list)
        plt.plot(val_err_list)
        plt.ylim([0, 0.25])
        plt.title("Train and Validation Error")
        plt.legend(("Train","Validate"))
        plt.show()


if __name__ == '__main__':
    if ('--help' in sys.argv) or (len(sys.argv)==1):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS] [REPEATS] [DATASET] [EXNAME] [STARTDELAY]]" % sys.argv[0])
        print("")
        print("MODEL: one of the below")
        print("       'mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       'focused_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an focused MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("       'focused_cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
        print("REPEATS: repeat whole experiments (default: 1)")
        print("DATASET: dataset 'mnist', 'mnist_cluttered', 'cifar10', 'fashion',(default: 'mnist')")
        print("EXNAME: name of the subfolder given to the experiment, default:mnist")
        print("DELAY: delay in hours to start the experiment. write 0.5 for an half an hour delay")
        print("example:")
        print("run mnist.py focused_mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0")
        print("for cnn:")
        print("run mnist.py cnn2d 350 5 cifar10 cifar 0.0")
    else:
        kwargs = {}
        n_reps = 1
        kwargs['num_epochs'] = 10
        kwargs['dataset'] = 'mnist'
        kwargs['exname'] = 'mnist'
        kwargs['delay'] = 0
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            n_reps = int(sys.argv[3])
            print("repeating ", n_reps, " reps")
        if len(sys.argv) > 4:
            kwargs['dataset'] = sys.argv[4]
            print("dataset ", sys.argv[4])
        if len(sys.argv) > 5:
            kwargs['exname'] = sys.argv[5]
            print("exname", sys.argv[5])
        if len(sys.argv) > 6:
            kwargs['delay'] = sys.argv[6]
            print("delay", sys.argv[6])
        
        # Sleep to delay
        time.sleep(3600*float(kwargs['delay']))
    
        # prepare destination folder
        #kwargs['folder'] = 'outputs/ESNN/'+kwargs['exname']
        kwargs['folder'] = 'outputs/'+kwargs['exname']
        
    
        #put a random seed.
        RANDSEED = 41
        lasagne.random.set_rng(np.random.RandomState(RANDSEED))  # Set random state so we can investigate results
        np.random.seed(RANDSEED)
        from datetime import datetime
        now = datetime.now()
        timestr = now.strftime("%Y%m%d-%H%M%S")
        # get the code for this run.
        from shutil import copyfile
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        if not os.path.exists(kwargs['folder']):
            os.mkdir(kwargs['folder'])
            
        if os.path.exists(kwargs['folder']):
            print(kwargs['folder'] +"code_"+"mnist_"+kwargs['model']+timestr+".py")
            copyfile("mnist.py", kwargs['folder'] +"code_"+"mnist_"+kwargs['model']+timestr+".py")
    
        for i in range(n_reps):
            print("Repetition: ",i+1)
            main(model=kwargs['model'], num_epochs=kwargs['num_epochs'], 
                 dataset=kwargs['dataset'], folder=kwargs['folder'],exp_start_time=str(timestr))
