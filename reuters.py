#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 01:10:52 2018

@author: btek

"""

import os
os.environ['THEANO_FLAGS']='device=cuda0, floatX=float32, gpuarray.preallocate=.1'
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
print_param_stats, get_params_values_wkey


from data_utils import load_dataset_reuters

def build_custom_mlp(input_var=None, input_shape=(None,1,28,28),
                     output_shape = 10,
                     depth=2, width=800, drop_input=.2, drop_hidden=.5,
                     batch_norm=True,regress=False):
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
    outlin = lasagne.nonlinearities.softmax
    if regress:
        outlin = lasagne.nonlinearities.linear
        nonlin = lasagne.nonlinearities.tanh
    
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
        l_hidden = lasagne.layers.DenseLayer(net_layers[k-1], num_units=width, 
                                             nonlinearity=lin, W=ini,
                                             name='dense-'+str(lay))
                
        net_layers.append(l_hidden)
        k+=1
        if batch_norm:
            l_bn = lasagne.layers.NonlinearityLayer(
                    lasagne.layers.BatchNormLayer(net_layers[k-1]), nonlinearity=nonlin)
        else:
            l_bn = lasagne.layers.NonlinearityLayer(net_layers[k-1],nonlinearity=nonlin)
        
        net_layers.append(l_bn)
        k+=1
        if drop_hidden:
            l_drp = lasagne.layers.dropout(net_layers[k-1], p=drop_hidden)
            net_layers.append(l_drp)
            k+=1
    # Output layer:
    
    l_out = lasagne.layers.DenseLayer(
            net_layers[k-1], num_units=output_shape, W=ini,
            nonlinearity=outlin, name='dense'+str(depth+1))
   
    return l_out

def build_focused_mlp(input_var=None, input_shape=(None,1,28,28), 
                      output_shape = 10,
                      depth=2, width=800, drop_input=.2,
                     drop_hidden=.5, update_mu=True, update_si=True,
                     init_mu='spread', initsigma=0.1, batch_norm=True, regress=False):
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
    if regress:
        outlin = lasagne.nonlinearities.linear
        nonlin = lasagne.nonlinearities.tanh
    else:
        softmax = lasagne.nonlinearities.softmax
        outlin = softmax  
    
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
                             scaler=1., weight_gain=1.0,
                             trainScaler=False, trainWs=True)
                
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
    
        
    l_out = lasagne.layers.DenseLayer(
            net_layers[k-1], num_units=output_shape, W=ini,
            nonlinearity=outlin, name='dense'+str(depth+1))
   
    return l_out


def build_network_model(model, input_var, input_shape, output_shape, 
                        batch_norm=True, regress=False):
     # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model.startswith('mlp'):
        extraparams = model.split(':', 1)
        if len(extraparams)==1:
            network = network = build_custom_mlp(input_var,input_shape=input_shape, 
                                       output_shape=output_shape,
                                       depth=int(2), width=int(100),
                                       batch_norm=batch_norm, regress=regress)
        else:
            depth, width, drop_in, drop_hid = extraparams[1].split(',')
            network = build_custom_mlp(input_var,input_shape=input_shape, 
                                       output_shape=output_shape,
                                       depth=int(depth), width=int(width),
                                       batch_norm=batch_norm, regress=regress)
        
    elif model.startswith('focused_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        depth, width, drop_in, drop_hid = int(depth), int(width), float(drop_in), float(drop_hid)
        network = build_focused_mlp(input_var,input_shape=input_shape, depth=depth,
                                    width=width, drop_input=drop_in, drop_hidden=drop_hid,
                                    output_shape=output_shape,
                                    update_mu=True, update_si=True,
                                    init_mu='spread',initsigma=.1,
                                    batch_norm=batch_norm, regress=regress)
    elif model.startswith('fixed_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        depth, width, drop_in, drop_hid = int(depth), int(width), float(drop_in), float(drop_hid)
        print(depth, width, drop_in, drop_hid)
        network = build_focused_mlp(input_var,input_shape=input_shape,
                                    output_shape= output_shape,
                                    depth=depth, width=width, drop_input=drop_in, 
                                    drop_hidden=drop_hid, init_mu='spread',
                                    update_mu=False, update_si=False,initsigma=.1,
                                    batch_norm=batch_norm, regress=regress)
    else:
        print("Unrecognized model type %r." % model)
        return
    
    return network

def build_functions(network, input_var, target_var, regress=False):
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    if not regress:
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    else:
        loss = lasagne.objectives.squared_error(prediction, target_var)
        
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    
    LR_rate = theano.shared(np.float32(0.0),name='lr_all')
    LR_MU = theano.shared(np.float32(0.0),name='lr_mu')
    LR_SI = theano.shared(np.float32(0.0),name='lr_si')
    LR_FW = theano.shared(np.float32(0.0),name='lr_fw')
    LR_params = [LR_rate, LR_MU, LR_SI, LR_FW]
    
    #updates = lasagne.updates.nesterov_momentum(
    #        loss, params, learning_rate=LR_rate, momentum=0.9)
    #updates = lasagne.updates.adagrad(loss, params, learning_rate=LR_rate)
    #updates = sgdWithLrLayers(loss, params, learning_rate=LR_rate, 
    #                     mu_lr=LR_rate, si_lr=LR_rate*.1, 
    #                     focused_w_lr=LR_rate, momentum=.90)
    
    updates = sgdWithLrsClip(loss, params, learning_rate=LR_rate, 
                         mu_lr=LR_MU, si_lr=LR_SI, 
                         focused_w_lr=LR_FW, momentum=.90)
 
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    if not regress:
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), 
                               T.argmax(target_var, axis=1)),dtype=theano.config.floatX)

    else:
        test_loss = lasagne.objectives.squared_error(test_prediction,
                                                     target_var)
        test_loss = test_loss.mean()
        test_acc = test_loss
        # No acccuracy here. but to be compatible I keep accuracy
        
    
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    
    train_fn = theano.function([input_var, target_var],
                               loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn, LR_params


def main(model='mlp', num_epochs=500, dataset='reuters', folder="", exp_start_time=None):
    # Load the dataset
    print("Loading data...")
      
    if dataset== 'boston':

        from sklearn import cross_validation
        from sklearn import preprocessing
        from sklearn import datasets
        #from sklearn.utils import shuffle
        boston = datasets.load_boston()
        X, y = boston.data.astype('float32'), boston.target.astype('float32')
        #X, y = shuffle(boston.data, boston.target, random_state=13)
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                X, y, test_size=0.1, random_state=42)
        
        #X_train = scaler.fit_transform(X_train)
        X_val = X_train.copy()
        y_val = y_train.copy()
        print("validation is just a copy of X_train, so results will be similar but with no drop out")
        
        
        from sklearn import ensemble
        from sklearn.metrics import mean_squared_error
        params = {'n_estimators': 150, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)

        clf.fit(X_train, y_train)   
        mse_train = mean_squared_error(y_train, clf.predict(X_train))
        mse_test = mean_squared_error(y_test, clf.predict(X_test))
        print("GRAD BOOST MSE train: %.4f" % mse_train)
        print("GRAD BOOST MSE test: %.4f" % mse_test)
        
        input_var = T.fmatrix('inputs')
        target_var = T.fvector('targets')
        input_shape= (None, X_train.shape[1])
        output_shape  = 1
        batch_num = 128
        regress= True
        batch_norm =False
        print(output_shape)
        
        network = build_network_model(model, input_var, input_shape, output_shape,
                                      batch_norm=batch_norm, regress=regress)
        print(network.output_shape)
        train_fn, eval_fn, LR_params = build_functions(network, input_var, 
                                                       target_var, 
                                                       regress=regress)
    elif dataset.startswith('reuters'):
        X_train, y_train, X_val, y_val, X_test, y_test =load_dataset_reuters('../datasets/reuters/')
        print ("Train: ",X_train.shape, "Val: ", X_val.shape, "Test: ",X_test.shape)
        input_var = T.fmatrix('inputs')
        target_var = T.fmatrix('targets')
        input_shape= (None, X_train.shape[1])
        output_shape  = y_train.shape[1]
        regress = False
        batch_norm = True
        
#        from sklearn import ensemble
#        from sklearn.metrics import accuracy_score
#        params = {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 2}
#        clf = ensemble.RandomForestClassifier(**params)
#
#        clf.fit(X_train, y_train)   
#        mse_train = accuracy_score(y_train, clf.predict(X_train))
#        mse_test = accuracy_score(y_test, clf.predict(X_test))
#        print("RANDOM  train: %2.4f" % mse_train)
#        print("RANDOM  test: %2.4f" % mse_test)
#        
#        from sklearn import neural_network
#        clf = neural_network.MLPClassifier(hidden_layer_sizes=(150,), 
#                                           activation='relu', 
#                                           solver='sgd', batch_size=16, 
#                                           learning_rate='constant', 
#                                           learning_rate_init=0.001, 
#                                           max_iter=250, 
#                                           early_stopping=True,
#                                           shuffle=True)
#        clf.fit(X_train, y_train)   
#        mse_train = accuracy_score(y_train, clf.predict(X_train))
#        mse_test = accuracy_score(y_test, clf.predict(X_test))
#        print("KNN  train: %2.4f" % mse_train)
#        print("KNN  test: %2.4f" % mse_test)
#        import pdb
#        pdb.set_trace()
        batch_num = 16
        network = build_network_model(model, input_var, input_shape, output_shape,
                                      batch_norm=batch_norm, regress=regress)
        print(network.output_shape)
        train_fn, eval_fn, LR_params = build_functions(network, input_var, 
                                                       target_var, regress=regress)
    
    # Prepare Theano variables for inputs and targets
    print("input shape:", input_shape)
    
    val_acc_list =[]
    tst_acc_list =[] 
    val_err_list = []
    trn_err_list = []
    
    print("Model", model)
    if model.startswith("mlp:"):
        lr_all = 5e-4
    else:
        lr_all = 1e-4 #reuters best 1e-4 for focused doesnt change 5e-5
        
    lr_all_decay = .9
    lr_mu = 0.001 
    lr_mu_decay = 0.9
    lr_si = 0.001
    lr_si_decay = 0.9
    lr_fw = 0.001
    lr_fw_decay = .9
    decay_epoch = 30
    print_int = 10
    if dataset=='boston':
        lr_all = 0.005
        lr_all_decay = .9
        lr_mu = 0.001
        lr_mu_decay = 0.9
        lr_si = 0.001
        lr_si_decay = 0.9
        lr_fw = 0.005
        lr_fw_decay = .9
        decay_epoch = 1000
        print_int = 1000
    
    set_params_value(LR_params,[lr_all,lr_mu,lr_si,lr_fw])
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
      
        if (epoch>1 and epoch%decay_epoch==1):
            #lr_all = 0.001
            #lr_fw = 0.001
            lr_all = lr_all * lr_all_decay
            lr_mu = lr_mu * lr_mu_decay
            lr_si = lr_si * lr_si_decay
            lr_fw = lr_fw * lr_fw_decay
            
            set_params_value(LR_params,[lr_all,lr_mu,lr_si,lr_fw])
        
        for batch in iterate_minibatches(X_train, y_train, batch_num, shuffle=True):
            inputs, targets = batch

            train_err += train_fn(inputs, targets)
            train_batches += 1
        
        trn_err_list.append(train_err/train_batches)
        
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        
        for batch in iterate_minibatches(X_val, y_val, y_val.shape[0], shuffle=False):
            inputs, targets = batch
            err, acc = eval_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            
            
        train_err = train_err / train_batches
        val_err = val_err / val_batches
        val_acc = val_acc / val_batches * 100
        val_err_list.append(val_err)
        # Then we print the results for this epoch:
        if (epoch%print_int==0):
            print("Model {} Epoch {} of {} took {:.3f}s".format(model, epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:2.6f}".format(train_err))
            print("  validation loss:\t\t{:2.6f}".format(val_err))
            if not regress:
                print("  validation accuracy:\t\t{:2.4f} %".format(val_acc))
            else:
                val_acc = 1-val_err
                
            print_param_stats(network)
            #debug_focus_vars(network)
        
        
        
        if np.isnan(train_err):
            print("Train error NAN")
            break
        tst_err, tst_acc = eval_fn(X_test, y_test)
        if not regress:
            tst_acc_list.append(tst_acc * 100) # to pick the tst error at best val accuracy.
            val_acc_list.append(val_acc)
        else:
            tst_acc_list.append(tst_err) # to pick the tst error at best val accuracy. 
            val_acc_list.append(val_err)
    # After training, we compute and print the test error:
    
    val_ac_np = np.asarray(val_acc_list)
    if regress:
        best_val = np.argmin(val_ac_np)
    else:
        best_val = np.argmax(val_ac_np)
    if np.isnan(train_err):
        return
    tst_err_fin, tst_acc_fin = eval_fn(X_test, y_test)
    print("\nFinal results:")
    print("  test loss:\t\t\t{:.6f}".format(tst_err_fin))
    print("  test accuracy:\t\t{:.4f} %".format(tst_acc_fin))
    
    print("\nTest result at best val epoch: ", best_val)
    print("  test accuracy:\t\t{:.4f} %".format(tst_acc_list[best_val]))
    
    best_test_early_stop = tst_acc_list[best_val]
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    print("_result_change")
    print(start_time, timestr)
    filename= str(folder+dataset+"_result_"+model+"_"+exp_start_time+"_"+timestr)
    
    np.savez(filename,(trn_err_list, val_err_list, val_acc_list, tst_err_fin, 
                       tst_acc_fin*100, tst_acc_list, best_test_early_stop))

    
    # save model and code 
    filename= str(folder+dataset+"_model_"+model+"_"+timestr)
    fixed_params = lasagne.layers.get_all_params(network, trainable=False)
    fixed_params =[t.name for t in fixed_params]
    trn_params = lasagne.layers.get_all_params(network, trainable=True)
    trn_params =[t.name for t in trn_params]
    fixed_param_values = lasagne.layers.get_all_param_values(network, trainable=False)
    trn_param_values = lasagne.layers.get_all_param_values(network, trainable=True)
    
    np.savez(filename, trn_params, trn_param_values, fixed_params, fixed_param_values)




    
        
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
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on Reuters using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print("Example run 'reuters.py' focused_mlp:2,150,0.2,0.25 100 1 reuters")
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       'focused_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an focused MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("       'focused_cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
        print("REPEATS: repeat whole experiments (default: 1)")
        print("DATASET: dataset (default: mnist)")
    else:
        kwargs = {}
        n_reps = 1
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
        
        kwargs['folder'] = "outputs/ESNN/boston/Jul_24/"
        time.sleep(3600*0)
        RANDSEED = 41
        lasagne.random.set_rng(np.random.RandomState(RANDSEED))  # Set random state so we can investigate results
        np.random.seed(RANDSEED)
        from datetime import datetime
        now = datetime.now()
        timestr = now.strftime("%Y%m%d-%H%M%S")
        # get the code for this run.
        from shutil import copyfile
        copyfile("reuters.py",  kwargs['folder'] +"code_"+"reuters_"+kwargs['model']+timestr+".py")

        for i in range(n_reps):
            print("Repetition: ",i)
            main(model=kwargs['model'], num_epochs=kwargs['num_epochs'], 
                 dataset=kwargs['dataset'], folder=kwargs['folder'],exp_start_time=str(timestr))
