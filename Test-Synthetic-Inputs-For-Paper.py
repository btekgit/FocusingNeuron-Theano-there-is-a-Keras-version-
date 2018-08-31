# -*- coding: utf-8 -*-

#from __future__ import unicode_literals
'''
/*
 * Focusing Neuron Test Case
 *
 * Code Authors:  F. Boray Tek and 
 * İlker Çam contributed to an earlier version
 *
 * All rights reserved.
 *
 * For details, see the paper:
 * 
 *  
 * 
 *
 * Permission to use, copy, modify, and distribute this software and
 * its documentation for educational, research, and non-commercial
 * purposes, without fee and without a signed licensing agreement, is
 * hereby granted, provided that the above copyright notice and this
 * paragraph appear in all copies modifications, and distributions.
 *
 * NOTE: THIS WORK IS PATENT PENDING!
 * Patent rights are owned by F. Boray Tek, İlker Çam, Işık University
 * Any commercial use or any redistribution of this software
 * requires a license from one of the above mentioned establishments.
 *
 * For further details, contact F. Boray Tek (boraytek@gmail.com).
 */
'''

# In[1]:

import os
os.environ['THEANO_FLAGS']='device=cpu,floatX=float32,preallocate=0.1'
os.environ['MKL_THREADING_LAYER']='GNU'
import time
import matplotlib.pyplot as plt
import lasagne
import theano
theano.config.exception_verbosity = 'high'
import theano.tensor as T
import numpy as np
from focusing import FocusedLayer1D
from collections import OrderedDict
from lasagne.updates import get_or_compute_grads, apply_momentum, sgd
from lasagne.updates import momentum, adam, adadelta
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

from plot_utils import save_fig, paper_fig_settings

from lasagne_utils import sgdWithLrsClip,categorical_focal_loss,\
get_shared_by_pattern,sgdWithLrs,iterate_minibatches,set_params_value,\
debug_print_param_stats, get_params_values_wkey,sgdWithWeightSupress

from data_utils import load_blob
    
def build_model(input_feas, classes, hidden_count, batchnorm=True):
    # Initers, Layers
    ini = lasagne.init.GlorotUniform()
    
    nonlin = lasagne.nonlinearities.relu
    softmax = lasagne.nonlinearities.softmax
    
    lin = lasagne.nonlinearities.linear
    
    # Input Layer
    l_in = lasagne.layers.InputLayer(shape=(None, input_feas))
    
    # Denses
    l_dense1 = lasagne.layers.DenseLayer(
            l_in, num_units=hidden_count, 
            nonlinearity=lin, 
            W=ini, name="dense1", b=None)
    
    
    if batchnorm:
    # if you close BATCHNORM weights get LARGE
        l_bn = lasagne.layers.NonlinearityLayer(
                lasagne.layers.BatchNormLayer(l_dense1), nonlinearity=nonlin)
    
    else:
    
        l_bn = lasagne.layers.NonlinearityLayer(l_dense1, nonlinearity=nonlin)
    #l_dense2 = lasagne.layers.DenseLayer(l_dense1, num_units=4, nonlinearity=lasagne.nonlinearities.tanh, W=ini, name='dense2')
    
    #l_drop1 = lasagne.layers.dropout(l_bn, p=0.1)
    
    # Output Layer
    l_out = lasagne.layers.DenseLayer(l_bn, num_units=classes, nonlinearity=softmax, W=ini, name='output')
    
    
    penalty = (l2(l_dense1.W)*1e-4)+(l1(l_dense1.W)*1e-6) +(l2(l_out.W)*1e-3)
    if not USE_PENALTY:
        penalty = penalty*0
    
    #penalty = penalty*0
    #penalty = (l2(l_dense1.W)*1e-30)#(l2(l_dense1.W)*1e-3)+(l1(l_dense1.W)*1e-6) +(l2(l_out.W)*1e-3)
    
    return l_out, penalty


def build_model_focusing(input_feas, classes, hidden_count, batchnorm=True):
    # Initers, Layers
    ini = lasagne.init.HeUniform()
    nonlin = lasagne.nonlinearities.relu
    linear = lasagne.nonlinearities.linear
    softmax = lasagne.nonlinearities.softmax

    
    # Input Layer
    
    l_in = lasagne.layers.InputLayer(shape=(None, input_feas))
    
    l_focus1 = FocusedLayer1D(l_in, num_units=hidden_count, 
                              nonlinearity=linear, name='focus1',
                              trainMus=UPDATE_MU, 
                              trainSis=UPDATE_SI, 
                              initMu=INIT_MU, 
                              W=ini, withWeights=WITH_WEIGHTS, 
                              bias=lasagne.init.Constant(0.0), 
                              initSigma=INIT_SI, 
                             scaler=INIT_SCALER, weight_gain=1.0, 
                             trainScaler=UPDATE_SCAlER, trainWs=True)    
    
    if batchnorm:
    # if you close BATCHNORM weights get LARGE
        l_bn = lasagne.layers.NonlinearityLayer(
                lasagne.layers.BatchNormLayer(l_focus1), nonlinearity=nonlin)
    
    else:
    
        l_bn = lasagne.layers.NonlinearityLayer(l_focus1, nonlinearity=nonlin)
 
    #l_drop1 = lasagne.layers.dropout(l_bn, p=0.1)
    
    # Output
    l_out = lasagne.layers.DenseLayer(l_bn, num_units=classes, 
                                      nonlinearity=softmax, W=ini, name='output')
    
    penalty = l2(l_out.W)*1e-3
    if WITH_WEIGHTS:
        penalty += l2(l_focus1.W)*1e-4+(l1(l_focus1.W)*1e-6) +l2(l_focus1.si)*1e-2
    
    if not USE_PENALTY:
        penalty = penalty*0
    
    
    return l_out, penalty


# Compile train and eval functions
def build_functions(using_model, using_parameters, penalty):
    X = T.fmatrix()
    y = T.ivector()

    # training output
    output_train = lasagne.layers.get_output(using_model, X, deterministic=False)

    # evaluation output. Also includes output of transform for plotting
    output_eval = lasagne.layers.get_output(using_model, X, deterministic=True)
    
    cost = T.mean(lasagne.objectives.categorical_crossentropy(output_train, y)) + penalty # Regularization
  
    #updates = adam(cost, params, learning_rate=0.001)
    #updates = momentum(cost, params, learning_rate=0.01, momentum=0.5)
    # We Use our own update function for giving different 
    #updates = sgdWithLrs(cost, using_parameters, LEARNING_RATE, LR_MU, LR_SI, LR_FW)
    updates = sgdWithLrsClip(cost,using_parameters,LEARNING_RATE, LR_MU, LR_SI, LR_FW)
    #updates = sgdWithWeightSupress(cost,using_parameters,LEARNING_RATE, LR_MU, LR_SI, LR_FW)
    
    test_acc = T.mean(T.eq(T.argmax(output_eval, axis=1), y), dtype=theano.config.floatX)

    eval = theano.function([X, y], [cost, test_acc], allow_input_downcast=True)
    train = theano.function([X, y], [cost, output_train, penalty], updates=updates, allow_input_downcast=True)
    
    return train, eval

# this initialization, weight variance correction method was not used in the paper.
def weight_adjustment_batch(X, output_func, getter, setter):
    print ("LSUV initialization")
    n_samples = 128
    random_ix = np.random.permutation(X.shape[0])
    X_batch = X[random_ix[0:n_samples]]
    #print ("X var: ",np.var(X_batch))
    Y = output_func(X_batch)
    
    #print ("Y shape:", Y[0].shape)
    W = getter()
    variance = np.mean(np.var(Y[0], axis=0))
    needed_variance = 1.0
    margin = 0.02
    iteration = 0
    #print ("Y var: ",np.var(Y[0], axis=0), "W var ", np.var(W, axis=0))
    while abs(needed_variance - variance) > margin and iteration<5:
        if np.abs(np.sqrt(variance)) < 1e-7:
        # avoid zero division
            break
    
        weights = W   
        #print "norming with:",np.var(Y,axis=0)
        #print (" shape", (np.var(Y[0],axis=0)).shape)
        weights /= np.sqrt(np.var(Y[0], axis=0))    
        #weights /= np.sqrt(variance) / np.sqrt(needed_variance)    
        setter(np.copy(W))
        Y = output_func(X)
        variance = np.mean(np.var(Y[0], axis=0))
        iteration = iteration + 1
        #print ("Y var: ",np.var(Y[0], axis=0), "W var ", np.var(W, axis=0))
        #print ("var mean: ",variance, "iter: ",iteration)
    
    print ("Y var: ",np.var(Y[0], axis=0), "W var ", np.var(W, axis=0))
    

# Epoch tranining with mini-batch
def train_epoch(X, y, trnfnc):
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    costs = []
    correct = 0
    penalties = []
    for i in range(num_batches):
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        X_batch = X[idx]
        y_batch = y[idx]
        cost, output_train, penalty = trnfnc(X_batch, y_batch)
        costs += [cost]
        penalties += [penalty]
        preds = np.argmax(output_train, axis=-1)
        correct += np.sum(y_batch == preds)
    
    return np.mean(costs), correct / float(num_samples), np.mean(penalties)

# Evaluation function
def eval_epoch(X, y, elvfnc):
    cost, acc = elvfnc(X, y)
    return cost, acc


# In[5]:
# GENERAL SETINGS
# Additional Settings
    
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ESNN/"
EXPERIMENT_ID = "EX-1"
RANDSEED = 42
paper_fig_settings()

from  datetime import datetime
now = datetime.now()
timestr = now.strftime("%Y%m%d-%H%M%S")
logdir = path = os.path.join(PROJECT_ROOT_DIR, "outputs", 
                        CHAPTER_ID+EXPERIMENT_ID)


lasagne.random.set_rng(np.random.RandomState(RANDSEED))  # Set random state so we can investigate results
np.random.seed(RANDSEED)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)

# In[6]:
# Training and Settings
NUM_EPOCHS = 250
 # just for experiment 34
BATCH_SIZE = 128

USEBATCHNORM = True
# note that learning rate of the focusing neuron weights will be x5
MOMENTUM = 0.9
USE_PENALTY = False  
USE_LSUV_INIT = False
PLOT_INT_RESULTS = False
PRINT_INTERVAL = 10
RECORD_INTERVAL= 1
N_REPEATS =1
    
best_results =[]

all_results = np.zeros((N_REPEATS,int(NUM_EPOCHS/RECORD_INTERVAL),2))

# the loop to repeat train-test cycle
for rep in range(N_REPEATS):
    
    # Dataset Settings
    CLASSES = 2
    #CLASSES = 2**(rep//3+1)
    print('classes:',CLASSES)
    FEATURES = 20
    #N_CLUSTERS = 2+2*(rep//3)
    N_CLUSTERS = 2    
    #FEATURES = 4*(5**(rep//3))
    SAMPLES = 2000
    #SAMPLES = 200*(10**(rep//3))
    print('samples:',SAMPLES)
    #DUMMY_POINTS = int(FEATURES * .5) # Percentage
    NOISE_DIMS = int(FEATURES *1.0) # Percentage5
    NOISE_SCALE = 1.0
    NOISE_PATTERN= "left"
    
    LEARNING_RATE = 0.001
    #LEARNING_RATE = 0.1 * (10**-(rep//3))
    
    #NUM_EPOCHS = 250*(1+rep//3)

    # NETWORK Settings
    #HIDDEN_COUNT = 40 # WORKS WHEN IT IS MORE THAN EQUAL TO FEATURES
    #HIDDEN_COUNT = 20*(rep//3+1)
    HIDDEN_COUNT = 4
    #HIDDEN_COUNT = FEATURES+2*NOISE_DIMS # WORKS WHEN IT IS MORE THAN EQUAL TO FEATURES
    #HIDDEN_COUNT = FEATURES+2*NOISE_DIMS-5 # WORKS WHEN IT IS MORE THAN EQUAL TO FEATURES
    print('neurons:',HIDDEN_COUNT)
    #if HIDDEN_COUNT < (FEATURES+NOISE_DIMS):
    #    print("focusing neuron works when the neurons are equal or more than input")
    #LR_MU = np.float32((0.5 / TOTAL_FEATURES) * .1, dtype='float32'); 
    LR_MU = np.float32((LEARNING_RATE)*.1)
    LR_SI = np.float32((LEARNING_RATE*.1))
    LR_FW = np.float32((LEARNING_RATE))
    LR_SCALER = 0.01; 
    print ("LR W: ", LEARNING_RATE, "LR_MU", LR_MU, "LR_SI", LR_SI)
    UPDATE_SI = True
    UPDATE_MU = True    #LR_SI = np.float32((0.5 / TOTAL_FEATURES) * .1, dtype='float32'); 
    UPDATE_SCAlER = False
    #INIT_MU = 'middle_random'#'spread' # 'middle' Or sth else
    INIT_MU = 'middle_random'# 'middle' Or sth else
    #INIT_MU = np.array([0.495,0.499,0.501,0.51])#'middle'#'spread' # 'middle' Or sth else
    INIT_SI = 0.08 #0.08# 0.20 best
    INIT_SCALER = 1.0
    WITH_WEIGHTS = True
    NORM = True
    PRUNE = False

# In[7]:
    # generate data first. 
    data = load_blob(CLASSES, FEATURES, SAMPLES, RANDSEED, noise_dims=NOISE_DIMS,
                     noise_scale=NOISE_SCALE, noise_pattern=NOISE_PATTERN,
                     clusters=N_CLUSTERS)
# In[8]:
# Plot part of the data
    n_features = data['X_train'].shape[1]
    if PLOT_INT_RESULTS:
        print(data['X_train'].shape)
    
        
        X = data['X_test']
        Y = data['y_test']
        plt.scatter(X[:,1], X[:,2], marker='o',c=Y)
        plt.show()

# In[9]:
# Construct  models and get function probes to read and set shared variables

    model_dense, penalty_dense = build_model(n_features, CLASSES, HIDDEN_COUNT,USEBATCHNORM)
    mp_dense = lasagne.layers.get_all_params(model_dense, trainable=True)
    l_dense = next(l for l in lasagne.layers.get_all_layers(model_dense) if l.name is "dense1")
    dense_mp = l_dense.W
    
    model_focus, penalty_focus = build_model_focusing(n_features, CLASSES, HIDDEN_COUNT,USEBATCHNORM)
    mp_focus = lasagne.layers.get_all_params(model_focus, trainable=True)
    l_focused = next(l for l in lasagne.layers.get_all_layers(model_focus) if l.name is 'focus1')
    #l_dense1 = next(l for l in lasagne.layers.get_all_layers(model_focus) if l.name is 'dense1')
    
    get_si = lambda: l_focused.si.get_value()
    get_mu = lambda: l_focused.mu.get_value()
    get_foci = lambda: l_focused.calc_u().eval().T
    get_scaler = lambda: l_focused.scaler.get_value()
    set_scaler = lambda value: l_focused.scaler.set_value(value)
    if WITH_WEIGHTS:
        get_w = lambda: l_focused.W.get_value()
        set_w = lambda value: l_focused.W.set_value(value)
    else:
        get_w = lambda: 1
        set_w = lambda x: x

# In[10]:
# construct training and evaluation, and (intermediate output funcs)

    train_focus, eval_focus = build_functions(model_focus, mp_focus, penalty_focus)
    train_dense, eval_dense = build_functions(model_dense, mp_dense, penalty_dense)
    
    X = T.fmatrix()
    output_focus = lasagne.layers.get_output(l_focused, X, deterministic=True)
    eval_output_focus = theano.function([X], [output_focus], allow_input_downcast=True)
    
    output_dense = lasagne.layers.get_output(l_dense, X, deterministic=True)
    eval_output_dense = theano.function([X], [output_dense], allow_input_downcast=True)

    #weight_adjustment_batch(data['X_train'], eval_output_dense, 
    #                        getter=l_dense.W.get_value,
    #                        setter=lambda value: l_dense.W.set_value(value))    
    

    total_time = 0
    costs_dense, costs_focus = [], []
    costs_tst_dense, costs_tst_focus = [], []
    accs_dense, accs_focus = [], []
    focus_outputs = []
    dense_weights = []
    mus = []
    sis = []
    foci = []
    w_change = []
    scalers = []
    try:
        for n in range(NUM_EPOCHS):
            
            if n==0 and USE_LSUV_INIT and WITH_WEIGHTS:
                #print ("Weight mean before ", np.var(get_w()))
                weight_adjustment_batch(data['X_train'], eval_output_focus, getter=get_w,setter=set_w) 
                
                #print ("Weight mean before ", np.var(get_w()))
            if n == 0: 
                mus.append(get_mu())
                sis.append(get_si())
                foci.append(get_foci()*get_w())
                scalers.append(get_scaler())
                w_change.append(get_w())
    
            #print("w:",np.asarray(get_w())[:,0], "mu: ",get_mu()[0],"si: ",get_si()[0])
            start_time = time.time()
            train_cost_dense, train_acc_dense, penalty_dense = train_epoch(data['X_train'], data['y_train'], train_dense)
            time_spent_dense = time.time() - start_time
            
            start_time = time.time()
            train_cost_focus, train_acc_focus, penalty_focus = train_epoch(data['X_train'], data['y_train'], train_focus)
            time_spent_focus = time.time() - start_time
            
            tst_acc_dense, acc_dense = eval_epoch(data['X_test'], data['y_test'], eval_dense)
            tst_acc_focus, acc_focus = eval_epoch(data['X_test'], data['y_test'], eval_focus)
            
            focus_output = eval_output_focus(data['X_test'])
            focus_outputs.append(focus_output)
    
    
            if np.mod(n, PRINT_INTERVAL) == 0:
                print ("Epoch Dense {0}: T.cost {1}, Val {2}, Penalty: {4}, Time: {3}".format(n, train_cost_dense, acc_dense, time_spent_dense, penalty_dense))
                print ("Epoch Focus {0}: T.cost {1}, val {2}, Penalty: {4}, Time: {3}".format(n, train_cost_focus, acc_focus, time_spent_focus, penalty_focus))
                
            if np.mod(n,RECORD_INTERVAL)==0: 
                
                mus.append(get_mu())
                sis.append(get_si())
                foci.append(get_foci()*get_w())
                scalers.append(get_scaler())
                w_change.append(get_w())
                costs_dense.append(train_cost_dense)
                costs_focus.append(train_cost_focus)
            
                costs_tst_dense.append(tst_acc_dense)
                costs_tst_focus.append(tst_acc_focus)
            
                accs_focus.append(acc_focus)
                accs_dense.append(acc_dense)
                dense_weights.append(dense_mp.get_value())
    
    except KeyboardInterrupt:
        pass

    acc_focus_np = np.array(accs_focus)
    acc_dense_np = np.array(accs_dense)
       
    best_result = [np.max(np.array(acc_focus_np[0:])), 
                   np.argmax(np.array(acc_focus_np[0:])),
                   np.max(np.array(acc_dense_np[1:])), 
                   np.argmax(np.array(acc_dense_np[1:]))]
    best_results.append(best_result)
    all_results[rep,:,0] = np.asarray(acc_focus_np)
    all_results[rep,:,1] = np.asarray(acc_dense_np)
    print ("Repeat ", rep," focus:",best_result[0],best_result[1])
    print ("Repeat ", rep,"FNN   :",best_result[2],best_result[3])
    

# In[10]: Summarize and log results
logfile = logdir+"/accs_" + timestr + ".npz"
best_results_a =  np.asarray(best_results)
best_results_mn = np.mean(best_results_a,axis=0)
best_results_std= np.std(best_results_a,axis=0)
best_results_win= np.sum(best_results_a[:,0]>best_results_a[:,2])
best_results_eq= np.sum(best_results_a[:,0]==best_results_a[:,2])
best_focus_epoch = best_results_a[:,1]
np.savez(logfile,(all_results,best_results_a,mus,sis,foci,costs_dense,costs_focus,costs_tst_dense,costs_tst_focus))
print("best_results: ", best_results_a," means ",best_results_mn, " std ",best_results_std," wins: ",best_results_win," == ", best_results_eq)
s=input()
# In[11]: Plot results    

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#plt.style.use('classic')
plt.figure(figsize=(4,4))  
for r in range(N_REPEATS):
    plt.plot(all_results[r,:,1],all_results[r,:,0],'.')
plt.plot([0.0,1.0],[0,1.0],'--', color='k', linewidth=1.2)
#plt.arrow(0.3,0.78,0.1,0.1,-)
plt.annotate("Epoch", xy=(0.6, 0.9), xytext=(0.4, 0.82),
             arrowprops=dict(arrowstyle="->",color="black"))
plt.xlabel("Dense Acc %")
plt.ylabel("Focused Acc %")
plt.xticks([0,.2,.4,.6,.8])
plt.yticks([0,.2,.4,.6,.8])
plt.grid(True)
save_fig("focus_acc_vs_dense")
plt.show()

    


# In[13]:

focus_outputs_np = np.array(focus_output).reshape((-1, HIDDEN_COUNT))
for i in range(1):
    x = focus_outputs_np[:, i]
    n, bins, patches = plt.hist(x, 50, normed=0, facecolor='green', alpha=0.75)

    plt.xlabel('Activation Distribution')
    plt.ylabel('Count')
    plt.title(r'$\mathrm{Histogram\ of\ focus\ Neuron:}\ '+ str(i) +'$')
    plt.grid(True)
    plt.show()

# In[14]:

plt.figure(figsize=(4,4))
plt.title(u"Training Set")
plt.plot(np.array(costs_focus), linestyle='--',color='r', linewidth=2)
plt.plot(np.array(costs_dense), linestyle='--', color='g', linewidth=2)
plt.plot(np.array(costs_tst_focus), color='r', linewidth=4)
plt.plot(np.array(costs_tst_dense), color='g', linewidth=4)

#plt.yscale('log')
#plt.grid(True, which='both')
plt.legend([u'Focused', u'Dense'])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Error \%')
plt.xlim(0,len(costs_focus))
save_fig('focus_train')
plt.show()

plt.figure(figsize=(4,4))
plt.title(u'Validation Set')
plt.plot(np.array(costs_tst_focus), color='r', linewidth=4)
plt.plot(np.array(costs_tst_dense), color='g', linewidth=4)
#plt.yscale('log')
plt.legend([u'Focused', u'Dense'])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Error \%')
save_fig('focus_validate')
plt.show()


acc_focus_np = np.array(accs_focus)
acc_dense_np = np.array(accs_dense)
plt.figure(figsize=(4,4))
#plt.title(u'Doğrulama Kümesi Doğruluk')
plt.plot(np.array(acc_focus_np), color='r', linewidth=4, alpha=0.5)
plt.plot(np.array(acc_dense_np), color='g', linewidth=4, alpha=0.5)
#plt.legend([u'Odaklanan Nöron Katmanlı Sinir Ağı', u'Tümden Bağlı Katmanli Yapay Sinir Ağı'])
plt.legend([u'Focused', u'Dense'])
save_fig('focus_acc')
plt.xlabel('Epok')
plt.ylabel(u'Doğruluk Oranı')
plt.grid(True)
plt.show()


# In[15]:

print ("focus:",np.max(np.array(acc_focus_np[0:])), np.argmax(np.array(acc_focus_np[0:])))
print ("FNN   :",np.max(np.array(acc_dense_np[1:])), np.argmax(np.array(acc_dense_np[1:])))

# In[16]:

import itertools
markers = itertools.cycle((',', '+', '.', '^', '*')) 
styler = itertools.cycle(('-', '--', '-.')) 

mu = np.array(mus)
print (mu.shape)

plt.figure(figsize=(4, 4))
plt.title(r'Foci center change')
plt.xlabel('Epoch')
plt.ylabel(r'$\mu$')
plt.ylim([0.2,0.8])

#plt.title(r'$\mu$' + u' Değisimi')
#plt.xlabel('Epok')
#plt.ylabel(r'$\mu$')

plt.plot(mu[:, 0] , marker='.')
for x in range(mu.shape[1]):
    plt.plot(mu[0:-1, x], marker='.')
plt.grid(True)
#plt.xlim((-100,NUM_EPOCHS))
#plt.xscale('log')
save_fig('mu_change')
plt.show()


# In[17]:


si = np.array(sis)

plt.figure(figsize=(4, 4))
plt.title(r'$\sigma$' + u' Change')


plt.xlabel('Epoch')
plt.ylabel(r'$\sigma$')

#plt.plot(mu[:, 0] * 16, marker='o')
for x in range(si.shape[1]):
    plt.plot(si[0:-1, x], marker='.')
plt.grid(True)
#plt.xlim((-100,NUM_EPOCHS))
#plt.xscale('log')
save_fig('si_change')
plt.show()


# In[18]:
from focusing import U_numeric as U
  

# Plot Gaussians
mu = np.array(mus).squeeze()
si = np.array(sis)
scaler = INIT_SCALER

subset_neurons = HIDDEN_COUNT
if subset_neurons>10:
    subset_neurons = 10   
n_set = np.arange(0,HIDDEN_COUNT,HIDDEN_COUNT//subset_neurons)

mu_initial = mu[0, :]
plot_epoch = int(best_focus_epoch[0])
mu_final = mu[plot_epoch, :]
si_initial = si[0, :] # (8 / 16) / (np.repeat(np.sqrt(16 / (8 * 1.0)), 8))
si_final = si[plot_epoch, :]

# Print Initial Gaussians
idxs = np.linspace(0, 1.0, n_features)
ex_init = U(idxs, mu_initial, si_initial, scaler)
#ex += (ex > 0.1)

fig=plt.figure(figsize=(4,3))
fig.set_tight_layout(True)
plt.title('Focus Initial')

#for i in range(ex.shape[0]):
plt.plot(np.repeat(idxs[:,np.newaxis],n_set.shape[0],axis=1), ex_init[n_set, :].T,'-')  # + idxs to see the overlapping gaussians
plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('Normalized Index')
plt.xlim([0.0,1.0])
fig.savefig('figures/foci_initial.png')

plt.show()


# PAPER FIGURES HERE...
ex_final = U(idxs, mu_final, si_final,scaler)

#for i in n_set:
fig=plt.figure(figsize=(4,3))
line_init =plt.plot(np.repeat(idxs[:,np.newaxis],n_set.shape[0],axis=1), ex_init[n_set, :].T,':')  # + idxs to see the overlapping gaussians
line_after=plt.plot(np.repeat(idxs[:,np.newaxis],n_set.shape[0],axis=1), ex_final[n_set, :].T,'-', linewidth=2)  # + idxs to see the overlapping gaussians
plt.grid(True)
plt.title(u'Focus Change')
plt.ylabel(u'Magnitude')
plt.xlabel(u'Normalized Index')
plt.xlim([0.0,1.0])
#plt.ylim([0.0,n_features])
plt.legend((line_init[0], line_after[0]),['initial', 'Epoch 250'])
fig.set_tight_layout(True)
fig.savefig('figures/foci_init_final.png')



fig=plt.figure(figsize=(4,3))
#fig.set_tight_layout(True)
plt.plot(mu[0], si[0],marker='s',markersize='8', linestyle='None', color='gray')  # + idxs to see the overlapping gaussians

#for i in n_set:
plt.plot(mu[:,n_set], si[:,n_set])  # + idxs to see the overlapping gaussians
#plt.plot(mu[0], si[0],marker='>',markersize='8', linestyle='None')  # + idxs to see the overlapping gaussian
plt.plot(mu[-1], si[-1],marker='>',markersize='8', linestyle='None')  # + idxs to see the overlapping gaussianss
plt.grid(True)
plt.ylabel(r'$ \sigma $')
plt.xlabel(r'$ \mu $')
plt.xlim([0.1,0.9])
plt.title(r'Focus trajectory')
fig.set_tight_layout(True)
fig.savefig('figures/foci_shift.png')


# In[19]:



weights_dense = np.array(dense_weights)
#print weights_dense.shape
#for i in range(weights_dense.shape[2]):
#    plt.figure(figsize=(8,5))
#    plt.title('Weight For Neuron' + str(i) + ' in dense-1 MLP')
#    plt.plot(weights_dense[0,:, i], color='black',marker='*')
#    plt.plot(weights_dense[-1,:, i], marker='o')
#    plt.grid(True)
#    plt.show()


# In[23]:


#weights = np.array(w_change)
#print weights.shape
#for i in range(weights.shape[2]):
#    plt.figure(figsize=(8,5))
#    plt.title('Weight For Focusing Neuron' + str(i) + ' in focus Layer')
#    plt.plot(weights[0,:, i], color='black',marker='*')
#    plt.plot(weights[-1,:, i], marker='o')
#    plt.grid(True)
#    plt.show()
#


weights_dense = np.array(dense_weights)

weights = np.array(w_change)



fig = plt.figure(figsize=(6,4))
plt.grid(True)
plt.title('Weight Change for Focus and FNN' + str(i) + ' in focus Layer')
c1 = itertools.cycle(('r', 'g', 'b', 'm', 'c')) 
c2 = itertools.cycle(('r', 'g', 'b', 'm', 'c')) 
leglist =[]
for i in range(weights_dense.shape[1]):
#for i in range(2):
    #mn2 = weights[0,i, 1]
    #plt.plot(weights[:, i, 1]-mn2, color=next(c1),marker='.', markersize='3')
    #leglist.append('fw'+str(i))
    n = 0
    mn1 = weights_dense[0,i, n]
    plt.plot(weights_dense[:, i, n], color=next(c2), markersize='3',linestyle='-.')
    
    leglist.append('dw'+str(i))
    
#plt.legend(leglist)
#plt.xscale('log')
#plt.yscale('log')

# In[23]:
#plt.plot(weights[:,:, 1], color='red',marker='p')
#plt.plot(weights_dense[:,:, 1], color='green',marker='.')

#plt.show()
# =============================================================================
# fig.savefig('figures/weights_change.png')
# plt.figure(figsize=(8,5))
# plt.plot(np.asarray(w_change)[:,:,0])
# plt.show()
# 
# e = np.arange(0,weights.shape[1])
# ix = 1
# 
# =============================================================================
# plotting weight absolute weight changes around initial weight value

#plt.figure()

fig, axs = plt.subplots(2, 1, figsize=(6,4),sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)
neurons =  [0,1]
for k in range(len(neurons)):
# Plot each graph, and manually set the y tick values
    ix = neurons[k]
    mn1 = weights_dense[0,:, ix]
    # total abs diff
    st1 = np.sum(np.abs(np.diff(weights_dense[:,:, ix],axis=0)),axis=0)
    mn2 = weights[0,:, ix]
    st2 = np.sum(np.abs(np.diff(weights[:,:, ix],axis=0)),axis=0)

    #axs[k].errorbar(e,mn1,yerr=st1,color='green',marker='.', linestyle='None')
    axs[0].violinplot(list((weights[:,:, ix]-mn2).T),showmeans=False,showmedians=True)
    axs[1].violinplot(list((weights_dense[:,:, ix]-mn1).T),showmeans=False,showmedians=True)

    #axs[k].errorbar(e+.01,mn2,yerr=st2,color='red',marker='+', linestyle='None')
    
    #axs[k].plot(e, ex.T[:,ix]*0.05, linewidth=1.0, linestyle='--')
    #axs[k].set_yticks(np.arange(-0.25, .50, 0.25))
    #legend = axs[k].legend(['N'+str(ix),'F'+str(ix)],loc=2,edgecolor=[0,0,0,0], markerfirst=False)
    #legend.get_frame().set_facecolor('#BBBBBB')
    #legend.draw_frame(True)

fig.savefig('figures/weigth_updates.png')

from focusing import U_numeric

# In[]
#paper_fig_settings

#plt.figure(figsize=(1,1))

fig, axs = plt.subplots(2, 1, sharex=True,figsize=(4,3))

fig.set_tight_layout(True)
# Remove horizontal space between axes

#plt.subplots_adjust(bottom=0.15, hspace=-1.05)
fig.subplots_adjust(hspace=0.00)

#neurons =  np.arange0,29)
neurons =n_set[0:6]
epoc =0
fay = U_numeric(np.linspace(0,1,weights.shape[1]),mu[epoc,neurons],si[epoc,neurons],1)
w = weights[epoc,:,neurons]
markers1 = itertools.cycle((',', '+', '.', '^', '*')) 
for t in neurons:
    axs[0].plot(idxs,(fay.T*w.T)[:,t], marker= next(markers1), markersize=3,linestyle=next(styler), alpha=0.8)

#axs[0].plot(idxs,fay.T*w.T,marker=None, linestyle='-',alpha=0.9)

axs[0].set_title('Focused weights')

#axs[0].set_ylim([-.05, .05])
#axs[0].set_yticks([-.05, .05])
axs[0].grid()


#plt.figure(figsize=(2,4))
#neurons =  np.arange0,29)
neurons =n_set[0:6]

epoc =NUM_EPOCHS
fay = U_numeric(np.linspace(0,1,weights.shape[1]),mu[epoc,neurons],si[epoc,neurons],1)
w = weights[epoc,:,neurons]
markers2 = itertools.cycle((',', '+', '.', '^', '*')) 
for t in neurons:
    axs[1].plot(idxs,(fay.T*w.T)[:,t], marker= next(markers2), markersize=3,linestyle=next(styler), alpha=0.8)
#axs[1].set_title('Focused weights')
#plt.ylim([-0.005, 0.005])
#axs[1].text(0.2,0.9, 'Epoch 10')



axs[1].set_xlabel('Normalized index')
axs[1].grid()
#axs[1].set_ylim([-.025, .025])
#plt.ylabel('Magnitude')
#axs[1].set_ylabel_coord(50,0)
plt.text(0.002, 3.20, 'Initial', bbox=dict(facecolor='green', alpha=0.2 ))
plt.text(0.002, 0.50, 'Epoch '+str(epoc), bbox=dict(facecolor='green', alpha=0.2))
axs[0].set_ylabel('Magnitude')
axs[0].set_yticks([-1,0,1])
axs[0].set_ylim([-1,1])
axs[0].yaxis.set_label_position('right')
axs[1].set_ylabel('Magnitude')
axs[1].yaxis.set_label_position('right')
axs[1].set_yticks([-1,0,1])
axs[1].set_ylim([-1,1])
#axs[0].yaxis.set_label_coords(1.
#fig.set_tight_layout(True)

fig.savefig('figures/focused_weights_before_after.png')
#save_fig('focused_weights_before_after')






#
#neurons =[0]
#for t in range(epoc):
#    w = weights[t,:,neurons]
#    fay = U_numeric(np.linspace(0,1,weights.shape[1]),mu[t,neurons],si[t,neurons],1)
#    if t == 0 :
#        c='r'
#        mark='s'
#    elif t== epoc-1:
#        c='g'
#        mark='^'
#    else:
#        c='black'
#        mark='.'
#    plt.plot((fay.T*w.T)[:].T, color=c, marker=mark, linestyle='--', alpha=0.2)
#plt.title('focused weights change for epochs')
#plt.show()


