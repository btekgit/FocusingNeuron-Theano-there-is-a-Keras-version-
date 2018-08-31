owa'''
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

# -*- coding: utf-8 -*-

#from __future__ import unicode_literals
# In[1]:

import os
os.environ['THEANO_FLAGS']='device=cuda,floatX=float32,preallocate=0.1'
os.environ['MKL_THREADING_LAYER']='GNU'
import time
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
import numpy as np
from focusing import FocusedLayer1D
from sklearn.datasets.samples_generator import make_blobs, make_classification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from lasagne.updates import get_or_compute_grads, apply_momentum, sgd
from lasagne.updates import momentum, adam, adadelta
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from sklearn.preprocessing import StandardScaler


# In[2]:
# To plot pretty figures
#
#plt.style.use('seaborn-paper')
plt.rc('font', family='sans-serif', weight='bold', size='16')
#matplotlib.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('axes',titlesize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('axes', labelsize=16)
""
# width as measured in inkscape
width = 3.45
height = width / 1.618
plt.rcParams["figure.figsize"] = (width,height)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "bilisim"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "outputs", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=144)

from datetime import datetime
now = datetime.now()
timestr = now.strftime("%Y%m%d-%H%M%S")
logdir = "focusing_syntethic/" + timestr + "/"

# In[3]:


# Test Settings
NUM_EPOCHS = 2500
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MOMENTUM = 0.9
USE_PENALTY = False

# Dataset Settings
CLASSES = 8
FEATURES = 8
SAMPLES = 2000
DUMMY_POINTS = int(FEATURES * .5) # Percentage
DUMMY_MULT = 1.5
TOTAL_FEATURES = FEATURES+(DUMMY_POINTS*2)

# Beamer Settings
BEAMER_COUNT = 12 # WORKS WHEN IT IS EQUAL TO FEATURES
#LR_MU = np.float32((0.5 / TOTAL_FEATURES) * .1, dtype='float32'); 
LR_MU = np.float32((LEARNING_RATE*0.2))
UPDATE_MU = True
#LR_SI = np.float32((0.5 / TOTAL_FEATURES) * .1, dtype='float32'); 
LR_SI = np.float32((LEARNING_RATE*0.1))
print ("LR W: ", LEARNING_RATE, "LR_MU", LR_MU, "LR_SI", LR_SI)
UPDATE_SI = True
LR_SCALER = 0.01; UPDATE_SCAlER = False
INIT_SI = 0.20# 0.20 best
INIT_SCALER = 1.0
WITH_WEIGHTS = True
NORM = True
PRUNE = False
MU_INIT_DIRECTION = 'spread' # 'middle' Or sth else

# Additional Settings
RANDSEED = 42
lasagne.random.set_rng(np.random.RandomState(RANDSEED))  # Set random state so we can investigate results
np.random.seed(RANDSEED)
theano.config.exception_verbosity = 'high'

# In[5]:


def load_blob(classes=3, features=10, samples=10, random_state=0):
    samples += int(samples * .30)  # 30% for test
    #xs, ys = make_blobs(n_samples=samples, centers=classes,
    #                    n_features=features, random_state=random_state)
    
    xs, ys = make_classification(n_samples=samples, n_features=features, n_informative=features, n_redundant=0, n_classes=classes, n_clusters_per_class=1)
    
#    xs -= xs.mean()
#    xs /= xs.var()
    scaler = StandardScaler()
    xs= scaler.fit_transform(xs)
    
    ys = np.float32(ys)

    X_train, X_test, y_train, y_test = train_test_split(xs, ys.squeeze(), test_size=0.3, random_state=42)
    
    # Concat dummy points for train
    dummy_points_train = np.random.rand(X_train.shape[0], DUMMY_POINTS) * DUMMY_MULT
    X_train = np.concatenate((dummy_points_train, X_train), axis=1)
    X_train = np.concatenate((X_train, dummy_points_train), axis=1)
    
    # Concat dummy points for test
    dummy_points_test = np.random.rand(X_test.shape[0], DUMMY_POINTS) * DUMMY_MULT
    X_test = np.concatenate((dummy_points_test, X_test), axis=1)
    X_test = np.concatenate((X_test, dummy_points_test), axis=1)
    
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

data = load_blob(CLASSES, FEATURES, SAMPLES, RANDSEED)


# In[6]:


print(data['X_train'].shape)

n_features = data['X_train'].shape[1]
X = data['X_test']
Y = data['y_test']
plt.scatter(X[:,1], X[:,2], marker='o',c=Y)
plt.show()

# In[7]:


def build_model(input_feas, classes):
    # Initers, Layers
    ini = lasagne.init.GlorotUniform()
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    softmax = lasagne.nonlinearities.softmax
    tanh = lasagne.nonlinearities.tanh
    
    # Input Layer
    l_in = lasagne.layers.InputLayer(shape=(None, input_feas))
    
    # Denses
    l_dense1 = lasagne.layers.DenseLayer(
            l_in, num_units=BEAMER_COUNT, 
            nonlinearity=lasagne.nonlinearities.linear, 
            W=ini, name="dense1", b=None)
    
    
    l_bn = lasagne.layers.NonlinearityLayer(lasagne.layers.BatchNormLayer(l_dense1), nonlinearity=relu)

    #l_dense2 = lasagne.layers.DenseLayer(l_dense1, num_units=4, nonlinearity=lasagne.nonlinearities.tanh, W=ini, name='dense2')
    
    l_drop1 = lasagne.layers.dropout(l_bn, p=0.1)
    
    # Output Layer
    l_out = lasagne.layers.DenseLayer(l_drop1, num_units=classes, nonlinearity=softmax, W=ini, name='output')
    
    
    penalty = (l2(l_dense1.W)*1e-4)+(l1(l_dense1.W)*1e-6) +(l2(l_out.W)*1e-3)
    if not USE_PENALTY:
        penalty = penalty*0
    
    #penalty = penalty*0
    #penalty = (l2(l_dense1.W)*1e-30)#(l2(l_dense1.W)*1e-3)+(l1(l_dense1.W)*1e-6) +(l2(l_out.W)*1e-3)
    
    return l_out, penalty

lasagne.random.set_rng(np.random.RandomState(RANDSEED))  # Set random state so we can investigate results
np.random.seed(RANDSEED)
model_dense, penalty_dense = build_model(n_features, CLASSES)
mp_dense = lasagne.layers.get_all_params(model_dense, trainable=True)
l_dense = next(l for l in lasagne.layers.get_all_layers(model_dense) if l.name is "dense1")
dense_mp = l_dense.W


# In[8]:


def build_model_beamer(input_feas, classes):
    # Initers, Layers
    ini = lasagne.init.HeUniform()
    relu = lasagne.nonlinearities.rectify
    elu = lasagne.nonlinearities.elu
    linear = lasagne.nonlinearities.linear
    softmax = lasagne.nonlinearities.softmax
    tanh = lasagne.nonlinearities.tanh
    
    # Input Layer
    
    l_in = lasagne.layers.InputLayer(shape=(None, input_feas))
    
    l_beamer1 = FocusedLayer1D(l_in, num_units=BEAMER_COUNT, 
                              nonlinearity=linear, name='beamer1',
                              trainMus=UPDATE_MU, 
                              trainSis=UPDATE_SI, 
                              initMu=MU_INIT_DIRECTION, 
                              W=ini, withWeights=WITH_WEIGHTS, 
                              bias=lasagne.init.Constant(0.0), 
                              initSigma=INIT_SI, 
                             scaler=INIT_SCALER, weight_gain=1.0, 
                             trainScaler=UPDATE_SCAlER, trainWs=True)    
    
    l_bn = lasagne.layers.NonlinearityLayer(
            lasagne.layers.BatchNormLayer(l_beamer1), nonlinearity=elu)
    # if you close BATCHNORM weights get LARGE
    #l_bn = lasagne.layers.NonlinearityLayer(l_beamer1, nonlinearity=tanh)
    
    # Denses
    #l_dense1 = lasagne.layers.DenseLayer(l_beamer1, num_units=20, nonlinearity=lasagne.nonlinearities.tanh, W=ini,name='dense1')
    
    l_drop1 = lasagne.layers.dropout(l_bn, p=0.1)
    
    # Output
    l_out = lasagne.layers.DenseLayer(l_drop1, num_units=classes, 
                                      nonlinearity=softmax, W=ini, name='output')
    
    
    penalty = l2(l_beamer1.W)*1e-4+(l1(l_beamer1.W)*1e-6) +l2(l_out.W)*1e-3+l2(l_beamer1.si)*1e-2
    if not USE_PENALTY:
        penalty = penalty*0
    
    
    return l_out, penalty

# Reset randomseed
lasagne.random.set_rng(np.random.RandomState(RANDSEED))  # Set random state so we can investigate results
np.random.seed(RANDSEED)

# Init model
model_beamer, penalty_beamer = build_model_beamer(n_features, CLASSES)
mp_beamer = lasagne.layers.get_all_params(model_beamer, trainable=True)
l_dynamic = next(l for l in lasagne.layers.get_all_layers(model_beamer) if l.name is 'beamer1')
#l_dense1 = next(l for l in lasagne.layers.get_all_layers(model_beamer) if l.name is 'dense1')

get_si = lambda: l_dynamic.si.get_value()
get_mu = lambda: l_dynamic.mu.get_value()
get_w = lambda: l_dynamic.W.get_value()
set_w = lambda value: l_dynamic.W.set_value(value)
get_scaler = lambda: l_dynamic.scaler.get_value()
set_scaler = lambda value: l_dynamic.scaler.set_value(value)



# In[9]:


# This function can give different learning rates to beamer layers
def sgdWithLrs(loss_or_grads, params, learning_rate, mu_lr, si_lr, scaler_lr):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        # import pdb; pdb.set_trace()
        if param.name == 'beamer1.mu' or param.name == 'beamer2.mu':
            updates[param] = param - mu_lr * grad
            print (param, grad, mu_lr)
        elif param.name == 'beamer1.si' or param.name == 'beamer2.si':
            updates[param] = param - si_lr * grad
            print (param, grad, si_lr)
        elif param.name == 'beamer1.scaler' or param.name == 'beamer2.scaler':
            updates[param] = param - scaler_lr * grad
            print (param, grad, scaler_lr)
        #elif param.name == 'beamer1.W' or param.name == 'beamer2.W':
        #    updates[param] = param - (np.float32(mu_lr) * grad)
        #    print (param, grad, mu_lr)
        else:
            updates[param] = param - learning_rate * grad
            print (param, grad, learning_rate)
    return apply_momentum(updates, momentum=MOMENTUM)


# In[10]:


# Compile train and eval functions
def build_functions(using_model, using_parameters, penalty):
    X = T.fmatrix()
    y = T.ivector()

    # training output
    output_train = lasagne.layers.get_output(using_model, X, deterministic=False)

    # evaluation output. Also includes output of transform for plotting
    output_eval = lasagne.layers.get_output(using_model, X, deterministic=True)
    
    cost = T.mean(lasagne.objectives.categorical_crossentropy(output_train, y)) + penalty # Regularization
  
    
    params = using_parameters
    #updates = adam(cost, params, learning_rate=0.001)
    #updates = momentum(cost, params, learning_rate=0.01, momentum=0.5)
    
    updates = sgdWithLrs(cost, using_parameters, LEARNING_RATE, LR_MU, LR_SI, LR_SCALER)
    
    test_acc = T.mean(T.eq(T.argmax(output_eval, axis=1), y), dtype=theano.config.floatX)

    eval = theano.function([X, y], [cost, test_acc], allow_input_downcast=True)
    train = theano.function([X, y], [cost, output_train, penalty], updates=updates, allow_input_downcast=True)
    
    return train, eval

train_beamer, eval_beamer = build_functions(model_beamer, mp_beamer, penalty_beamer)
train_dense, eval_dense = build_functions(model_dense, mp_dense, penalty_dense)

X = T.fmatrix()
output_beamer = lasagne.layers.get_output(l_dynamic, X, deterministic=True)
eval_output_beamer = theano.function([X], [output_beamer], allow_input_downcast=True)

output_dense = lasagne.layers.get_output(l_dense, X, deterministic=True)
eval_output_dense = theano.function([X], [output_dense], allow_input_downcast=True)


# In[11]:

# this part 
def weight_adjustment_batch(X, output_func,getter,setter):
    print ("LSUV initialization")
    n_samples = 128
    random_ix = np.random.permutation(X.shape[0])
    X_batch = X[random_ix[0:n_samples]]
    print ("X var: ",np.var(X_batch))
    Y = output_func(X_batch)
    
    print ("Y shape:", Y[0].shape)
    W = getter()
    variance = np.mean(np.var(Y[0], axis=0))
    needed_variance = 1.0
    margin = 0.02
    iteration = 0
    #print "var: ",np.var(Y[0], axis=0)
    print (" var mean: ",variance, "iter: ",iteration)
    print ("Y var: ",np.var(Y[0], axis=0), "W var ", np.var(W, axis=0))
    while abs(needed_variance - variance) > margin and iteration<5:
        if np.abs(np.sqrt(variance)) < 1e-7:
        # avoid zero division
            break
    
        weights = W
       
        #print "norming with:",np.var(Y,axis=0)
        print (" shape", (np.var(Y[0],axis=0)).shape)
        weights /= np.sqrt(np.var(Y[0], axis=0)) 
       
        #weights /= np.sqrt(variance) / np.sqrt(needed_variance)
        
        setter(np.copy(W))
        Y = output_func(X)
        variance = np.mean(np.var(Y[0], axis=0))
        iteration = iteration + 1
        print ("Y var: ",np.var(Y[0], axis=0), "W var ", np.var(W, axis=0))
        print ("var mean: ",variance, "iter: ",iteration)
    
    print ("Y var: ",np.var(Y[0], axis=0), "W var ", np.var(W, axis=0))
    
#




#weight_adjustment_batch(data['X_train'], eval_output_dense, 
#                        getter=l_dense.W.get_value,
#                        setter=lambda value: l_dense.W.set_value(value))    
use_lsuv_adjustment = False
if use_lsuv_adjustment:
    print ("Weight mean before ", np.var(get_w()))
    weight_adjustment_batch(data['X_train'], eval_output_beamer, getter=get_w,setter=set_w) 
    print ("Weight mean before ", np.var(get_w()))

    print ("Gains before ", get_scaler())
    weight_adjustment_batch(data['X_train'], eval_output_beamer, getter=get_scaler,setter=set_scaler) 
    print ("Gains after ", get_scaler())

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


# # Training Happens Beyond This

# In[12]:


np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)

total_time = 0
costs_dense, costs_beamer = [], []
costs_tst_dense, costs_tst_beamer = [], []
accs_dense, accs_beamer = [], []
beamer_outputs = []
dense_weights = []
mus = []
sis = []
w_change = []
scalers = []
try:
    for n in range(NUM_EPOCHS):
        if n == 0:
            mus.append(get_mu())
            sis.append(get_si())
            mus.append(get_mu())
            sis.append(get_si())

        start_time = time.time()
        train_cost_dense, train_acc_dense, penalty_dense = train_epoch(data['X_train'], data['y_train'], train_dense)
        time_spent_dense = time.time() - start_time
        
        start_time = time.time()
        train_cost_beamer, train_acc_beamer, penalty_beamer = train_epoch(data['X_train'], data['y_train'], train_beamer)
        time_spent_beamer = time.time() - start_time
        
        tst_acc_dense, acc_dense = eval_epoch(data['X_test'], data['y_test'], eval_dense)
        tst_acc_beamer, acc_beamer = eval_epoch(data['X_test'], data['y_test'], eval_beamer)
        
        beamer_output = eval_output_beamer(data['X_test'])
        beamer_outputs.append(beamer_output)

        costs_dense.append(train_cost_dense)
        costs_beamer.append(train_cost_beamer)
        
        costs_tst_dense.append(tst_acc_dense)
        costs_tst_beamer.append(tst_acc_beamer)
        
        accs_beamer.append(acc_beamer)
        accs_dense.append(acc_dense)
        
        w_change.append(get_w())
        
        mus.append(get_mu())
        sis.append(get_si())
        scalers.append(get_scaler())
        
        dense_weights.append(dense_mp.get_value())

        if np.mod(n, 10) == 0:
            print ("Epoch Dense {0}: T.cost {1}, Val {2}, Penalty: {4}, Time: {3}".format(n, train_cost_dense, acc_dense, time_spent_dense, penalty_dense))
            print ("Epoch Beamer {0}: T.cost {1}, val {2}, Peanlty: {4}, Time: {3}".format(n, train_cost_beamer, acc_beamer, time_spent_beamer, penalty_beamer))

except KeyboardInterrupt:
    pass

acc_beamer_np = np.array(accs_beamer)
acc_dense_np = np.array(accs_dense)
print ("Beamer:",np.max(np.array(acc_beamer_np[0:])), np.argmax(np.array(acc_beamer_np[0:])))
print ("FNN   :",np.max(np.array(acc_dense_np[1:])), np.argmax(np.array(acc_dense_np[1:])))
logfile = "log_syntethic/accs_" + timestr + ".npz"
np.savez(logfile,(acc_beamer_np,acc_dense_np,mus,sis,costs_dense,costs_beamer,costs_tst_dense,costs_tst_beamer))

# In[13]:

beamer_outputs_np = np.array(beamer_output).reshape((-1, BEAMER_COUNT))
for i in range(1):
    x = beamer_outputs_np[:, i]
    n, bins, patches = plt.hist(x, 50, normed=0, facecolor='green', alpha=0.75)

    plt.xlabel('Activation Distribution')
    plt.ylabel('Count')
    plt.title(r'$\mathrm{Histogram\ of\ Beamer\ Neuron:}\ '+ str(i) +'$')
    plt.grid(True)
    plt.show()


# In[14]:

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.figure(figsize=(8,6))
plt.title(u"Training Set")
plt.plot(np.array(costs_beamer), linestyle='--',color='r', linewidth=2)
plt.plot(np.array(costs_dense), linestyle='--', color='g', linewidth=2)
plt.plot(np.array(costs_tst_beamer), color='r', linewidth=4)
plt.plot(np.array(costs_tst_dense), color='g', linewidth=4)

#plt.yscale('log')
#plt.grid(True, which='both')
plt.legend([u'FocusNN', u'FullNN'])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Error \%')
plt.xlim(0,100)
save_fig('beamer_train')
plt.show()

plt.figure(figsize=(8,6))
plt.title(u'Validation Set')
plt.plot(np.array(costs_tst_beamer), color='r', linewidth=4)
plt.plot(np.array(costs_tst_dense), color='g', linewidth=4)
#plt.yscale('log')
plt.legend([u'Focusing Layer NN', u'Fully Conn. Layer NN'])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Error \%')
save_fig('beamer_validate')
plt.show()


acc_beamer_np = np.array(accs_beamer)
acc_dense_np = np.array(accs_dense)
plt.figure(figsize=(8,6))
plt.title(u'Doğrulama Kümesi Doğruluk')
plt.plot(np.array(acc_beamer_np), color='r', linewidth=4, alpha=0.5)
plt.plot(np.array(acc_dense_np), color='g', linewidth=4, alpha=0.5)
plt.legend([u'Odaklanan Nöron Katmanlı Sinir Ağı', u'Tümden Bağlı Katmanli Yapay Sinir Ağı'])
save_fig('beamer_acc')
plt.xlabel('Epok')
plt.ylabel(u'Doğruluk Oranı')
plt.grid(True)
plt.show()


# In[15]:


print ("Beamer:",np.max(np.array(acc_beamer_np[0:])), np.argmax(np.array(acc_beamer_np[0:])))
print ("FNN   :",np.max(np.array(acc_dense_np[1:])), np.argmax(np.array(acc_dense_np[1:])))


# In[16]:


mu = np.array(mus)
print (mu.shape)

plt.figure(figsize=(8, 6))
plt.title(r'$\mu$' + u' Değisimi')
plt.xlabel('Epok')
plt.ylabel(r'$\mu$')

#plt.plot(mu[:, 0] * 16, marker='o')
for x in range(mu.shape[1]):
    plt.plot(mu[0:-1, x], marker='o')
plt.grid(True)
#plt.xlim((-100,NUM_EPOCHS))
plt.xscale('log')
save_fig('mu_change')
plt.show()


# In[17]:


si = np.array(sis)

plt.figure(figsize=(8, 6))
plt.title(r'$\sigma$' + u' Değişimi')


plt.xlabel('Epok')
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


def U(idxs, mus, sis, scaler):
    up = (idxs - mus[:, np.newaxis]) ** 2
    down = (2 * (sis[:, np.newaxis] ** 2))
    ex = np.exp(-up / down)
    sums = np.sum(ex,axis=1)
    
    
    # current beamer normalizes each neuron to receive one full 
    ex /= sums[:,np.newaxis]
    
    #scaler = get_scaler()
    ex = ex*scaler
    
    
    return (ex)

# Plot Gaussians
mu = np.array(mus).squeeze()
si = np.array(sis)
scaler = n_features

mu_initial = mu[0, :]
mu_final = mu[-1, :]
si_initial = si[0, :] # (8 / 16) / (np.repeat(np.sqrt(16 / (8 * 1.0)), 8))
si_final = si[-1, :]

# Print Initial Gaussians
idxs = np.linspace(-0.1, 1.1, 128)
ex = U(idxs, mu_initial, si_initial, scaler)
#ex += (ex > 0.1)

plt.figure(figsize=(8,5))
plt.title(u'Odakların İlk Durumu')
for i in range(ex.shape[0]):
    plt.plot(idxs, ex[i, :],'-.')  # + idxs to see the overlapping gaussians
plt.grid(True)
plt.ylabel(u'Büyüklük')
plt.xlabel(u'Normalize İndis')
plt.show()
save_fig('gauss_initial')

#plot initial
plt.figure(figsize=(8,5))
#plt.plot(idxs, ex[0, :], marker='*', markersize=10)
# Print Final Gaussians
idxs = np.linspace(-0.1, 1.1, 128)
ex = U(idxs, mu_final, si_final,scaler)



for i in range(ex.shape[0]):
    ax = plt.plot(idxs, ex[i, :], linewidth=3.0)
#plt.semilogy()
#plt.ylim([0.1,110])
plt.grid(True)

plt.title(u'Odakların Son Durumu')
plt.ylabel(u'Büyüklük')
plt.xlabel(u'Normalize İndis')
save_fig('gauss_final')
#plt.legend(range(ex.shape[0]))
plt.show()


plt.figure(figsize=(8,5))
plt.title(u'Odak kayması')
plt.plot(mu[0], si[0],marker='o',markersize='12', linestyle='None')  # + idxs to see the overlapping gaussians
for i in range(ex.shape[0]):
    plt.plot(mu, si)  # + idxs to see the overlapping gaussians
#plt.plot(mu[0], si[0],marker='>',markersize='8', linestyle='None')  # + idxs to see the overlapping gaussians
plt.grid(True)
plt.ylabel(u'Si')
plt.xlabel(u'Mu')
plt.show()
save_fig('mu_sigma_change')


# In[19]:


#print np.sum(ex[0])


# In[20]:


#print np.array(scalers)


# In[21]:


#print np.array(w_change)


# # Expectation: Weights between 0-Random Points are nearly zero, to cancel useless information out

# In[22]:


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
#    plt.title('Weight For Focusing Neuron' + str(i) + ' in Beamer Layer')
#    plt.plot(weights[0,:, i], color='black',marker='*')
#    plt.plot(weights[-1,:, i], marker='o')
#    plt.grid(True)
#    plt.show()
#
weights = np.array(w_change)



plt.figure(figsize=(8,5))
plt.grid(True)
plt.title('Weight Change for Focus and FNN' + str(i) + ' in Beamer Layer')
plt.plot(weights[:,:, 0], color='red',marker='*')
plt.plot(weights_dense[:,:, 1], color='green',marker='*')
plt.plot(weights[:,:, 1], color='red',marker='p')
plt.plot(weights_dense[:,:, 1], color='green',marker='.')
save_fig('weight_change')
plt.show()

plt.figure(figsize=(8,5))
plt.plot(np.asarray(w_change)[:,:,4])
plt.show()
