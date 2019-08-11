#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:02:20 2018

@author: btek
"""

from __future__ import print_function

import h5py
from keras.callbacks import Callback
import keras.backend as K

def print_layer_names(model):
    for layer in model.layers:
        print(layer)
        print(layer.name)
        print(layer.name=='gauss')
        
def print_layer_weights(model):
    for layer in model.layers:
        print(layer)
        print(layer.name)
        g=layer.get_config()
        print(g)
        w = layer.get_weights()
        print(len(w))
        print(w)

def get_layer_weights(model,layer_name):
    
    for layer in model.layers:
        if (layer.name==layer_name):
            print("Layer: ", layer)
            print('name:',layer.name)
            g=layer.get_config()
            print(g)
            w = layer.get_weights()
            print(len(w))
            return w
        else:
            return None


def kdict(d,name_prefix=''):
    r = {}
    for k,v in d.items():
        r[k] = K.variable(v, name=name_prefix+str(k).replace(':','_'))
        return r     
        
def eval_Kdict(d):
    '''evaluates all variables in a dictionary'''
    l = [str(k)+':'+str(K.eval(v)) for k,v in d.items()]
    return l
        

class WeightHistory(Callback):

    def __init__(self, model, layername):
        self.batchlist=[]
        self.epochlist=[]
        self.sess = None
        self.warn = True
        self.model = model
        self.layername = layername
        
        print("Weight history set for: ", self.model.get_layer(self.layername))
        super(WeightHistory, self).__init__()
    
    def set_model(self, model):
        self.model = model
        print(self.model.summary())
    
    def on_train_begin(self, logs={}):
        self.batchlist = []
        self.epochlist = []
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

#    def on_batch_end(self, batch, logs={}):
##        gauss_layer = self.model.get_layer(self.layername)  
##        gauss_layer_var = gauss_layer.get_weights()
##        #warn = True
##        if len(self.batchlist)< 10000:
##            self.batchlist.append(gauss_layer_var[0])
    
    def on_epoch_begin(self, batch, logs={}):
        gauss_layer = self.model.get_layer(self.layername)
        gauss_layer_var = gauss_layer.get_weights()
        #print("yes called")
        #warn = True
        if len(self.epochlist)< 10000:
            self.epochlist.append(gauss_layer_var[0])
                
        
    def get_batchlist(self):
        return self.batchlist
    
    def get_epochlist(self):
        return self.epochlist
        

def dump_keras_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()
        

class RecordVariable(Callback):
        def __init__(self,name,var):
            self.layername = name
            self.varname = var
        
        def setVariableName(self,name, var):
            self.layername = name
            self.varname = var
        def on_train_begin(self, logs={}):
            self.record = []

        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))
            
        def on_epoch_end(self,epoch, logs={}):
            all_params = self.model.get_layer(self.layername)._trainable_weights
            all_weights = self.model.get_layer(self.layername).get_weights()
            
            for i,p in enumerate(all_params):
                #print(p.name)
                if (p.name.find(self.varname)>=0):
                    #print("recording", p.name)
                    self.record.append(all_weights[i])
                    
                    
class PrintLayerVariableStats(Callback):
        def __init__(self,name,var,stat_functions,stat_names):
            self.layername = name
            self.varname = var
            self.stat_list = stat_functions
            self.stat_names = stat_names
        
        def setVariableName(self,name, var):
            self.layername = name
            self.varname = var
        def on_train_begin(self, logs={}):
            all_params = self.model.get_layer(self.layername)._trainable_weights
            all_weights = self.model.get_layer(self.layername).get_weights()
            
            for i,p in enumerate(all_params):
                #print(p.name)
                if (p.name.find(self.varname)>=0):
                    stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                    print("Stats for", p.name, stat_str)

        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))
            
        def on_epoch_end(self, epoch, logs={}):
            all_params = all_weights = self.model.get_layer(self.layername)._trainable_weights
            all_weights = self.model.get_layer(self.layername).get_weights()
            
            for i,p in enumerate(all_params):
                #print(p.name)
                if (p.name.find(self.varname)>=0):
                    stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                    print("Stats for", p.name, stat_str)
                    

class PrintAnyVariable(Callback):
        
        def __init__(self,scope, varname):
            print("THIS DOES NOT WORK!!!!",)
            self.scope = scope
            self.varname = varname
            
        def setVariableName(self,scope, varname):
            self.scope = scope
            self.varname = varname
        def on_train_begin(self, logs={}):
            vs =[n.name for n in K.tf.get_default_graph().as_graph_def().node]
            print(vs)
            for v in vs:
                if v.name ==self.varname:
                    with K.get_session() as sess:
                        print(":", K.get_value(v))
                    #print(self.varname," ", K.get_value(v))
        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))
            
        def on_epoch_end(self, epoch, logs={}):
            g = K.tf.get_default_graph()
            v = g.get_tensor_by_name(self.varname)
            print(v)
            vs =[n for n in K.tf.get_default_graph().as_graph_def().node]
            for v in vs:
                if v.name ==self.varname:
                    print(v.name)
                    #K.tf.print(v)
                    #print(self.varname," ", K.get_value(v))


#def clip_variable_after_batch(Callback):
#    def __init__(self,name,var,min_val, max_val):
#            self.layername = name
#            self.varname = var
#            self.stat_list = stat_functions
#            self.stat_names = stat_names
#            self.min_val = K.variable(min_val)
#            self.max_val = K.variable(max_val)
#    
#    def on_batch_end(self,batch, logs={}):
#        all_params = self.model.get_layer(self.layername)._trainable_weights
#        for i,p in enumerate(all_params):
#                #print(p.name)
#                if (p.name.find(self.varname)>=0):
#                    #print("recording", p.name)
#                    self.record.append(all_weights[i])
#            x = keras.backend.clip(x, self.min_value, self.max_value)
#    
from keras.optimizers import Optimizer

from six.moves import zip


#from keras.utils.generic_utils  import serialize_keras_object
#from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces
    
class SGDwithLR(Optimizer):
    """Stochastic gradient descent optimizer with different LEARNING RATES
    CODED BY BTEK

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        LR: a dictionary of floats for, float >= 0
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr={'all':0.1}, momentum={'all':0.0}, decay={},
                 clips={}, decay_epochs=None,
                 nesterov=False, **kwargs):
        super(SGDwithLR, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            if 'all' not in lr.keys():
                print('adding LR for all elements')
                lr.setdefault('all',0.1)
            if isinstance(lr, (float,int)):  
                print('This SGD works with dictionaries')
                lr = {'all',lr}
                #print(lr)
            
            if  isinstance(momentum, (float,int)):  
                print('This SGD works with dictionaries')
                momentum = {'all',momentum}
                #print(lr)
                

            
            self.lr = kdict(lr,'lr_')
            #print("Learning rate: ", self.lr)
            self.momentum = kdict(momentum,'mom_')
            self.decay = kdict(decay,'dec_')
            self.clips = kdict(clips,'clips')
            self.clips_val = clips
            if decay_epochs is not None:
                self.decay_epochs=K.variable(decay_epochs, dtype='int64')
            else:
                self.decay_epochs=[]
                    
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        
        # first update the number of iterations
        self.updates = [K.update_add(self.iterations, 1)]
        #print("HEREERREERERERERERERE", self.iterations in self.decay_epochs)
#        print(self.lr)
#        print(self.momentum)
#        print(self.decay)
#        print(self.initial_decay)
        #ite_casted = K.cast(self.iterations, K.dtype(self.decay_epochs))
        #hit_decay_epoch = K.any(K.equal(ite_casted, self.decay_epochs))#K.any(ite_casted - self.decay_epochs)
        #print("HIT DECAY EPOCH", K.get_value(ite_casted), K.get_value(self.decay_epochs), K.get_value(hit_decay_epoch))
        if self.decay_epochs:
            ite_casted = K.cast(self.iterations, K.dtype(self.decay_epochs))
            hit_decay_epoch = K.any(K.equal(ite_casted, self.decay_epochs))#K.any(ite_casted - self.decay_epochs)
        
            print(hit_decay_epoch)
            lr = K.switch(hit_decay_epoch, self.lr['all']*self.decay['all'],
                          self.lr['all'])
            #print("LRR : ", K.eval(lr))
            
            a = K.switch(hit_decay_epoch, 
                          K.print_tensor(self.lr['all'],message='Decays:'), K.print_tensor(self.lr['all'],message=' '))
            
            
            self.updates.append(K.update(self.lr['all'],lr))
            
     
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        #print(self.weights)

        for p, g, m in zip(params, grads, moments):
            
            if p.name in self.lr.keys():
                print("Setting different learning rate for", p.name, ",", K.eval(self.lr[p.name]))
                lr = self.lr[p.name]
                if self.decay_epochs:
                    if  p.name in self.decay.keys():
                        print("Adding decay to ", p.name)
                        lr = K.switch(hit_decay_epoch, self.lr[p.name]*self.decay[p.name],
                                  self.lr[p.name])
                        self.updates.append(K.update(self.lr[p.name],lr))
                    else:
                        print("Adding decay to ", p.name)
                        lr = K.switch(hit_decay_epoch, self.lr[p.name]*self.decay['all'],
                                  self.lr[p.name])
                        self.updates.append(K.update(self.lr[p.name],lr))
                        print("Adding decay to ", K.eval(lr))
                      
            else:
                lr = self.lr['all']
            
            #print("USING LR", p.name, " ", K.eval(lr))
            
            if p.name in self.momentum.keys():
                print("Setting different momentum for ", p.name, ",", K.eval(self.momentum[p.name]))
                momentum = self.momentum[p.name]
            else:
                momentum = self.momentum['all']   
            
                       
            v = momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + momentum * v - lr * g
            else:
                new_p = p + v
                
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            
            if p.name in self.clips.keys():
                print("Clipping variable",p.name," to ", self.clips[p.name] )
                c = K.eval(self.clips[p.name])
                new_p = K.clip(new_p, c[0], c[1])
            #print("updates for ", p.name, " lr: ", K.eval(lr), " mom:", K.eval(momentum))
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': str(eval_Kdict(self.lr)),
                  'momentum': str(eval_Kdict(self.momentum)),
                  'decay': str(eval_Kdict(self.decay)),
                  'clips': str(eval_Kdict(self.clips)),
                  'nesterov': self.nesterov}
        base_config = super(SGDwithLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    

class RMSpropwithClip(Optimizer):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude]
          (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.,clips={},
                 **kwargs):
        super(RMSpropwithClip, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.clips = kdict(clips,'clips')
            self.clips_val = clips
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
                
            if p.name in self.clips.keys():
                print("CLpping variable",p.name," to ", self.clips[p.name] )
                c = K.eval(self.clips[p.name])
                new_p = K.clip(new_p, c[0], c[1])

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RMSpropwithClip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))