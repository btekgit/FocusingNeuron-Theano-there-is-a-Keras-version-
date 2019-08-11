#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:36:24 2019

@author: btek
"""

from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, regularizers, constraints
from keras import initializers
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf


#Keras TF implementation of Focusing Neuron.
class FocusedLayer1D(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gain=1.0,
                 init_mu = 'spread',
                 init_sigma=0.1,
                 init_w=initializers.glorot_uniform(),
                 init_bias = initializers.Constant(0.0),
                 train_mu=True,train_sigma=True, 
                 train_weights=True,
                 reg_bias=None,
                 normed=2,
                 verbose=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FocusedLayer1D, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.gain = gain
        self.init_sigma=init_sigma
        self.init_weights =init_w
        self.init_mu = init_mu
        self.train_mu = train_mu
        self.train_sigma = train_sigma
        self.train_weights = train_weights
        self.normed = normed
        self.verbose = verbose
        self.sigma=None
        
            
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'))
        print(kwargs)
        
        #super(Focused, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        
        #self.kernel = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        
        
        mu, si = mu_si_initializer(self.init_mu, self.init_sigma, self.input_dim,
                                   self.units, verbose=self.verbose)
        
        idxs = np.linspace(0, 1.0,self.input_dim)
        idxs = idxs.astype(dtype='float32')
        
        self.idxs = K.constant(value=idxs, shape=(self.input_dim,), 
                                   name="idxs")
        
        from keras.initializers import constant
         # create trainable params.
        self.mu = self.add_weight(shape=(self.units,), 
                                  initializer=constant(mu), 
                                  name="Mu", 
                                  trainable=self.train_mu)
        self.sigma = self.add_weight(shape=(self.units,), 
                                     initializer=constant(si), 
                                     name="Sigma", 
                                     trainable=self.train_sigma)
        # idx is not trainable
        
      
        # value caps for MU and SI values
        # however these can change after gradient update.
        MIN_SI = 0.01  # zero or below si will crashed calc_u
        MAX_SI = 1.0 
        
        # create shared vars.
        self.MIN_SI = np.float32(MIN_SI)#, dtype='float32')
        self.MAX_SI = np.float32(MAX_SI)#, dtype='float32')
        
        
        self.W = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.weight_initializer_fw_bg,
                                      name='Weights',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.built = True
        
        #super(FocusedLayer1D, self).build(input_shape)  # Be sure to call this somewhere!
        
        #super(FocusedLayer1D, self).build(input_shape)
        
    def call(self, inputs):
        u = self.calc_U()
        if self.verbose:
            print("weights shape", self.W.shape)
        self.kernel = self.W*u
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'focusing vars': "not here"
        }
        base_config = super(FocusedLayer1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def weight_initializer(self,shape):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        W = np.zeros(shape=shape, dtype='float32')
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        
        for c in range(W.shape[1]):
            fan_in = np.sum((kernel[:,c])**2)
            
            #fan_in *= self.input_channels no need for this in repeated U. 
            if initer == 'He':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
            else:
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
            
            std = np.float32(std)
            if c == 0 and verbose:
                print("Std here: ",std, type(std),W.shape[0],
                      " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
            if distribution == 'uniform':
                std = std * sqrt32(3.0)
                std = np.float32(std)
                w_vec = np.random.uniform(low=-std, high=std, size=W.shape[:-1])
            elif distribution == 'normal':
                std = std/ np.float32(.87962566103423978)           
                w_vec = np.random.normal(scale=std, size=W.shape[0])
                
            W[:,c] = w_vec.astype('float32')
            
        return W

    def weight_initializer_fw_bg(self,shape):
        #only implements channel last and HE uniform
        initer = 'Glorot'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        
        W = np.zeros(shape=shape, dtype='float32')
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        sum_over_domain = np.sum(kernel**2,axis=1) # r base
        sum_over_neuron = np.sum(kernel**2,axis=0)
        for c in range(W.shape[1]):
            for r in range(W.shape[0]):
                fan_out = sum_over_domain[r]
                fan_in = sum_over_neuron[c]
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
                else:
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
                    print(r,",",c," Fan in ", fan_in, " Fan_out:", fan_out, W[r,c])
                    
                if distribution == 'uniform':
                    std = std * sqrt32(3.0)
                    std = np.float32(std)
                    w_vec = np.random.uniform(low=-std, high=std, size=1)
                elif distribution == 'normal':
                    std = std/ np.float32(.87962566103423978)           
                    w_vec = np.random.normal(scale=std, size=1)
                    
                W[r,c] = w_vec.astype('float32')
                
        return W
    
    def calc_U(self,verbose=False):
        """
        function calculates focus coefficients. 
        normalizes and prunes if
        """
        up= (self.idxs - K.expand_dims(self.mu,1))**2
        #print("up.shape", up.shape)
        #up = K.expand_dims(up,axis=1,)
        #print("up.shape",up.shape)
        # clipping scaler in range to prevent div by 0 or negative cov. 
        sigma = K.clip(self.sigma,self.MIN_SI,self.MAX_SI)
        #cov_scaler = self.cov_scaler
        dwn = K.expand_dims(2 * ( sigma ** 2), axis=1)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        #print("down shape :",dwn.shape)
        result = K.exp(-up / dwn)
        kernel= K.eval(result)
        #print("RESULT max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
  
        
        
      
        #print("U shape :",result.shape)
        #print("inputs shape",inputs.shape)

        #sum normalization each filter has sum 1
        #sums = K.sum(masks**2, axis=(0, 1), keepdims=True)
        #print(sums)
        #gain = K.constant(self.gain, dtype='float32')
        
        #Normalize to 1
        
        if self.normed==1:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
        
        elif self.normed==2:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
            result *= K.sqrt(K.constant(self.input_dim))

            if verbose:
                kernel= K.eval(result)
                print("RESULT after NORMED max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            #
        #Normalize to get equal to WxW Filter
        #masks *= K.sqrt(K.constant(self.input_channels*self.kernel_size[0]*self.kernel_size[1]))
        # make norm sqrt(filterw x filterh x self.incoming_channel)
        # the reason for this is if you take U all ones(self.kernel_size[0],kernel_size[1], num_channels)
        # its norm will sqrt(wxhxc)
        #print("Vars: ",self.input_channels,self.kernel_size[0],self.kernel_size[1])
        
        
        return K.transpose(result)

        
def mu_si_initializer(initMu, initSi, num_incoming, num_units, verbose=True):
    '''
    Initialize focus centers and sigmas with regards to initMu, initSi
    
    initMu: a string, a value, or a numpy.array for initialization
    initSi: a string, a value, or a numpy.array for initialization
    num_incoming: number of incoming inputs per neuron
    num_units: number of neurons in this layer
    '''
    
    if isinstance(initMu, str):
        if initMu == 'middle':
            #print(initMu)
            mu = np.repeat(.5, num_units)  # On paper we have this initalization                
        elif initMu =='middle_random':
            mu = np.repeat(.5, num_units)  # On paper we have this initalization
            mu += (np.random.rand(len(mu))-0.5)*(1.0/(float(20.0)))  # On paper we have this initalization                
            
        elif initMu == 'spread':
            mu = np.linspace(0.2, 0.8, num_units)
        else:
            print(initMu, "Not Implemented")
            
    elif isinstance(initMu, float):  #initialize it with the given scalar
        mu = np.repeat(initMu, num_units)  # 

    elif isinstance(initMu,np.ndarray):  #initialize it with the given array , must be same length of num_units
        if initMu.max() > 1.0:
            print("Mu must be [0,1.0] Normalizing initial Mu value")
            initMu /=(num_incoming - 1.0)
            mu = initMu        
        else:
            mu = initMu
    
    #Initialize sigma
    if isinstance(initSi,str):
        if initSi == 'random':
            si = np.random.uniform(low=0.05, high=0.25, size=num_units)
        elif initSi == 'spread':
            si = np.repeat((initSi / num_units), num_units)

    elif isinstance(initSi,float):  #initialize it with the given scalar
        si = np.repeat(initSi, num_units)# 
        
    elif isinstance(initSi, np.ndarray):  #initialize it with the given array , must be same length of num_units
        si = initSi
        
    # Convert Types for GPU
    mu = mu.astype(dtype='float32')
    si = si.astype(dtype='float32')

    if verbose:
        print("mu init:", mu)
        print("si init:", si)
        
    return mu, si


def U_numeric(idxs, mus, sis, scaler, normed=2):
    '''
    This function provides a numeric computed focus coefficient vector for
    
    idxs: the set of indexes (positions) to calculate Gaussian focus coefficients
    
    mus: a numpy array of focus centers
    
    sis: a numpy array of focus aperture sigmas
    
    scaler: a scalar value
    
    normed: apply sum normalization
        
    '''
    
    up = (idxs - mus[:, np.newaxis]) ** 2
    down = (2 * (sis[:, np.newaxis] ** 2))
    ex = np.exp(-up / down)
    
    if normed==1:
        #sums = np.sqrt(np.sum(ex**2,axis=1))
        # current focus normalizes each neuron to receive one full 
        #ex /= sums[:,np.newaxis]
        #num_incoming = idxs.shape[0]
        #ex = ex*scaler*np.sqrt(num_incoming)
        print('not completed')

    return (ex.astype(dtype='float32'))


def sqrt32(x):
    return np.sqrt(x,dtype='float32')

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



        
        
    


def test(dset='mnist', random_sid=9, epochs=10, 
         data_augmentation=False,batch_size = 512, mod='focused'):
    import keras
    from keras.losses import mse
    from keras.optimizers import SGD
    from keras.datasets import mnist,fashion_mnist, cifar10
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Dropout, Flatten,Conv2D, BatchNormalization
    from keras.layers import Activation, Permute,Concatenate
    from skimage import filters
    from keras import backend as K
    from keras_utils import WeightHistory as WeightHistory
    from keras_utils import RecordVariable, \
    PrintLayerVariableStats, PrintAnyVariable, SGDwithLR, eval_Kdict
    from keras_preprocessing.image import ImageDataGenerator

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
   # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
    K.clear_session()


    sid = random_sid
 
    
    #test_acnn = True
    
    np.random.seed(sid)
    tf.random.set_random_seed(sid)
    tf.compat.v1.random.set_random_seed(sid)
    
    from datetime import datetime
    #$now = datetime.now()
    
    if dset=='mnist':
        # input image dimensions
        img_rows, img_cols = 28, 28  
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        n_channels=1
        
        lr_dict = {'all':0.1,
                   'focus-1/Sigma:0': 0.01,'focus-1/Mu:0': 0.01,'focus-1/Weights:0': 0.1,
                   'focus-2/Sigma:0': 0.01,'focus-2/Mu:0': 0.01,'focus-2/Weights:0': 0.1}

        mom_dict = {'all':0.9,'focus-1/Sigma:0': 0.9,'focus-1/Mu:0': 0.9,
                   'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}
    
        decay_dict = {'all':0.9, 'focus-1/Sigma:0': 0.1,'focus-1/Mu:0':0.1,
                  'focus-2/Sigma:0': 0.1,'focus-2/Mu:0': 0.1}

        clip_dict = {'focus-1/Sigma:0':(0.05,1.0),'focus-1/Mu:0':(0.0,1.0),
                 'focus-2/Sigma:0':(0.05,1.0),'focus-2/Mu:0':(0.0,1.0)}
        
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*100, e_i*150], dtype='int64')
    
    elif dset=='cifar10':    
        img_rows, img_cols = 32,32
        n_channels=3
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        lr_dict = {'all':0.001,
                  'focus-1/Sigma:0': 0.01,'focus-1/Mu:0': 0.01,'focus-1/Weights:0': 0.1,
                  'focus-2/Sigma:0': 0.01,'focus-2/Mu:0': 0.01,'focus-2/Weights:0': 0.1}

        mom_dict = {'all':0.9,'focus-1/Sigma:0': 0.9,'focus-1/Mu:0': 0.9,
                   'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}
    
        decay_dict = {'all':0.9, 'focus-1/Sigma:0': 0.1,'focus-1/Mu:0':0.1,
                  'focus-2/Sigma:0': 0.1,'focus-2/Mu:0': 0.1}

        clip_dict = {'focus-1/Sigma:0':(0.05,1.0),'focus-1/Mu:0':(0.0,1.0),
                 'focus-2/Sigma:0':(0.05,1.0),'focus-2/Mu:0':(0.0,1.0)}
        
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*30,e_i*80,e_i*120,e_i*180], dtype='int64')
    
    
    num_classes = np.unique(y_train).shape[0]
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)

   
    

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, _, x_test = standarize_image_025(x_train,tst=x_test)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
    input_shape = (img_rows, img_cols, n_channels)
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    
    node_in = Input(shape=input_shape, name='inputlayer')
    #node_in = Dropout(0.2)(node_in)
 
        

    node_fl = Flatten(data_format='channels_last')(node_in)
    # smaller initsigma does not work well. 
    node_drp = Dropout(0.2)(node_fl)
    heu= initializers.he_uniform
    if mod=='focused':
        node_foc = FocusedLayer1D(units=800,
                                  name='focus-1',
                                  activation='linear',
                                  init_sigma=0.1, 
                                  init_mu='spread',
                                  init_w= None,
                                  train_sigma=True, 
                                  train_weights=True,
                                  train_mu = True,normed=2)(node_drp)
    else:
        node_foc = Dense(800,activation='linear',kernel_initializer=heu())(node_drp)
    
    node_foc = BatchNormalization()(node_foc)
    node_foc = Activation('relu')(node_foc)
    node_foc = Dropout(0.25)(node_foc)
    
    if mod=='focused':
        node_foc = FocusedLayer1D(units=800,
                                  name='focus-2',
                                  activation='linear',
                                  init_sigma=0.1, 
                                  init_mu='spread',
                                  init_w= None,
                                  train_sigma=True, 
                                  train_weights=True,
                                  train_mu = True,normed=2)(node_foc)
    else:
        node_foc = Dense(800,activation='linear',kernel_initializer=heu())(node_drp)
        
    node_foc = BatchNormalization()(node_foc)
    node_foc = Activation('relu')(node_foc)
    node_foc = Dropout(0.5)(node_foc)
    #node_foc = Dense(10)(node_fl)
    
    node_fin = Dense(num_classes, activation='softmax', 
                     kernel_initializer=initializers.he_uniform(),
                    kernel_regularizer=None)(node_foc)
    
    
    
    #decay_check = lambda x: x==decay_epoch
    opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, decay_epochs)#, decay=None)
    
    #opt= SGDwithLR({'all': 0.01},{'all':0.9})#, decay=None)
    #opt= SGD(lr=0.01, momentum=0.9)#, decay=None)
    
    model = Model(inputs=node_in, outputs=[node_fin])
        
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    
    #print("iterations",model.optimizer.get_updates())
    
    
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    #callbacks = [tb]
    callbacks = []
    
    if mod=='focused':
        pr_1 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
        rv_weights_1 = RecordVariable("focus-1","Weights:0")
        rv_sigma_1 = RecordVariable("focus-1","Sigma:0")
        print_lr_rates_callback = keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print("iter: ", 
                                                       K.eval(model.optimizer.iterations),
                                                       " LR RATES :", 
                                                       eval_Kdict(model.optimizer.lr)))
    
        callbacks+=[pr_1,pr_2,pr_3,rv_weights_1,rv_sigma_1,print_lr_rates_callback]
    
        
    if not data_augmentation:
        print('Not using data augmentation.')
        history=model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format='channels_last',
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow().
        history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks, 
                            steps_per_epoch=x_train.shape[0]//batch_size)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score,history,model


    

def repeated_trials(dset='mnist',N=1, epochs=2, augment=False,delayed_start=0,
                    mod='focused'):
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    
    list_scores =[]
    list_histories =[]
    models = []
    
    import time 
    from shutil import copyfile
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    
    filename = 'outputs/Kfocusing/'+dset+'/'+timestr+'_'+mod+'.model_results.npz'
    copyfile("Kfocusing.py",filename+"code.py")
    for i in range(N):
        sc, hs, ms = test(dset=dset,random_sid=i*17, epochs=epochs,
                          data_augmentation=augment,mod=mod)
        list_scores.append(sc)
        list_histories.append(hs)
        models.append([ms])
    
    
    print("Final scores", list_scores,)
    mx_scores = [np.max(list_histories[i].history['val_acc']) for i in range(len(list_histories))]
    print("Max sscores", mx_scores)
    np.savez_compressed(filename,mx_scores =mx_scores, list_scores=list_scores, modelz=models)
    return mx_scores, list_scores, models


if __name__ == "__main__":
    print("Run as main")
    #test()
    mod='focused'
    repeated_trials(dset='cifar10',N=5,epochs=200, augment=True, mod=mod)
    # focused MNIST Augmented accuracy (200epochs): ~99.25-30
    # focused CIFAR-10 Augmented accuracy(200epochs): ~0.675
