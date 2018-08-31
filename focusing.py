# -*- coding: utf-8 -*-

#from __future__ import unicode_literals
"""
/*
 * Focusing Neuron 1D
 *
 * Code Authors:  F. Boray Tek 
                  (İlker Çam contributed to an earlier version)
 *
 *
 *
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
 * requires a license from F. Boray Tek or Işık University.
 * 
 *
 * For further details, contact F. Boray Tek (boraytek@gmail.com).
 */
 
 
Project : FocusingNeuron
Created By : btek, 12/04/2018, 13:45

"""

from lasagne.layers import Layer
import theano
import theano.tensor as T
import numpy as np
from lasagne import nonlinearities, init

__all__ = [
    "FocusedLayer1D",
    "U_numeric"
]

class FocusedLayer1D(Layer):
    """
    FocusedLayer1D(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=1, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer
        
    num_leading_axes : int
        Number of leading axes to distribute the dot product over. These axes
        will be kept in the output tensor, remaining axes will be collapsed and
        multiplied against the weight matrix. A negative number gives the
        (negated) number of trailing axes to involve in the dot product.
        Not sure this is implemented in the current version, we inherit this
        from Lasagne.
        
    W : This is for compatibility with lasagne.layers. We do not use it.

    bias : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.
        
    withWeights: Can set the focusing layer withWeights=True or withoutWeightd=False
                 (default=True)
                 
    withPruning: Can prune the focus coefficients with two standard devs from the mean
                 (default=False)
     
    trainMus: Can switch training of focus centers (default=True), 
     
    trainSis: Can switch training of focus apertures (default=True),
     
    trainWs: : Can switch training of focus weights (default=True),
     
    initMu: How to set up mu initially, can  string={'middle', 'middle_random','spread'}, or
    a floating point number or a numpy float array of size num_units.
    default(middle)
     
    initSi: How to set up mu initially, can  string={random'}, or
    a floating point number or a numpy float array of size num_units.
    default(0.1)
     
    weight_gain: a floating point number to scale weight initialization. default(1.0) 
    
    focus_scaler: a floating point number to scale focus coeefficients default(1.0)
    By default the sum of focus coefficients is equal to incoming input size.
     
    trainScaler: it is possible to train scaler, sum of focus coeffs. default(False), 
     
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> l_in = InputLayer((100, 20))
    >>> l_1 = FocusedLayer1D(
                l_in, num_units=10, nonlinearity=lasagne.nonlinearities.softmax, 
                              name='focus-1',
                              trainMus=True,
                              trainSis=True,
                              initMu='spread',
                              W=lasagne.init.Constant(0.0), withWeights=True,
                              bias=lasagne.init.Constant(0.0),
                              initSigma=initsigma,
                             scaler=1.0, weight_gain=1.0,
                             trainScaler=False, trainWs=True)

    Must clarify what happend if the input has more than two axes, by default, all trailing axes will be
    flattened. This is useful when a dense layer follows a convolutional layer.??
    """
    def __init__(self, incoming, num_units, num_leading_axes=1, 
                 W=None, bias=None, 
                 withWeights=False, withPruning=False,trainMus=False, 
                 trainSis=False, trainWs=False, initMu='spread', initSigma=0.1, 
                 weight_gain=1.0, scaler=1.0, 
                 trainScaler=False, nonlinearity=nonlinearities.linear,
                 name=None,
                 **kwargs):
        super(FocusedLayer1D, self).__init__(incoming, name, **kwargs)
        
        self.num_units = num_units
        self.incoming = incoming
        self.num_leading_axes = num_leading_axes
        self.num_incoming = int(np.prod(self.input_shape[num_leading_axes:]))
        self.nonlinearity = nonlinearity
        self.withWeights = withWeights
        self.withPruning = withPruning
        self.trainMus = trainMus
        self.trainSis = trainSis
        self.weight_gain = weight_gain

        # Initalizations  for focus center mu and aperture sigma     
        mu, si = mu_si_initializer(initMu, initSigma, self.num_incoming, num_units)
        
        #  setup the input positional frame           
        idxs = T.arange(0, self.num_incoming) / T.cast(self.num_incoming, 
                       dtype=theano.config.floatX)
        idxs = idxs.astype(dtype='float32')

        # create trainable params.
        self.mu = self.add_param(spec=mu, shape=(num_units, ), name="mu", trainable=trainMus)
        self.si = self.add_param(spec=si, shape=(num_units,), name="si", trainable=trainSis)
        # idx is not trainable
        self.idxs = self.add_param(spec=idxs, shape=(self.num_incoming,), 
                                   name="idxs", trainable=False,
                                   regularizable=False)
        self.scaler = self.add_param(init.Constant(scaler), (num_units,), name="scaler", trainable=trainScaler)
        self.b = None# default variable for bias is set to None we setup a new one.
        
      
        # value caps for MU and SI values
        # however these can change after gradient update.
        MIN_SI = 0.01  # zero or below si will crashed calc_u
        MAX_SI = 1.0 
        
        # create shared vars.
        self.MIN_SI = T.constant(MIN_SI, dtype='float32')
        self.MAX_SI = T.constant(MAX_SI, dtype='float32')
        
        # if weights are used call weight_init method
        if withWeights:
            W = self.weight_init((self.num_incoming , num_units))
            self.W = self.add_param(W, (self.num_incoming , num_units), 
                                    name="W", trainable=trainWs)
            if bias is not None:
                self.bias = self.add_param(bias, (self.num_units,), 
                                           name="bias", trainable=True,
                                           regularizable=True)


    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units,)


    
    def weight_init(self, shape):

        wu = self.calc_u().eval().T
        W = np.zeros_like(wu, dtype='float32')
        # for Each Gaussian initialize a new set of weights
        for c in range(wu.shape[1]):
            fan_in = np.sum(wu[:,c]**2)
            #print ("fan_in", fan_in)
            
            # Jul 9 2018
            # below init does not work bad
            #std = self.weight_gain * np.sqrt(2.0 / (fan_in + self.num_units))
            #w_vec = np.random.uniform(low=-std,high=std,size=(wu.shape[0],))
            # np.sqrt(2/(800+800)) = 0.035
            # for HE normal
            #std = self.weight_gain * np.sqrt(2.0) / np.sqrt((fan_in))
            #w_vec = np.random.normal(loc=0, scale=std, size=(wu.shape[0],))
            # HE uniform 
            #std = self.weight_gain * np.sqrt(6.0) / np.sqrt((fan_in))
            # this was the mnist8 result
            
            std = self.weight_gain * np.sqrt(6.0) / np.sqrt(fan_in)
            w_vec = np.random.uniform(low=-std,high=std,size=(wu.shape[0],))
            
            if c==0 or c==(wu.shape[1]-1) or c==(wu.shape[1]/2):

                print ("neuron", c, "WU: ", (wu[:,c]*w_vec)[:6])
                print ("std:, fan_in", std,fan_in)
                
            W[:,c] = w_vec.astype('float32')
        
        return  W
    
        
    def weight_init_product_variance(self, shape):
        """ weights are sampled from different variance to have 1/m variance 
            Var[w_i * f_i]=1/m
            with fay**2 (sx**2+mux**2) variance
        """
        #from lasagne.init import Normal as initializer
        # for weight init, calculate initial Gaussians.
        wu = self.calc_u().eval().T
        W = np.zeros_like(wu, dtype='float32')
        N = wu.shape[0]
        # for Each Gaussian initialize a new set of weights
        for c in range(wu.shape[1]):
            for k in range(wu.shape[0]):
                
                fay = np.clip(wu[k,c], 1.0,N)
                std = np.sqrt(6)/np.sqrt(N*(fay**2))
                # overwrite above for  wu(j,c)**2 variance. 
                w_k = np.random.uniform(low=-std,high=std,size=1)            
                W[k,c] = w_k
            
            if c==0 or c==(wu.shape[1]-1) or c==(wu.shape[1]/2):

                print ("neuron", c, "WU: ", (wu[:,c]*W[:,c])[:6])
                print ("std: ", std)
      
        return  W.astype('float32')
    
    
    def get_output_for(self, inputs, **kwargs):
        '''
        Calculates output by first
        1) Calling calc_u to compute focus coeeficients
        2) Multiply focus coeffs with weights
        3) Dot product focused weights with inputs
        '''    
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += inputs.ndim
        if inputs.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            inputs = inputs.flatten(num_leading_axes + 1)

        # calculate focus coeffs
        u  = self.calc_u()
         
        if self.withWeights:
            wu = self.W * u.T
            activation = T.dot(inputs, wu)
            if self.bias is not None:
                activation = activation + self.bias
        else:            
            activation = T.dot(inputs, u.T)
        
        return self.nonlinearity(activation)


    def calc_u(self):
        '''
        function calculates focus coefficients. 
        normalizes and prunes if
        '''
        si_clipped = T.clip(self.si, self.MIN_SI, self.MAX_SI)
        mu_clipped = self.mu
      
        #make a column out of a 1d vector (N to Nx1)
        up = (self.idxs - mu_clipped.dimshuffle(0, 'x')) ** 2
        down = (2 * (si_clipped.dimshuffle(0, 'x') ** 2))
        
        # this was working in all experiments. Jul 10 2018
        ex = T.exp(-up / down)
        # this was working Jul 18
        ex /= T.sum(ex,axis=1).dimshuffle(0,'x')
        ex *= self.num_incoming * self.scaler.dimshuffle(0, 'x')
        
        # pruning is not tested with this version
        if self.withPruning:
            ex *= abs(self.idxs - self.mu.dimshuffle(0, 'x')) <= (self.si.dimshuffle(0, 'x') * 0.2)  # Pruning
            
        return ex
 
    
def mu_si_initializer(initMu, initSi, num_incoming, num_units):
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
            #mu += (np.random.rand(len(mu))-0.5)*(1.0/(float(num_incoming*num_incoming)))  # On paper we have this initalization                
            mu += (np.random.rand(len(mu))-0.5)*(1.0/(float(20.0)))  # On paper we have this initalization                
            print("mu init:", mu)
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
            si = np.random.uniform(low=0.05,high=0.25,size=num_units)
        elif initSi == 'spread':
            si = np.repeat((initSi / num_units), num_units)

    elif isinstance(initSi,float):  #initialize it with the given scalar
        si = np.repeat(initSi, num_units)# 
        
    elif isinstance(initSi, np.ndarray):  #initialize it with the given array , must be same length of num_units
        si = initSi
        
    # Convert Types for GPU
    mu = mu.astype(dtype='float32')
    si = si.astype(dtype='float32')
        
    return mu, si


def U_numeric(idxs, mus, sis, scaler, normed=True):
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
    
    if normed:
        sums = np.sum(ex,axis=1)
        # current focus normalizes each neuron to receive one full 
        ex /= sums[:,np.newaxis]
    num_incoming = idxs.shape[0]
    #print('num_incoming', num_incoming)
    ex = ex*scaler*num_incoming

    return (ex.astype(dtype='float32'))