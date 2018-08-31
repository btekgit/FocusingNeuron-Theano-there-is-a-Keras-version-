"""

Project : focusing neuron
Initial version created By : ilker, 20/02/2017, 22:03
Current version is by: btek
"""

from theano.tensor import basic as tensor, subtensor, opt, elemwise
import theano.tensor as T
import numpy as np
import lasagne
import pickle

def save_params(filename,param_values):
    '''
    store the parameters to the  given file
    '''
    f = open(filename, 'wb')
    pickle.dump(param_values,f,protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def load_params(filename):
    '''
    reads all parameters from the  given file
    '''
    f = open(filename, 'rb')
    param_values = pickle.load(f)
    f.close()
    return param_values


     
def acc_weighted_cross_entropy(pred, targets):
    '''
    loss only counting misclassified 
    '''
    #weights = np.not_equal(pred, targets).astype("float")
    #return -np.sum(weights* targets * tensor.log(pred), axis=pred.ndim - 1)
    if targets.ndim == pred.ndim:
        weights = tensor.neq(pred, targets)
        return -tensor.sum(weights * targets * tensor.log(pred),
                           axis=pred.ndim - 1)
    else:
        print (targets.ndim," ",pred.ndim)
        raise TypeError('rank mismatch between coding and true distributions')


def categorical_focal_loss(prediction, target_var, gamma=0.0, alpha=1.0):
    '''
    focal loss function which lowers the cost of allready classified samples 
    '''     
    if target_var.ndim == prediction.ndim:
        # the code is adapted from original categorical loss
        ## f_loss= -(np.log(1-p)*((p)**gamma))
        prediction = T.clip(prediction, 1e-3, 1-1e-3)
        return -T.sum(target_var * T.log(prediction)*((1-prediction)**gamma),
                           axis=prediction.ndim - 1)
    else:         
        print(target_var.ndim, "/=", prediction.ndim)
        raise TypeError('rank mismatch between coding and true distributions')
        


def get_shared_by_name(in_list, name):
    '''
    the method takes in a parameter inlist, 
    returns a single param which has the pattern the name 
    '''
    for ts in in_list:
        
        if ts.name.find(name)>=0:
            
            return ts
    print ("cant find", name)
    return T.shared(np.float32('0'), name='Auto')


def get_shared_by_pattern(in_list, name):
    ''' 
        the method takes in a parameter list, returns a subset 
        which include the pattern 
    '''
    out_list = []
    for ts in in_list:
        
        if ts.name.find(name)>=0:
            
            out_list.append(ts)
    if(len(out_list)==0):
        print ("cant find", name)
    else:
        return out_list
    


def set_params_value(LR_params, new_values, verbose=True):
    ''' 
    set new values --new_lrs-- for a list of shared LR_params 
    '''
    for par,new_lr in zip(LR_params, new_values):
        if verbose:
            print(par.name, " new value:", new_lr)
        par.set_value(np.float32(new_lr))

def set_all_param_values(params_list, new_values, verbose=True):
    ''' 
    set new values --new_lrs-- for a list of shared LR_params 
    '''
    for par,newval in zip(params_list, new_values):
        if verbose:
            print("setting param new value ",par.name)
        par.set_value(newval)

        


def update_learning_rates_wkey(LR_params, key_list, val_list, verbose=True):
    ''' takes a shared param list, a key list and val_list
        it updates the params with the name in key_list with val_list values
        a partial match  of key with the name is enough
    '''
    for par in LR_params:
        for key, val in zip(key_list,val_list):
            if par.name.find(key)>=0:
                par.set_value(np.float32(val))    
                if verbose:
                    print('masking', par.name, " new value:", par.get_value())
                    

def get_params_values_wkey(LR_params, key_list,  verbose=True):
    '''give a LR_params shared param list, a key list
        it returns the params with the name in key_list
        a partial match  of key with the name is enough
    '''
    lr_val =dict()
    if key_list is None or key_list==['all'] or key_list=='all':
        for par in LR_params:
            lr_val[par.name]=par.get_value()
            
        return lr_val
    else:
        
        for par in LR_params:
            for key in key_list:
                if par.name.find(key)>=0:
                    lr_val[par.name]=par.get_value()
            
        return lr_val


def set_params_wkey(LR_params, key_list, val_list, verbose=True):
    '''takes LR_params shared param list, a key list
    it sets the params with the name in key_list with val_list values
    '''
    
    for par in LR_params:
        for key, val in zip(key_list,val_list):
            if par.name.find(key)>=0:
                par.set_value(np.float32(val))
                if verbose:
                    print('masking', par.name, " new value:", par.get_value())
    
    

def debug_print_param_stats(network):
    ''' prints all network varibles' simple statistics
    '''
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    all_param_values = lasagne.layers.get_all_param_values(network, trainable=True)
    n =len(all_params)
    for i in range(n):
        vals = all_param_values[i]
        print(all_params[i]," : mn", np.mean(vals), " mx", np.max(vals), " min", 
              np.min(vals), " std ", np.std(vals))
    


def clip_tensor(tensor, min_v, max_v):
    '''clips a Theano sensor between min and max'''
    #dtype = np.dtype(theano.config.floatX).type
    target_tensor = T.clip(tensor, min_v, max_v)
    return target_tensor

    
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def sgdWithLrs(loss_or_grads, params, learning_rate=.01, mu_lr=.01, si_lr=.001, 
               focused_w_lr=.01, momentum=.9):
    '''
    # This function provides SGD with different learning rates to focus params mu, si, w
    '''
    from collections import OrderedDict
    from lasagne.updates import get_or_compute_grads, apply_momentum
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    momentum_params_list =[]
    print(params)
    for param, grad in zip(params, grads):
        # import pdb; pdb.set_trace()
        #grad = clip_tensor(grad, -0.01, 0.01)
        if param.name.find('focus')>=0 and param.name.find('mu')>=0:
            updates[param] = param - mu_lr * grad
            momentum_params_list.append(param)

        elif param.name.find('focus')>=0 and param.name.find('si')>=0:
            updates[param] = param - si_lr * grad
            #momentum_params_list.append(param)
   
        elif param.name.find('focus')>=0 and param.name.find('W')>=0:
            updates[param] = param - (focused_w_lr * grad)
            momentum_params_list.append(param)
   
        else:
            updates[param] = param - learning_rate * grad
            momentum_params_list.append(param)
            #print (param, grad, learning_rate)
    return apply_momentum(updates, params=momentum_params_list, momentum=momentum)

def sgdWithLrsClip(loss_or_grads, params, learning_rate=.01, mu_lr=.01, si_lr=.001, 
               focused_w_lr=.01, momentum=.9):
    '''
    Sames as sgdWithLrs bu applies clips after updates
    '''
    from collections import OrderedDict
    from lasagne.updates import get_or_compute_grads, apply_momentum
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    #momentum_params_list =[]
    print("Params List",params)
    for param, grad in zip(params, grads):
        
        #grad = clip_tensor(grad, -0.001, 0.001)
        if param.name.find('focus')>=0 and param.name.find('mu')>=0:
            updates[param] = param - mu_lr * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            updates[param] =clip_tensor(updates[param], np.float32(0.01), np.float32(0.99))

        elif param.name.find('focus')>=0 and param.name.find('si')>=0:
            updates[param] = param - si_lr * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            updates[param] =clip_tensor(updates[param], 0.01, 0.5)
            
        elif param.name.find('focus')>=0 and param.name.find('W')>=0:
            updates[param] = param - (focused_w_lr * grad)
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            #updates[param] =clip_tensor(updates[param], -0.5, 0.5)
        else:
            updates[param] = param - learning_rate * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            if param.name.find('W')>=0:
                print (param, grad, learning_rate)
    return updates

def sgdWithLrLayers(loss_or_grads, params, learning_rate=.01, mu_lr=.01, si_lr=.001, 
               focused_w_lr=.01, momentum=.9):
    '''
    # This function updates each layer parameters with a different learning rate. 
    Under dev.
    '''
    from collections import OrderedDict
    from lasagne.updates import get_or_compute_grads, apply_momentum
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    #momentum_params_list =[]
    #print(params)
    for param, grad in zip(params, grads):
        # import pdb; pdb.set_trace()
        grad = clip_tensor(grad, -0.01, 0.01)
        if param.name.find('focus')>=0 and param.name.find('mu')>=0:
            updates[param] = param - mu_lr * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum/2)
            updates[param] =clip_tensor(updates[param], 0.05, 0.95)
            #momentum_params_list.append(param)
            #print (param,mu_lr)
            #print (param, grad, mu_lr)
        elif param.name.find('focus')>=0 and param.name.find('si')>=0:
            updates[param] = param - si_lr * grad
            #momentum_params_list.append(param)
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            updates[param] =clip_tensor(updates[param], 0.01, 0.5)
            
            #print (param,si_lr)
            #print (param, grad, si_lr)
            #print (param, grad, scaler_lr)
        elif param.name.find('focus')>=0 and (param.name.find('W')>=0 or param.name.find('bias')>=0):
            level= int(str.split(param.name,'-')[1].split('.')[0])
            #print(param.name, level)
            updates[param] = param - (learning_rate*(1./(level+1))) * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            if (param.name.find('W')>=0):
                updates[param] =clip_tensor(updates[param], -0.4, 0.4)
            #momentum_params_list.append(param)
            #print (param,focused_w_lr)
        elif param.name.find('W')>=0 or param.name.find('b')>=0:
            if param.name.find('-')>=0:
                level= int(str.split(param.name,'-')[1].split('.')[0])
                updates[param] = param - (learning_rate*(1./level)) * grad
                updates = apply_momentum(updates, params=[param], momentum=momentum)
            else:
                updates[param] = param - (learning_rate) * grad
            #momentum_params_list.append(param)
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            if (param.name.find('W')>=0):
                updates[param] = clip_tensor(updates[param], -0.4, 0.4)
            
            if (param.name.find('b')>=0):
                updates[param] = clip_tensor(updates[param], -1.0, 1.0)
        else:
            updates[param] = param - (learning_rate) * grad
            #momentum_params_list.append(param)
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            if (param.name.find('beta')>=0):
                updates[param] = clip_tensor(updates[param], -1., 1.)
            #print (param, grad, learning_rate)
            
    return updates



def sgdWithWeightSupress(loss_or_grads, params, learning_rate=.01, mu_lr=.01, si_lr=.001, 
               focused_w_lr=.01, momentum=.9):
    ''' this update function masks focus weights after they are updated.
    The idea is that weights outside of the focus function must be suppressed
    to prevent weight memory when focus changes its position
    
    To do this I get mu and si values of the focus layer, calculate a Gauss,
    window scale it so the center is 1 but outside is close to 0, and then multiply
    it with the weights.
    '''
    from collections import OrderedDict
    from lasagne.updates import get_or_compute_grads, apply_momentum
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    #momentum_params_list =[]
    print(params)
    for param, grad in zip(params, grads):
        
        #grad = clip_tensor(grad, -0.001, 0.001)
        if param.name.find('focus')>=0 and param.name.find('mu')>=0:
            updates[param] = param - mu_lr * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            updates[param] =clip_tensor(updates[param], 0.01, 0.99)

        elif param.name.find('focus')>=0 and param.name.find('si')>=0:
            updates[param] = param - si_lr * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            updates[param] =clip_tensor(updates[param], 0.01, 0.5)
            
        elif param.name.find('focus')>=0 and param.name.find('W')>=0:
            param_layer_name = param.name.split(".")[0]
            mu_name = param_layer_name +'.mu'
            
            si_name = param_layer_name+".si"
            print("Hey::",param.name, " ", mu_name, " ",si_name, " w shape ", param.shape)
            mu_si_w = get_params_values_wkey(params,[mu_name,si_name, param.name])
            print("Hey weight shape::",mu_si_w[param.name].shape)

            from focusing import U_numeric
            us = U_numeric(np.linspace(0,1,mu_si_w[param.name].shape[0]),mu_si_w[mu_name],
                           mu_si_w[si_name],1, normed=False)
            print("Hey us shape::",us.shape)
            
            updates[param] = (param - (focused_w_lr * grad))
            
    
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            # here we are masking the weights, so they can not stay out of envelope
            us[us>0.1] = 1.0
            updates[param] = updates[param] * us.T 
            #updates[param] = updates[param]*, -0.5, 0.5)
        else:
            updates[param] = param - learning_rate * grad
            updates = apply_momentum(updates, params=[param], momentum=momentum)
            #print (param, grad, learning_rate)
    return updates