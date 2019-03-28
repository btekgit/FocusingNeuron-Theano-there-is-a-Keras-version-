# FocusingNeuron

Repo for focusing neuron (adaptive locally connected neuron)

Paper: https://arxiv.org/abs/1809.09533
Notes: 
1) Paper is an older version with slightly different focus function normalization 
2) Current code can provide even better results. 



## Code

Depends on other libraries: numpy, scikit, theano, lasagne


### EXAMPLES

- Quick example runs on synthetically generated classication datasets:
*python Test-Synthetic-Inputs.py

- MNIST example

*python mnist.py focused_mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0

Test set accuracy is ~99.10-99.20

*python mnist.py mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0

Test set accuracy is ~98.9-99.05


### DATA 

Requires mnist.npz or downloads it from http://yann.lecun.com/exdb/mnist/
Other datasets such as cifar_10 and fashion can be downloaded with keras.datasets
Note: mnist_cluttered data is difficult to find in internet again. Email me if you cant find it. I will upload it 



### EXPERIMENTS
Repeated trial experiments are implemented .sh files. Contains my local directory references.
