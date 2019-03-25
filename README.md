# FocusingNeuron

Repo for focusing neuron (adaptive locally connected neuron)

Paper: https://arxiv.org/abs/1809.09533


## Code

Depends on other libraries: numpy, scikit, theano, lasagne


### EXAMPLES

- Quick example:
python Test-Synthetic-Inputs.py

- MNIST example
python mnist.py focused_mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0
python mnist.py mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0


### DATA 

requires mnist.npz or downloads it from http://yann.lecun.com/exdb/mnist/
Other datasets such as cifar_10 and fashion can be downloaded with keras
mnist_cluttered data is difficult to find in internet
