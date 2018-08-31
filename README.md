# FocusingNeuron
Repo for focusing neuron (adaptive locally connected neuron)

Requirements: numpy, scikit, theano, lasagne
Real datasets require : /datasets mnist.npz (can be downloaded), cifar10.npz ...

Quick example:
python Test-Synthetic-Inputs.py

# requires mnist.npz
python mnist.py focused_mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0

