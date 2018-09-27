IN PROGRESS...

Constructing NN to determine when Markov Chain step proposed by lattice QCD 
alogrithm will not produce change in global topological susceptibility. At 
the moment Fer and I are experimenting with various types of input -- 
we are trying to distill the information contained at each of the lattice
sites by averaging various observables over LQCD size chunks. 

Files and info:

1.) top_sus_NN_Reg.py
This file contains two classes. The first, Topological_NN, contains the NN 
architecture and trains the NN. The second, ImportGraph, is used for NN 
evalulation. At the moment, we are implimenting a 3 layer feedforward network,
on the data which has been standardized, with sigmoid activation functions and
a guassian cost function (where the predicted value has been rounded, since 
topological charge takes integer values). Any of this can be trivially modified.

2.) trainNN.py
This is just a quick wrapper function to train a new NN.
Variables that can be adjusted: Number of nodes, step size,
number of training epochs, regularizer (applied to all matricies).

3.) plotter.py
Used to make plots, can be run quickly with run_plots.py

