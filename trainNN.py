import os
from top_sus_NN import *

dat_file = 'updates.dat'
hiddenN = 20
epochs = 1000
step = 1e-2
regularizer = 1e-1

nnet = Topological_NN(dat_file, h_nodes=hiddenN, epochs=epochs, step_s=step, reg_scale=regularizer)
nnet.make_nn()
nnet.trainn_NN()
