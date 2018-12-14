import os
from top_sus_NN_Reg import *

dat_file = 'datablocks.dat'
hiddenN = 2
epochs = 200
step = 1e-2
regularizer = 1e-8 # Not needed yet...

nnet = Topological_NN(dat_file, h_nodes=hiddenN, epochs=epochs, step_s=step, reg_scale=regularizer)
nnet.make_nn()
nnet.trainn_NN()
