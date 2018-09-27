import os
from top_sus_NN_Reg import *

dat_file = 'Esymdep.out'
hiddenN = 100
epochs = 1000
step = 1e-2
regularizer = 1e-10 # Not needed yet...

nnet = Topological_NN(dat_file, h_nodes=hiddenN, epochs=epochs, step_s=step, reg_scale=regularizer)
nnet.make_nn()
nnet.trainn_NN()
