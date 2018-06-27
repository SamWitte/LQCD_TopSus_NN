import os
from top_sus_NN import *

dat_file = 'updates.dat'
hiddenN = 10
epochs = 1000
step = 1e-1

nnet = Topological_NN(dat_file, h_nodes=hiddenN, epochs=epochs, step_s=step)
nnet.make_nn()
nnet.trainn_NN()
