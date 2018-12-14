import os
from schwinger_NN_Reg import *

dat_file = 'fermion_determinant'   # ['fermion_determinant', 'top_charge', 'pion_correlator']
hiddenN = 100 # Best Found: [???, ~10]
epochs = 1000
step = 1e-4 # Best Found: [???, 1e-2]
regularizer = 1e-10 # Not needed yet...

nnet = Topological_NN(dat_file, h_nodes=hiddenN, epochs=epochs, step_s=step, reg_scale=regularizer)
nnet.make_nn()
nnet.trainn_NN()
