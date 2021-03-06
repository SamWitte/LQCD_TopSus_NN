from plotter import *
import os

path = os.getcwd()


# accuarcy plot, should be linear
fileN = 'pion_correlator' # ['fermion_determinant', 'top_charge']
hiddenN = 100
step = 1e-3
metaFile = path + '/MetaGraphs/Schwinger_' + fileN + '_Hnodes_{:.0f}_Ssize_{:.0e}'.format(hiddenN, step)
accuarcy_plot(fileN, metaFile)
