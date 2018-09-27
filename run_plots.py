from plotter import *
import os

path = os.getcwd()

# only used for quick visualization...
#make_histogram()

# accuarcy plot, should be linear
fileN = 'Esymdep.out'
hiddenN = 100
step = 1e-2
metaFile = path + '/MetaGraphs/Topological_jump__Hnodes_{:.0f}_Ssize_{:.0e}'.format(hiddenN, step)
accuarcy_plot(fileN, metaFile)
