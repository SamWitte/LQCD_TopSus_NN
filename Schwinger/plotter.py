import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os
from schwinger_NN_Reg import *


from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15

path = os.getcwd()

import warnings
#warnings.simplefilter("ignore", Warning)


# I dont remember what this is....
#def make_histogram(fileN='Esymdep.out'):
#    Qhit = 2
#
#    fig = plt.figure(figsize=(8., 6.))
#    ax = plt.gca()
#    figname = path + '/plots/Histogram_' + fileN[:-4] + '.pdf'
#
#    dataL = np.loadtxt(path + '/data/' + fileN)
#    pos = dataL[:,:-1][np.abs(dataL[:,-1]) == Qhit].flatten()
#    neg = dataL[:,:-1][dataL[:,-1] == 0].flatten()
#
#
#    #number_inputs = pos.shape[1]
#    #for i in range(number_inputs):
#    plt.hist(pos, 20, density=True, facecolor='red', edgecolor='red', linewidth=1., alpha=.2)
#    plt.hist(neg, 20, density=True, facecolor='blue', edgecolor='blue', linewidth=1.2, alpha=.2)
#
#    fig.set_tight_layout(True)
#    pl.savefig(figname)
#    return


def accuarcy_plot(fileN, metaF):
    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    figname = path + '/plots/Accuracy_plot_' + fileN[:-4] + '.pdf'
    
    input_data_file = path + '/NN_data/Schwinger_' + fileN + '_input_data.dat'
    output_data_file = path + '/NN_data/Schwinger_' + fileN + '_output_data.dat'
    input_data = np.loadtxt(input_data_file)
    output_data = np.loadtxt(output_data_file)
    
    maxVal = np.max(output_data)
    minVal = np.min(output_data)
    
    lattice_dim = 18

    if fileN == 'fermion_determinant':
        output_features = 1
        round = False
        yshift = 100.
        alph = 0.15
    elif fileN == 'top_charge':
        output_features = 1
        round = True
        yshift = 2.
        alph = 0.15
    elif fileN == 'pion_correlator':
        output_features = lattice_dim
        round = False
        yshift = 2.
        alph = 0.02
    
    input_features = len(input_data) / len(output_data)
    input_data = input_data.reshape(len(output_data), input_features)

    NNet = ImportGraph(metaF, fileN)
    predict = NNet.run_yhat(input_data, round=round)
    plt.plot(output_data, predict, 'bo', alpha=0.2, ms=4)
    
    xvals = np.linspace(minVal - 1, maxVal + 1, 10)
    plt.plot(xvals, xvals, 'r', lw=1)
    if round:
        highV = xvals + 0.5
        lowV = xvals - 0.5
        plt.fill_between(xvals, lowV, highV, color='r', alpha=alph)

    plt.ylim([minVal - yshift, maxVal + yshift])
    plt.xlim([minVal - 0.1, maxVal  + 0.1])
    fig.set_tight_layout(True)
    pl.savefig(figname)

    return
