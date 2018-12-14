import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os
from top_sus_NN_Reg import *


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

def make_histogram(fileN='Esymdep.out'):
    Qhit = 2

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    figname = path + '/plots/Histogram_' + fileN[:-4] + '.pdf'

    dataL = np.loadtxt(path + '/data/' + fileN)
    pos = dataL[:,:-1][np.abs(dataL[:,-1]) == Qhit].flatten()
    neg = dataL[:,:-1][dataL[:,-1] == 0].flatten()
    

    #number_inputs = pos.shape[1]
    #for i in range(number_inputs):
    plt.hist(pos, 20, density=True, facecolor='red', edgecolor='red', linewidth=1., alpha=.2)
    plt.hist(neg, 20, density=True, facecolor='blue', edgecolor='blue', linewidth=1.2, alpha=.2)

    fig.set_tight_layout(True)
    pl.savefig(figname)
    return


def accuarcy_plot(fileN, metaF):
    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    figname = path + '/plots/Accuracy_plot_' + fileN[:-4] + '.pdf'
    
    dataL = np.loadtxt(path + '/data/' + fileN)
    inputs = dataL[:,:-1]
    outputs = dataL[:, -1]
    
    NNet = ImportGraph(metaF, fileN)
    predict = NNet.run_yhat(inputs)
    plt.plot(np.abs(outputs), predict, 'bo', alpha=0.1, ms=2)
    
    xvals = np.linspace(-10, 10, 10)
    plt.plot(xvals, xvals, 'r', lw=1)
    
    plt.ylim([-3, 10])
    plt.xlim([-3, 10])
    fig.set_tight_layout(True)
    pl.savefig(figname)

    return
