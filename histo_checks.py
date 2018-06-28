import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os


from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15

path = os.getcwd()

import warnings
warnings.simplefilter("ignore", Warning)

def make_histogram(fileN='updates.dat'):
    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    figname = path + '/plots/Histogram_' + fileN[:-4] + '.pdf'

    dataL = np.loadtxt(path + '/data/' + fileN)
    pos = dataL[:,:-1][dataL[:,-1] == 1]
    neg = dataL[:,:-1][dataL[:,-1] == 0]
    

    number_inputs = pos.shape[1]
    for i in range(number_inputs):
        plt.hist(pos[:,i], 20, density=True, facecolor='None', edgecolor='black', linewidth=1.)
        plt.hist(neg[:,i], 20, density=True, facecolor='None', edgecolor='blue', linewidth=1.2)

    fig.set_tight_layout(True)
    pl.savefig(figname)
    return
