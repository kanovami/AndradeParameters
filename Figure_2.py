# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:25:08 2023

@author: Michaela Walterová
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
import scipy.odr as scodr

mpl.rc('font',family='Times New Roman',size=16)
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

#===================
# Define linear function
#===================
def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

#===================
# PLOTTING
#===================
# define figure parameters
fig, ax = plt.subplots(1, 1, figsize=(7,5))
fig.tight_layout()
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)

zetaN = []
rigsN = []
logzeta_sigmaN = []
rigs_sigmaN = []

JCvarN = []
betaN = []
JCvar_sigmaN = []
beta_sigmaN = []

logtausN = []
logtaasN = []
logtaus_sigmaN = []
logtaas_sigmaN = []

#===================
# Read Excel file:
#===================
dir_name = '../Datasets/'
file_names = ['Jackson_etal_2001.xlsx', './Jackson_etal_2004.xlsx', './Tan_etal_2001.xlsx',
              'Qu_etal_2021.xlsx', 'Barnhoorn_etal_2016.xlsx']
colors = ['k', 'gray', 'lightgray', 'skyblue', 'r']
labels = ['Jackson et al. (2002)', 'Jackson et al. (2004), with melt', 
          'Tan et al. (2001)', 'Qu et al. (2021), Ol+Px', 'Barnhoorn et al. (2016), MgO']

headers = [1, 1, 1, 0, 0]

for i,file_name in enumerate(file_names):
    data = pd.read_excel(dir_name+file_name, header=headers[i])

    taus = data['tau']
    taas = data['tau A']
    etas = np.log10(data['viscosity (Pa*s)'])

    if labels[i] == 'Qu et al. (2021), Ol+Px':
        indices = [i for i, n in enumerate(etas) if n!=20]
        taus = taus[indices]
        taas = taas[indices]
        taus_sigma = data['sigma tau'][indices]
        taas_sigma = data['sigma tau A'][indices]

    else:
        taus_sigma = data['sigma tau']
        taas_sigma = data['sigma tau A']

    logtaus_sigma = taus_sigma/taus/np.log(10)
    logtaas_sigma = taas_sigma/taas/np.log(10)

    ax.errorbar(np.log10(taus), np.log10(taas), 
              yerr=logtaas_sigma, xerr=logtaus_sigma, 
              fmt='none', ecolor='gray', elinewidth=0.3, zorder=0)
        
    logtausN = np.concatenate([logtausN, np.log10(taus)])
    logtaasN = np.concatenate([logtaasN, np.log10(taas)])
    
    ax.scatter(np.log10(taus), np.log10(taas), c=colors[i], label=labels[i], edgecolors='k', linewidths=0.5)

ax.set_xlabel(r'$\log\;\tau_{M}$ [s]')
ax.set_ylabel(r'$\log\;\tau_{A}$ [s]')
ax.grid(c='lightgray', ls='dotted')
ax.legend(fontsize=14)

xm = 5.2 #3.8
ax.plot(np.linspace(1, xm, 100), np.linspace(1, xm, 100), c='k', ls='--', lw=1)
ax.plot(np.linspace(1, xm, 100), 1+np.linspace(1, xm, 100), c='k', ls='--', lw=1)
ax.plot(np.linspace(1, xm, 100), -1+np.linspace(1, xm, 100), c='k', ls='--', lw=1)
ax.text(xm+0.05, 3.9+1.2, r'$\zeta=1$', rotation=5, color='k', size=14)
ax.text(xm+0.05, 5+1.2, r'$\zeta=10$', rotation=5, color='k', size=14)
ax.text(xm+0.05, 2.8+1.2, r'$\zeta=0.1$', rotation=5, color='k', size=14)

ax.set_xlim(0.6, 5.8)
ax.set_ylim(-2, 20)

plt.savefig('taum_taua_all.png', dpi=300, bbox_inches = 'tight')