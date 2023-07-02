# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:49:09 2023

@author: Michaela Walterová
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import pandas as pd

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
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
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

zetas = []

for i,file_name in enumerate(file_names[:-1]):
    data = pd.read_excel(dir_name+file_name, header=headers[i])

    zeta = data['zeta']
    alpha = data['alpha']
    beta = data['beta']
    rigs = data['rigidity (Pa)']/1e9
    etas = np.log10(data['viscosity (Pa*s)'])
    temps = data['Temperature (C)']
    JCvar = data['eta^-alpha*mu^(1-alpha)']
    taus = data['tau']

    logzeta_sigma = data['sigma log zeta']
    rigs_sigma = data['sigma rigidity']/1e9
    alpha_sigma = data['sigma alpha']
    beta_sigma = data['sigma beta']
    JCvar_sigma = data['sigma eta^-alpha*mu^(1-alpha)']
    etas_sigma = data['sigma viscosity']*0
    taus_sigma = data['sigma tau']
    
    method = data['Method']
    
    zetas.append(zeta)
    
    for j,m in enumerate(method):
        if 'TO' in m:
            marker = 'o'
        else:
            marker = '^'
       
        if labels[i] == 'Qu et al. (2021), Ol+Px' and etas[j]==20:
            ax[0].errorbar(rigs[j], alpha[j] ,
                  yerr=alpha_sigma[j], xerr=rigs_sigma[j], 
                  fmt='none', ecolor='gray', elinewidth=0.3, zorder=0)
            ax[0].scatter(rigs[j], alpha[j], c=colors[i], marker=marker, 
                          edgecolors='k', linewidths=0.5)

        else:
            ax[0].errorbar(rigs[j], alpha[j] ,
                  yerr=alpha_sigma[j], xerr=rigs_sigma[j], 
                  fmt='none', ecolor='gray', elinewidth=0.3, zorder=0)
            ax[0].scatter(rigs[j], alpha[j], c=colors[i], marker=marker, 
                          edgecolors='k', linewidths=0.5)

            ax[1].errorbar(rigs[j], np.log10(zeta[j]), 
                  yerr=logzeta_sigma[j], xerr=rigs_sigma[j], 
                  fmt='none', ecolor='gray', elinewidth=0.3, zorder=0)
            ax[1].scatter(rigs[j], np.log10(zeta[j]), c=colors[i], marker=marker, 
                          edgecolors='k', linewidths=0.5)

line1 = ax[0].scatter(100, 0.1, c='w', label='torsion oscillations', edgecolors='k', marker='o', linewidths=0.5)
line2 = ax[0].scatter(100, 0.1, c='w', label='microcreep', edgecolors='k', marker='^', linewidths=0.5)
line3 = ax[0].scatter(100, 0.1, c=colors[0], label=labels[0], edgecolors='k', marker='s', linewidths=0.5)
line4 = ax[0].scatter(100, 0.1, c=colors[1], label=labels[1], edgecolors='k', marker='s', linewidths=0.5)
line5 = ax[0].scatter(100, 0.1, c=colors[2], label=labels[2], edgecolors='k', marker='s', linewidths=0.5)
line6 = ax[0].scatter(100, 0.1, c=colors[3], label=labels[3], edgecolors='k', marker='s', linewidths=0.5)
line7 = ax[0].scatter(100, 0.1, c=colors[4], label=labels[4], edgecolors='k', marker='s', linewidths=0.5)

#ax.legend(fontsize=14)

legend1 = ax[0].legend(handles=[line1, line2], loc=3, fontsize=14)
ax[0].add_artist(legend1)
ax[0].legend(handles=[line3, line4, line5, line6], loc=2, fontsize=14)

ax[0].set_xlim(20, 73)
ax[0].grid(c='lightgray', ls='dotted')
ax[0].set_xlabel(r'$\mu$ [GPa]')
ax[0].set_ylabel(r'$\alpha$')

ax[1].set_xlim(20, 73)
ax[1].grid(c='lightgray', ls='dotted')
ax[1].set_xlabel(r'$\mu$ [GPa]')
ax[1].set_ylabel(r'$\log\; \zeta$')

# zetas = np.concatenate(zetas)
# zetas_true = [z for z in zetas if z>1e-4 and z<1e4]
# print(np.mean(np.log10(zetas_true)),np.median(zetas_true))

plt.subplots_adjust(wspace=0.2)

plt.savefig('mu_alphazeta_all.png', dpi=300, bbox_inches = 'tight')