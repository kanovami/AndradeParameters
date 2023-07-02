# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:01:29 2023

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

    beta  = data['beta']
    JCvar = data['eta^-alpha*mu^(1-alpha)']
    beta_sigma = data['sigma beta']
    JCvar_sigma = data['sigma eta^-alpha*mu^(1-alpha)']
    etas = np.log10(data['viscosity (Pa*s)'])

    if labels[i] == 'Qu et al. (2021), Ol+Px':
        indices = [i for i, n in enumerate(etas) if n!=20]
        
        beta  = beta[indices]
        JCvar = JCvar[indices]
        beta_sigma  = beta_sigma[indices]
        JCvar_sigma = JCvar_sigma[indices]
    
    method = data['Method']

    ax.errorbar(beta, JCvar, 
              yerr=JCvar_sigma, xerr=beta_sigma, 
              fmt='none', ecolor='k', elinewidth=0.3, zorder=0)

    for j,m in enumerate(method):
        if 'TO' in m:
            #colors.append('k')
            marker = 'o'
        else:
            #colors.append('r')
            marker = '^'
    ax.scatter(beta, JCvar, c=colors[i], label=labels[i], edgecolors='k', linewidths=0.5)
 
    if i<=4:
        betaN = np.concatenate([betaN, beta])
        JCvarN = np.concatenate([JCvarN, JCvar])
        beta_sigmaN = np.concatenate([beta_sigmaN, beta_sigma])
        JCvar_sigmaN = np.concatenate([JCvar_sigmaN, JCvar_sigma])
  
linear = scodr.Model(f)
#mydata = scodr.RealData(betaN, JCvarN, sx=beta_sigmaN, sy=JCvar_sigmaN)
mydata = scodr.RealData(betaN, JCvarN, sx=None, sy=None)

myodr = scodr.ODR(mydata, linear, beta0=[1., 2.])
myoutput = myodr.run()
myoutput.pprint()

p = myoutput.beta
rvar = myoutput.res_var

tvar = sum(abs(JCvarN - JCvarN.mean())**2)
rvar = sum(abs(JCvarN - f(p, betaN))**2)
print('R^2: ', 1-rvar/tvar)

#betax = np.linspace(0e-11, 0.07e-11, 100)
betax = np.linspace(0e-11, 3.5e-11, 100)
ax.plot(betax, f(p, betax), c='gray', lw=2, ls='--')

ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\eta^{-\alpha}\;\mu^{-(1-\alpha)}$')
ax.invert_yaxis()

ax.grid(c='lightgray', ls='dotted')
ax.legend(fontsize=14)
#ax.set_xlim(20, 73)
# ax.set_xticks(np.arange(0, 4.5, 0.5)*1e-11)
# ax.set_xticks(np.arange(0, 1.5, 0.5)*1e-11)

# line1 = ax.scatter(80, 0.1, c='w', label='torsion oscillations', edgecolors='k', marker='o', linewidths=0.5)
# line2 = ax.scatter(80, 0.1, c='w', label='microcreep', edgecolors='k', marker='^', linewidths=0.5)
# line3 = ax.scatter(80, 0.1, c='k', label='Jackson et al. (2002)', edgecolors='k', marker='s', linewidths=0.5)
# line4 = ax.scatter(80, 0.1, c='gray', label='Jackson et al. (2004)', edgecolors='k', marker='s', linewidths=0.5)
# line5 = ax.scatter(80, 0.1, c='lightgray', label='Tan et al. (2001)', edgecolors='k', marker='s', linewidths=0.5)
# #ax.legend(fontsize=14)

# legend1 = ax.legend(handles=[line1, line2], loc=3, fontsize=14)
# ax.add_artist(legend1)
# ax.legend(handles=[line3, line4, line5], loc=2, fontsize=14)

ax.text(2e-11, 1.73e-11, r'$y=0.42x+1.52\times10^{-12}$', color='gray', size=14)

plt.savefig('JC2011_all.png', dpi=300, bbox_inches = 'tight')