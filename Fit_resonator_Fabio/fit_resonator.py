#!/usr/bin/env python
# coding: utf-8

# # Import

# In[37]:


from __future__ import division
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.optimize import differential_evolution
from Resonator_formula import *
import warnings
import time as ti


# # Define input and output paths, select studied element

# In[38]:


imp_path='/home/sjoly/cernbox/PS_CST_Tsutsui_wake_imp_files2'
wake_path='/home/sjoly/cernbox/PS_IW_model_full/Wakes/Transverse'
save_plot_path='/home/sjoly/cernbox/Resonators/GA_Resonator_fit'

element = 'PE.KFA21'

#Define frequency window to fit a resonator, frequencies in Hz
def f_window_test(impedance_file, component, fmin, fmax):
    impedance_file = impedance_file.loc[(impedance_file['f'] >= fmin) & (impedance_file['f'] <= fmax)]
    frequency_data = impedance_file['f'].to_numpy() #frequencies in GHz
    impedance_data = np.array(impedance_file['Re({})'.format(component)] + 1j*impedance_file['Im({})'.format(component)])

    return frequency_data, impedance_data

#Functions for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(pars):
    
    """
    Complex sum of squared errors
    """
    dict_params = pars_to_dict(pars)
    #warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    sumOfSquaredErrorReal = np.nansum((impedance_data.real - 
                                       n_Resonator_transverse_imp(frequencies, dict_params).real) ** 2)
    sumOfSquaredErrorImag = np.nansum((impedance_data.imag - 
                                       n_Resonator_transverse_imp(frequencies, dict_params).imag) ** 2)
    return (sumOfSquaredErrorReal + sumOfSquaredErrorImag)


#Bounds on parameters are set in generate_Initial_Parameters() below
def generate_Initial_Parameters(parameterBounds):

    """ This function allows to generate adequate initial guesses to use in the minimization algorithm.
    In this step we do not necessarily need to have converged results. Approximate results are enough.
    We use the complex sum of squared errors (sumOfSquaredError) as a loss function, it can be changed
    depending on the problem type.
    One can tweak the number of iterations (maxiter) to stop the algorithm earlier for faster computing time.
    All the arguments are detailed on this page :
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    Setting workers= -1 means all CPUs available will be used for the computation.
    """
    
    result = differential_evolution(sumOfSquaredError,
                                     parameterBounds, workers=-1)#,seed=3)
    return result.x


# Computations

impedance_file = pd.read_csv(imp_path+'/'+element+'_CST_2015_10GHz.imp', 
                             sep='\s+', index_col=False, 
                             names=["f", "Re(Zdipx)", "Im(Zdipx)", 
                                    "Re(Zdipy)", "Im(Zdipy)",
                                    "Re(Zquadx)", "Im(Zquadx)",
                                    "Re(Zquady)", "Im(Zquady)"], header=0)

#Studied component
component = "Zdipy"

#Number of resonators
Nres = 3

#x,y_array correspond to all the data (x=0 excluded to avoid dividing by 0)
#frequencies in GHz and impedances in Ohm/m


frequencies, impedance_data = f_window_test(impedance_file, component, 1e-15, np.inf)

""" Bounds on resonators parameters, it's possible to manually set them as well.
Bounds have this format [(Rt_min, Rtmax), (Q_min, Q_max), (fres_min, fres_max)].
ParameterBounds allows us to manually add a resonator with desired parameters """
bounds = [(-7e5, 7e5), (0, 20), (0e9, 10e9)]
parameterBounds = [(-7e5, 7e5), (0, 20), (1e8, 10e9)] + (Nres-1)*bounds
def pars_to_dict(pars):
    dict_params = {}
    for i in range(int(len(pars)/3)):
        dict_params[i] = pars[3*i:3*i+3]
    return dict_params
###############################################################################

t0 = ti.clock()

#Generate initial parameter values using our bounds
initialParameters = generate_Initial_Parameters(parameterBounds)


t1 = ti.clock()


#Curve fit the impedance using initial parameters from genetic algorithm

pars = minimize(sumOfSquaredError, x0=initialParameters, bounds=parameterBounds,
                method='Powell', options={'maxfev': 100000, 'disp': True})
pars = pars.x

dict_params = pars_to_dict(pars)
dict_gen = pars_to_dict(initialParameters)

for i in range(Nres):
    print('Resonator {}'.format(i+1))
    print('Rt = {:.2e} [Ohm/m], Q = {:.2f}, fres = {:.1e} [Hz]'.format(*dict_params[i]))
    print('-'*60)

t2 = ti.clock()

print('\n')
print("Elapsed time for genetic algorithm={:.2f} s".format(t1-t0))
print("Elapsed time for curve fit algorithm={:.2f} s".format(t2-t1))

#%%
###############################################################################

#Plot real part of impedance
sns.set_style(style="darkgrid", rc={"xtick.bottom" : True, "ytick.left" : True})
sns.set_context('talk')

fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(15,8))
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams['axes.titlesize']=14    
plt.rcParams['axes.labelsize']=14

ax0.plot(frequencies, impedance_data.real, "black", label='CST data')
ax0.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_params).real, 'royalblue',
         lw = 3, label='Genetic algorithm + Resonator fit')
ax0.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_gen).real,
         lw = 3, linestyle='--', color='firebrick', label='Genetic algorithm')

ax1.plot(frequencies, impedance_data.real, "black", label='CST data')
ax1.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_params).real, 'royalblue',
         lw = 3, label='Genetic algorithm + Resonator fit')
ax1.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_gen).real,
         lw = 3, linestyle='--', color='firebrick', label='Genetic algorithm')

ax1.set_xscale('log')

ax0.set_xlabel('f [GHz]')
ax1.set_xlabel('f [GHz]')

ax0.set_ylabel('$Z_{transverse}$ [$\Omega$ /m]')
ax1.set_ylabel('$Z_{transverse}$ [$\Omega$ /m]')

ax0.set_title('Fit of the real part of ' + element + ' impedance, ' + component, fontsize=20)

ax0.legend(loc='best', fontsize=14)
ax1.legend(loc='best', fontsize=14)

fig.tight_layout()
#plt.savefig(save_plot_path+'/'+element+'_'component+'_real.png')


#Plot imaginary part of impedance

fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(15,8))
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams['axes.titlesize']=14    
plt.rcParams['axes.labelsize']=14

ax0.plot(frequencies, impedance_data.imag, "black", label='CST data')
ax0.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_params).imag, 'royalblue',
         lw = 3, label='Genetic algorithm + Resonator fit')
ax0.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_gen).imag,
         lw = 3, linestyle='--', color='firebrick', label='Genetic algorithm')

ax1.plot(frequencies, impedance_data.imag, "black", label='CST data')
ax1.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_params).imag, 'royalblue',
         lw = 3, label='Genetic algorithm + Resonator fit')
ax1.plot(frequencies, n_Resonator_transverse_imp(frequencies, dict_gen).imag,
         lw = 3, linestyle='--', color='firebrick', label='Genetic algorithm')

ax1.set_xscale('log')

ax0.set_xlabel('f [GHz]')
ax1.set_xlabel('f [GHz]')

ax0.set_ylabel('$Z_{transverse}$ [$\Omega$ /m]')
ax1.set_ylabel('$Z_{transverse}$ [$\Omega$ /m]')

ax0.set_title('Fit of the imaginary part of ' + element + ' impedance, ' + component, fontsize=20)

ax0.legend(loc='best', fontsize=14)
ax1.legend(loc='best', fontsize=14)

fig.tight_layout()
#plt.savefig(save_plot_path+'/'+element+'_'component+'_imag.png')

#%%

#Load data from wake file
wake_data = pd.read_csv(wake_path+'/'+element+'_CST_2015_10GHz.wake', 
                        sep='\s+', index_col=False, names=["t", "Wdipx", "Wdipy",
                                                           "Wquadx", "Wquady"])
#Choose which component to study
Wt = component.replace('Z', 'W')

wake_data = wake_data.loc[wake_data['t'] >= 0]
t_array = np.array(wake_data['t'])
Wt_array = np.array(wake_data[Wt])

#It's also possible to fit the wake with this part, by default it's not used
"""
def sumOfSquaredErrorWake(pars):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((Wt_array - n_Resonator_transverse_wake(t_array, *pars)) ** 2)



def generate_Initial_Parameters():
    # min and max used for bounds

    parameterBounds = [(-70, 70), (0, 20), (1e-1, 2e-1)] + [(-70, 70), (0, 20), (0, 10)]*(Nres-1)
    #parameterBounds = [(-70, 70), (0, 20), (0, 10)]*(Nres)

    # "seed" the numpy random number generator for repeatable results
    result1 = differential_evolution(sumOfSquaredErrorWake, parameterBounds,seed=3)

    return result1.x




t0 = ti.clock()

# generate initial parameter values
initialParametersWake = generate_Initial_Parameters()

t1 = ti.clock()


# curve fit the test data
pars, pcov_wake = curve_fit(n_Resonator_transverse_wake, t_array, Wt_array*1e15, initialParametersWake, 
                                 bounds=bounds_fit, maxfev=100000)
perr_wake = np.nansum((n_Resonator_transverse_wake(t_array, *pars) - Wt_array)**2)

t2 = ti.clock()


print("Error for wake fit {}".format(perr_wake))

for i in range(Nres):
    print('Resonator '+str(i+1))
    print('Rt = {} [Ohm/m], Q = {}, fres = {} [GHz]'.format(1e4*pars[3*i], pars[3*i+1], pars[3*i+2]))
    print('-'*60)

print("Elapsed time for genetic algorithm={} s".format(t1-t0))
print("Elapsed time for curve fit algorithm={} s".format(t2-t1))

df_comp = save_parameters(element, df_comp, pars)

"""


sns.set_style(style="darkgrid", rc={"xtick.bottom" : True, "ytick.left" : True})
sns.set_context('talk')


fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(15,8))
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams['axes.titlesize']=14    
plt.rcParams['axes.labelsize']=14

ax0.plot(t_array, Wt_array, "black", lw=2, label='CST data')
ax0.plot(t_array, n_Resonator_transverse_wake(t_array, dict_params)*1e-15,
         lw = 3, label='Genetic algorithm + Resonator fit')
ax0.plot(t_array, n_Resonator_transverse_wake(t_array, dict_gen)*1e-15,
         lw = 3, linestyle='--', color='firebrick', label='Genetic algorithm')

ax1.plot(t_array, Wt_array, "black", lw=2, label='CST data')
ax1.plot(t_array, n_Resonator_transverse_wake(t_array, dict_params)*1e-15,
         lw = 3, label='Genetic algorithm + Resonator fit')
ax1.plot(t_array, n_Resonator_transverse_wake(t_array, dict_gen)*1e-15,
         lw = 3, linestyle='--', color='firebrick', label='Genetic algorithm')
ax1.set_xscale('log')

ax0.set_xlabel('t (s)')
ax1.set_xlabel('t (s)')

ax0.set_ylabel('$W_{transverse}$ [V/pC/m]')
ax1.set_ylabel('$W_{transverse}$ [V/pC/m]')

ax0.set_xlim(-5e-10, 10e-9)

ax0.set_title("Wake function from resonators parameters, "+element+", short range, " + Wt, fontsize=16)

ax0.legend(loc='best', fontsize=14)
ax1.legend(loc='best', fontsize=14)

fig.tight_layout()
#plt.savefig(save_plot_path+'/'+element+'_'+Wt+'_.png')