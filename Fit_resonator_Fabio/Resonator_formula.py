#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
"""
import numpy as np

""" All formulas used come from E. Metral USPAS lecture, availible here :
    http://emetral.web.cern.ch/emetral/USPAS09course/WakeFieldsAndImpedances.pdf """

def Resonator_longitudinal_real(x_array, Rt1, Q1, fres1):
    return Rt1/(Q1*Q1*((x_array/fres1)-(fres1/x_array))**2 + 1)

def Resonator_longitudinal_imag(x_array, Rt1, Q1, fres1):
    return -Rt1*Q1*((x_array/fres1)-
             (fres1/x_array))/(Q1*Q1*((x_array/fres1)-(fres1/x_array))**2 + 1)
    
    
def Resonator_longitudinal_imp(frequencies, Rt, Q, resonant_frequency):
    Zl = Rt / (1 + 1j*Q*(
            frequencies/resonant_frequency - resonant_frequency/frequencies))
    return Zl

def Resonator_transverse_real(x_array, Rt1, Q1, fres1):
    return (fres1/x_array)*Rt1/(Q1*Q1*((x_array/fres1)-
                                (fres1/x_array))**2 + 1)

def Resonator_transverse_imag(x_array, Rt1, Q1, fres1):
    return -(fres1/x_array)*Rt1*Q1*((x_array/fres1)-
             (fres1/x_array))/(Q1*Q1*((x_array/fres1)-(fres1/x_array))**2 + 1)
    
def Resonator_transverse_imp(frequencies, Rt, Q, resonant_frequency):
    Zt = resonant_frequency / frequencies * Rt / (1 + 1j*Q*(
            frequencies/resonant_frequency - resonant_frequency/frequencies))
    return Zt

def n_Resonator_longitudinal_real(x_array, *pars):
    """ x_array : frequencies used in broadband resonator formula
    pars : either list or array with the 3 needed parameters per resonator (Rt, Q, fres)
    pars must look like this [Rt1, Q1, fres1, ..., Rtn, Qn, fresn] """
    Z = 0
    for i in range(int(len(pars)/3)):
        Z += Resonator_longitudinal_real(x_array, *pars[3*i:3*i+3]) 
    return Z

def n_Resonator_longitudinal_imag(x_array, *pars):
    """ x_array : frequencies used in broadband resonator formula
    pars : either list or array with the 3 needed parameters per resonator (Rt, Q, fres)
    pars must look like this [Rt1, Q1, fres1, ..., Rtn, Qn, fresn] """
    Z = 0
    for i in range(int(len(pars)/3)):
        Z += Resonator_longitudinal_imag(x_array, *pars[3*i:3*i+3]) 
    return Z

def n_Resonator_longitudinal_imp(frequencies, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rt, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Zt = 0
    for resonator in dict_params.keys():
        Zt += Resonator_longitudinal_imp(frequencies, *dict_params[resonator]) 
    return Zt

def n_Resonator_transverse_real(x_array, *pars):
    """ x_array : frequencies used in broadband resonator formula
    pars : either list or array with the 3 needed parameters per resonator (Rt, Q, fres)
    pars must look like this [Rt1, Q1, fres1, ..., Rtn, Qn, fresn] """
    Z = 0
    for i in range(int(len(pars)/3)):
        Z += Resonator_transverse_real(x_array, *pars[3*i:3*i+3]) 
    return Z


def n_Resonator_transverse_imag(x_array, *pars):
    """ x_array : frequencies used in broadband resonator formula
    pars : either list or array with the 3 needed parameters per resonator (Rt, Q, fres)
    pars must look like this [Rt1, Q1, fres1, ..., Rtn, Qn, fresn] """
    Z = 0
    for i in range(int(len(pars)/3)):
        Z += Resonator_transverse_imag(x_array, *pars[3*i:3*i+3]) 
    return Z

def n_Resonator_transverse_imp(frequencies, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rt, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Zt = 0
    for resonator in dict_params.keys():
        Zt += Resonator_transverse_imp(frequencies, *dict_params[resonator]) 
    return Zt


def Resonator_longitudinal_wake(times, Rt, Q, resonant_frequency):
    """Be careful, units for this formula are Rt [Ohm/m], Q [], fres [hz]"""
    return (Rt*2*np.pi*resonant_frequency)*np.exp(-2*np.pi*resonant_frequency*times/(2*Q))/Q*(np.cos(2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q))))*times)+(-2*np.pi*resonant_frequency/(2*Q)/2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q)))))*np.sin(2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q))))*times))

def Resonator_transverse_wake(times, Rt, Q, resonant_frequency):
    """Be careful, units for this formula are Rt [Ohm/m], Q [], fres [hz]"""
    return ((Rt*2*np.pi*resonant_frequency)/(Q*np.sqrt(np.abs(1-(1/(4*Q*Q))))))*np.exp(-1*np.pi*resonant_frequency*times/Q)*np.sin(2*np.pi*resonant_frequency*np.sqrt(np.abs(1-(1/(4*Q*Q))))*times)

def n_Resonator_longitudinal_wake(times, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rt, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Wt = 0
    for resonator in dict_params.keys():
        Wt += Resonator_longitudinal_wake(times, *dict_params[resonator])
    return Wt

def n_Resonator_transverse_wake(times, dict_params):
    """
    frequencies : frequencies used in broadband resonator formula
    dict_params : dict where each key corresponds to a single resonator
    resonator parameters are stored in a list following this format : 
        [Rt, Q, resonant_frequency]
    Example dict_params = {1: [1000000.0, 1, 1000000000.0],
                           2 : [2000000.0, 1, 500000000.0]}
    """
    Wt = 0
    for resonator in dict_params.keys():
        Wt += Resonator_transverse_wake(times, *dict_params[resonator])
    return Wt