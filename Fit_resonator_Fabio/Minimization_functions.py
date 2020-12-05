#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:34:10 2020

@author: sjoly
"""
import numpy as np

def pars_to_dict(pars):
    dict_params = {}
    for i in range(int(len(pars)/3)):
        dict_params[i] = pars[3*i:3*i+3]
    return dict_params

def sumOfSquaredError(pars, resonator_function, frequencies, impedance_data):
    
    """
    Complex sum of squared errors
    """
    dict_params = pars_to_dict(pars)
    #warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    sumOfSquaredErrorReal = np.nansum((impedance_data.real - 
                                       resonator_function(frequencies, dict_params).real) ** 2)
    sumOfSquaredErrorImag = np.nansum((impedance_data.imag - 
                                       resonator_function(frequencies, dict_params).imag) ** 2)
    return (sumOfSquaredErrorReal + sumOfSquaredErrorImag)