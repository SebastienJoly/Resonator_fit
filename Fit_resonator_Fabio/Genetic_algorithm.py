#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:33:41 2020

@author: sjoly
"""
import numpy as np
from scipy.optimize import differential_evolution

def generate_Initial_Parameters(parameterBounds, Minimization_function):

    """ This function allows to generate adequate initial guesses to use in the minimization algorithm.
    In this step we do not necessarily need to have converged results. Approximate results are enough.
    We use the complex sum of squared errors (sumOfSquaredError) as a loss function, it can be changed
    depending on the problem type.
    One can tweak the number of iterations (maxiter) to stop the algorithm earlier for faster computing time.
    All the arguments are detailed on this page :
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    Setting workers= -1 means all CPUs available will be used for the computation.
    """
    
    result = differential_evolution(Minimization_function,
                                     parameterBounds, workers=-1)#,seed=3)
    return result.x