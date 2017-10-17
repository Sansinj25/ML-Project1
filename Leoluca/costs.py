# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

def compute_mse(y, tx, w):
    #compute the loss by mse.
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

"""def compute_mse(y, tx, w):
    #Calculate the loss using mse 
    
    error=(y-np.dot(tx,w))
    loss=(1/(2*len(y)))*np.dot(error.T,error)
    return loss"""



def compute_mae(y, tx, w):
    error=(y-np.dot(tx,w))
    loss=(1/(len(y)))*(sum(abs(error)))
    return loss