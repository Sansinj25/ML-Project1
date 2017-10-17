# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import *




def least_squares_GD(y, tx, intial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss,gradient=compute_gradient(y,tx,w)
        w=w-gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, losses




def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        #this for only gives minibatch_y and tx, computes one iteration only per call creates minibatch randomly
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size): 
            loss,gradient=compute_gradient(minibatch_y,minibatch_tx,w)
            #we use gradient from new minibatch for each iteration of the gradient
            w=w-gamma*gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return w, losses




def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w=np.linalg.solve(a, b)
    error=(y-np.dot(tx,w))
    loss=compute_mse(y,tx,w)
    #mae=compute_mae(y,tx,w)
    return w, loss
    #return mae,w
    
    
    

def ridge_regression(y, tx,lambda_):    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w=np.linalg.solve(a, b)
    #error=(y-np.dot(tx,w))
    loss=compute_mse(y,tx,w)
    #mae=compute_mae(y,tx,w)
    return w, loss
    #return mae,w

    
    
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return w, loss




def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return w, loss




#########################################
#########################################

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error=y-np.dot(tx,w)
    gradient=(-1/len(y))*np.dot(tx.T,error)
    losses=compute_loss(y, tx, w)
    return losses,gradient

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    error=y-np.dot(tx,w)
    gradient=(-1/len(y))*np.dot(tx.T,error)
    losses=compute_loss(y, tx, w)
    return losses,gradient



