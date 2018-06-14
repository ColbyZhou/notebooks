# Vanilla RNN Demo

import numpy as np
import random

# Hyperparameters
hidden_size = 3
input_size = 4
seq_length = 4
# inputs & outputs example
inputs = [2,1,2,0,3,1,2] # a sequence of input (word index)
targets = [1,2,0,3,1,2,2] # a sequence of output (label)

#model parameters
Wxh = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Whf = np.random.randn(input_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
bf = np.zeros((input_size, 1)) # output bias

def getLoss(inputs, targets, hprev):
    """
    get Gradients
    hprev is last hidhen layer output
    return loss, gradients and last hidhen layer output
    """
    xs, hs, fs, ps = {}, {}, {}, {} # all vaues,key: time t, value: vectors of time t
    hprev = np.zeros((hidden_size,1)) # previous hidden layer output
    hs[-1] = np.copy(hprev) # last hidhen layer output
    loss = 0

    # 1. forword propagation
    for t in xrange(len(inputs)):
        xs[t] = np.zeros((input_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Whh, hs[t-1]) + np.dot(Wxh, xs[t]) + bh)
        fs[t] = np.dot(Whf, hs[t]) + bf
        ps[t] = np.exp(fs[t]) / np.sum(np.exp(fs[t]))
        loss += -np.log(ps[t][targets[t], 0])

    # 2. backward propagation
    dWxh, dWhh, dWhf = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Whf)
    dbh, dbf = np.zeros_like(bh), np.zeros_like(bf)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(xrange(len(inputs))):
        df = np.copy(ps[t])
        df[targets[t]] -= 1
        dWhf += np.dot(df, hs[t].T)
        dbf += df
        dh = np.dot(Whf.T, df) + dhnext # + dhnext, means back-prop twice at one time
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dhnext = np.dot(Whh.T, dhraw)
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)

    # 3. cut gradients to avoid exploding
    for dparam in [dWxh, dWhh, dWhf, dbh, dbf]:
        np.clip(dparam, -5,5, out=dparam)

    return loss, dWxh, dWhh, dWhf, dbh, dbf, hs[len(inputs)-1]


# memory variables for Adagrad
mWxh, mWhh, mWhf = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Whf)
mbh, mbf = np.zeros_like(bh), np.zeros_like(bf)

learning_rate = 0.1

# random data
data = np.random.randint(0, input_size, size = 10000)
iter = 0
p = 0
hprev = np.zeros((hidden_size,1))

while iter < 10000:
    
    # initialization hprev and data pointer
    if p+seq_length+1 >= len(data):
        p = 0
        hprev = np.zeros((hidden_size,1))
    inputs = data[p:p+seq_length]
    targets = data[p:p+seq_length+1]
    
    # update model parameters through Adagrad
    loss, dWxh, dWhh, dWhf, dbh, dbf, hprev = getLoss(inputs, targets, hprev)
    
    if iter % 100 == 0:
        print(str(iter) + ' loss: ' + str(loss))
    
    for param, dparam, mem in zip( [Wxh, Whh, Whf, bh, bf],
                         [dWxh, dWhh, dWhf, dbh, dbf],
                         [mWxh, mWhh, mWhf, mbh, mbf] ):
        mem += dparam * dparam
        param += -learning_rate * dparam / (np.sqrt(mem) + 1e-8)
    
    # iteration over
    p += seq_length
    iter += 1

        

def gradient_check(inputs, targets, hprev):

    num_to_check = 100
    delta = 1e-5

    # get current parameters
    loss, dWxh, dWhh, dWhf, dbh, dbf, hprev = getLoss(inputs, targets, hprev)
    for param, dparam, name in zip( [Wxh, Whh, Whf, bh, bf],
                         [dWxh, dWhh, dWhf, dbh, dbf],
                         ['Wxh', 'Whh', 'Whf', 'bh', 'bf'] ):
        for i in range(0, num_to_check):
            idx = random.randint(0, param.size)

            # store old param
            old_val = param.flat[idx]

            # get L(x + h)
            param.flat[idx] = old_val + delta
            L_val_1,_,_,_,_,_,_ = getLoss(inputs, targets, hprev)
            # get L(x - h)
            param.flat[idx] = old_val + delta
            L_val_2,_,_,_,_,_,_ = getLoss(inputs, targets, hprev)
            
            # restor old parameter
            param.flat[idx] = old_val

            # get analytic gradient
            analytic_grad = dparam.flat[idx]
            # compute numerical gradient
            numerical_grad = L_val_2 - L_val_1 / (2 * delta)

            # relative error
            err = abs(numerical_grad - analytic_grad) / abs(numerical_grad + analytic_grad)
            print(errr)
            
    











    
    
