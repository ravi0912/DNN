
# coding: utf-8

# In[1]:


#created by Rk
#Here we are creating helper function for different functions like
#relu ,sigmoid, relu_backward, sig_backward
#Here for every forward function we will check for the dimension and store the current value in cache
# which will help us in calculating backward prop efficiently
import numpy as np


# In[6]:


def relu(z):
    A = np.maximum(0,z)
    assert(A.shape == z.shape)
    cache = z
    return A,cache


# In[10]:


def sigmoid(z):
    A = 1/(1+np.exp(-z))
    cache = z
    return A,cache


# In[12]:


#dA -- post-activation gradient
#cache -- we stored the value of current node
#dZ -- Gradient of the cost wrt z
def relu_backward(dA, cache):
    z = cache
    dZ = np.array(dA,copy = True)
    dZ[z<=0] = 0
    assert(dZ.shape == z.shape)
    return dZ
    


# In[17]:


def sig_backward(dA,cache):
    z = cache
    s = 1/(1+np.exp(-z))
    dZ = dA*s*(1-s)
    assert(dZ.shape == z.shape)
    return dZ


