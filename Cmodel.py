#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pickle5 as pickle
from collections import OrderedDict

def get_data():
    with open("data.pickle", 'rb') as f: 
        return pickle.load(f)

def get_model():
    with open("model.pickle", 'rb') as f: 
        return pickle.load(f)
        
def save_model():
    global model
    model = eegNet
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t) 
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
        
    def cross_entropy_error(slef, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        if t.size == y.size:
            t = t.argmax(axis=1)
             
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)
        return dx


# In[69]:


class Adam:

    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])      
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


# In[63]:


class MultiLayerNet:
    
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])
        self.lastLayer = SoftmaxWithLoss()
        
    def __init_weight(self, weight_init_std):
    
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
        
    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        
        return grads


# In[52]:


class Model:
    def __init__(self):
        self.data = get_data()
    
    def makemodel():
        eegNet3 = MultiLayerNet(10, [10, 5], 3)
    
    def sep(data):
        x = np.zeros(10, int)
        t = np.zeros(3, int)
        
        for i in range(len(data)):
            x_p = data[i, :10]
            x = np.vstack((x, x_p))
            t_p = data[i, 10:13]
            t = np.vstack((t, t_p))

        x = np.delete(x, 0, 0)
        x = np.delete(x, 0, 0)
        t = np.delete(t, 0, 0)
        t = np.delete(t, 0, 0)
        return x, t
    
    def train():
        x, t = sep(data)
        learning_rate = 0.1
        step = 10000
        optimizer = Adam()

        for i in range(step):
            grads = eegNet3.gradient(x, t)
            optimizer.update(eegNet3.params, grads)


# In[47]:


data = get_data()


# In[48]:


print(len(data))


# In[76]:


eegNet = get_model()


# In[49]:


def sep(data):
    x = np.zeros(10, int)
    t = np.zeros(3, int)
    
    for i in range(len(data)):
        x_p = data[i, :10]
        x = np.vstack((x, x_p))
        t_p = data[i, 10:13]
        t = np.vstack((t, t_p))

    x = np.delete(x, 0, 0)
    x = np.delete(x, 0, 0)
    t = np.delete(t, 0, 0)
    t = np.delete(t, 0, 0)
    return x, t


# In[66]:


x, t = sep(data)


# In[65]:


def train():
    x, t = sep(data)
    learning_rate = 0.01
    step = 20000
    optimizer = Adam()

    for i in range(step):
        grads = eegNet.gradient(x, t)
        optimizer.update(eegNet.params, grads)


# In[70]:


eegNet = MultiLayerNet(10, [30, 20, 15], 3)


# In[71]:


train()


# In[78]:


correct = 0
for idx in range(len(x)):
    if (np.argmax(eegNet.predict(np.array([x[idx]])))) == (np.argmax(t[idx])):
        correct = correct+1

print(correct/len(x))


# In[75]:


save_model()


# In[ ]:




