import numpy as np
import pickle5 as pickle
from collections import OrderedDict

def get_data():
    # 데이터 로드
    with open("data.pickle", 'rb') as f: 
        return pickle.load(f)

def get_model():
    # 모델 로드
    with open("model.pickle", 'rb') as f: 
        return pickle.load(f)
        
def save_model():
    # 모델 저장
    global model
    model = eegNet
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def softmax(x):
    # Softmax 함수 구현
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class SoftmaxWithLoss:
    # 손실 함수와 Softmax 구현
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
        # 역전파 계산
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
        
    def cross_entropy_error(self, y, t):
        # 크로스 엔트로피 오차 계산
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        if t.size == y.size:
            t = t.argmax(axis=1)
             
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Relu:
    # ReLU 활성화 함수
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
    # Affine 계층 구현
    def __init__(self, W, b):
        self.W = W
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

class Adam:
    # Adam 옵티마이저
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

class MultiLayerNet:
    # 다층 신경망 구현
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std='relu', weight_decay_lambda=0):
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
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
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
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)], grads['b' + str(idx)] = self.layers['Affine' + str(idx)].dW, self.layers['Affine' + str(idx)].db
        
        return grads

# 데이터 및 모델 초기화
data = get_data()
eegNet = MultiLayerNet(10, [30, 20, 15], 3)

def sep(data):
    # 데이터 분리 함수
    x = np.zeros(10, int)
    t = np.zeros(3, int)
    
    for i in range(len(data)):
        x_p = data[i, :10]
        x = np.vstack((x, x_p))
        t_p = data[i, 10:13]
        t = np.vstack((t, t_p))

    x = np.delete(x, 0, 0)
    t = np.delete(t, 0, 0)
    return x, t

def train():
    # 모델 학습 함수
    x, t = sep(data)
    learning_rate = 0.01
    step = 20000
    optimizer = Adam()

    for i in range(step):
        grads = eegNet.gradient(x, t)
        optimizer.update(eegNet.params, grads)

# 학습 실행
train()

# 모델 평가
x, t = sep(data)
correct = 0
for idx in range(len(x)):
    if np.argmax(eegNet.predict(np.array([x[idx]]))) == np.argmax(t[idx]):
        correct += 1

print(correct / len(x))

# 모델 저장
save_model()
