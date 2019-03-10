import numpy as np

class SYF_ANN:
    def __init__(self, x, y):
        self.x = x

        neurons = 128
        self.lr = 0.50
        self.n_layers=3

        ip_dim = x.shape[1]
        op_dim = y.shape[1]
        
        self.W=[]
        self.B=[]
        self.A=[None for i in range(self.n_layers)]

        for i in range(self.n_layers):
            if i==0:
                self.W.append(np.random.randn(ip_dim, neurons))
                self.B.append(np.zeros((1, neurons)))
            elif i==(self.n_layers-1):
                self.B.append(np.zeros((1, op_dim)))
                self.W.append(np.random.randn(neurons, op_dim))
            else:
                self.W.append(np.random.randn(neurons, neurons))
                self.B.append(np.zeros((1, neurons)))

        self.y = y

    def fit(self,**kwargs):
        if 'epochs' in kwargs:
            epochs=kwargs['epochs']
        else:
            epochs = 1500
        for x in range(epochs):
            print("Epoch:",x, end=' ')
            self.feedforward()
            self.backprop()

    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))

    def sigmoid_derv(self,s):
        return s * (1 - s)

    def softmax(self,s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def cross_entropy(self,pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res/n_samples

    def error(self,pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def feedforward(self):
        for i in range(self.n_layers):
            if i==0:
                self.A[i]=self.sigmoid(np.dot(self.x,self.W[i])+self.B[i])
            else:
                self.A[i]=self.sigmoid(np.dot(self.A[i-1],self.W[i])+self.B[i])

        
    def backprop(self):
        loss = self.error(self.A[-1], self.y)
        print('Error :', loss)

        a3_delta = self.cross_entropy(self.A[2], self.y) # w3
        z2_delta = np.dot(a3_delta, self.W[2].T)
        a2_delta = z2_delta * self.sigmoid_derv(self.A[1]) # w2
        z1_delta = np.dot(a2_delta, self.W[1].T)
        a1_delta = z1_delta * self.sigmoid_derv(self.A[0]) # w1

        self.W[2] -= self.lr * np.dot(self.A[1].T, a3_delta)
        self.B[2] -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.W[1] -= self.lr * np.dot(self.A[0].T, a2_delta)
        self.B[1] -= self.lr * np.sum(a2_delta, axis=0)
        self.W[0] -= self.lr * np.dot(self.x.T, a1_delta)
        self.B[0] -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.A[-1].argmax()
    
