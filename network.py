import numpy as np

class InputLayer:
    def __init__(self, input_size, hidden_size):
        self.W = np.random.randn(hidden_size, input_size) * 0.01
        self.b = np.zeros((hidden_size, 1))

    def forward(self, X):
        return np.dot(self.W, X) + self.b
    
class HiddenLayer:
    def __init__(self, hidden_size):
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x_proj, h_prev):
        return np.tanh(np.dot(self.W_hh, h_prev) + x_proj + self.b_h)
    
class OutputLayer:
    def __init__(self, hidden_size, output_size):
        self.W = np.random.randn(output_size, hidden_size) * 0.01
        self.b = np.zeros((output_size, 1))

    def forward(self, h):
        return np.dot(self.W, h) + self.b
     
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_layer = InputLayer(input_size, hidden_size)
        self.hidden_layer = HiddenLayer(hidden_size)
        self.output_layer = OutputLayer(hidden_size, output_size)

        self.hidden_size = hidden_size

    def softmax(self, x):
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp, keepdims=True)
    
    def forward(self, inputs, h_prev):
        xs, hs, ys, probs_dict = {}, {}, {}, {}
        hs[-1] = h_prev

        for t in range(len(inputs)):
            x_proj = self.input_layer.forward(inputs[t])
            h = self.hidden_layer.forward(x_proj, hs[t-1])
            y = self.output_layer.forward(h)
            probs = self.softmax(y)

            xs[t] = inputs[t]
            hs[t] = h
            ys[t] = y
            probs_dict[t] = probs

        return xs, hs, ys, probs_dict
        
    def backward(self, xs, hs, ys, targets, lr=1e-3):
        
        dWxh = np.zeros_like(self.input_layer.W)
        dbx = np.zeros_like(self.input_layer.b)

        dWhh = np.zeros_like(self.hidden_layer.W_hh)
        dbh = np.zeros_like(self.hidden_layer.b_h)

        dWhy = np.zeros_like(self.output_layer.W)
        dby = np.zeros_like(self.output_layer.b)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(xs))):

            
            dy = ys[t] - targets[t]

            dWhy += np.dot(dy, hs[t].T)
            dby += dy

           
            dh = np.dot(self.output_layer.W.T, dy) + dh_next

            
            dh_raw = (1 - hs[t] ** 2) * dh

            dbh += dh_raw

            
            dWhh += np.dot(dh_raw, hs[t-1].T)

           
            dWxh += np.dot(dh_raw, xs[t].T)
            dbx += dh_raw

           
            dh_next = np.dot(self.hidden_layer.W_hh.T, dh_raw)

       
        for d in [dWxh, dWhh, dWhy, dbh, dby, dbx]:
            np.clip(d, -5, 5, out=d)

        
        self.input_layer.W -= lr * dWxh
        self.input_layer.b -= lr * dbx

        self.hidden_layer.W_hh -= lr * dWhh
        self.hidden_layer.b_h -= lr * dbh

        self.output_layer.W -= lr * dWhy
        self.output_layer.b -= lr * dby