"""Simple recurrent neural network layers and training utilities."""

import numpy as np


class InputLayer:
    """Project input vectors into the RNN hidden state space."""

    def __init__(self, input_size, hidden_size):
        self.W = np.random.randn(hidden_size, input_size) * 0.01
        self.b = np.zeros((hidden_size, 1))

    def forward(self, X):
        """Compute the input-to-hidden affine projection.

        Args:
            X: Input vector of shape (input_size, 1).

        Returns:
            Projected input of shape (hidden_size, 1).
        """
        return np.dot(self.W, X) + self.b


class HiddenLayer:
    """Recurrent hidden layer with tanh activation."""

    def __init__(self, hidden_size):
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x_proj, h_prev):
        """Compute the next hidden state.

        Args:
            x_proj: input projection of shape (hidden_size, 1).
            h_prev: previous hidden state of shape (hidden_size, 1).

        Returns:
            Next hidden state of shape (hidden_size, 1).
        """
        return np.tanh(np.dot(self.W_hh, h_prev) + x_proj + self.b_h)


class OutputLayer:
    """Compute the network output logits from hidden state."""

    def __init__(self, hidden_size, output_size):
        self.W = np.random.randn(output_size, hidden_size) * 0.01
        self.b = np.zeros((output_size, 1))

    def forward(self, h):
        """Map hidden state to raw output logits."""
        return np.dot(self.W, h) + self.b
     
class RNN:
    """A simple recurrent neural network with perception and production heads.

    The perception head classifies each timestep into vocabulary labels.
    The production head maps the final hidden state into a 30-dimensional
    utterance vector representation.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_layer = InputLayer(input_size, hidden_size)
        self.hidden_layer = HiddenLayer(hidden_size)
        self.output_layer = OutputLayer(hidden_size, output_size)
        self.production_layer = OutputLayer(hidden_size, input_size)
        self.hidden_size = hidden_size

    def softmax(self, x):
        """Compute a numerically stable softmax over a vector."""
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp, keepdims=True)

    def production_output(self, h):
        """Compute the 30-dimensional production vector from the final hidden state."""
        return self.production_layer.forward(h)

    def forward(self, inputs, h_prev):
        """Run the network forward over a sequence of inputs.

        Args:
            inputs: list of input column vectors, one per timestep.
            h_prev: initial hidden state of shape (hidden_size, 1).

        Returns:
            xs: dictionary of input vectors by timestep.
            hs: dictionary of hidden states by timestep.
            ys: dictionary of raw logits by timestep.
            probs_dict: dictionary of softmax probabilities by timestep.
        """
        xs, hs, ys, probs_dict = {}, {}, {}, {}
        hs[-1] = h_prev

        for t in range(len(inputs)):
            x_proj = self.input_layer.forward(inputs[t])
            h = self.hidden_layer.forward(x_proj, hs[t - 1])
            y = self.output_layer.forward(h)
            probs = self.softmax(y)

            xs[t] = inputs[t]
            hs[t] = h
            ys[t] = y
            probs_dict[t] = probs

        return xs, hs, ys, probs_dict

    def backward(
        self,
        xs,
        hs,
        probs,
        targets=None,
        lr=1e-3,
        production_target=None,
        production_targets=None,
        production_loss_weight=1.0,
        freeze_perception=False,
        freeze_production=False,
    ):
        """Backpropagate through time and update model parameters.

        Args:
            xs: dictionary of inputs for each timestep.
            hs: dictionary of hidden states for each timestep.
            probs: dictionary of predicted softmax probabilities.
            targets: list of target distributions for each timestep, or None to skip perception updates.
            lr: learning rate.
            production_target: optional 30-dimensional production target for the final state (deprecated).
            production_targets: list of 30-dimensional targets for each timestep, one per phoneme.
            production_loss_weight: scale factor for production gradients.
            freeze_perception: when True, do not update perception parameters.
            freeze_production: when True, do not update the production head.
        """
        dWxh = np.zeros_like(self.input_layer.W)
        dbx = np.zeros_like(self.input_layer.b)

        dWhh = np.zeros_like(self.hidden_layer.W_hh)
        dbh = np.zeros_like(self.hidden_layer.b_h)

        dWhy = np.zeros_like(self.output_layer.W)
        dby = np.zeros_like(self.output_layer.b)

        dWprod = np.zeros_like(self.production_layer.W)
        dby_prod = np.zeros_like(self.production_layer.b)

        dh_next = np.zeros((self.hidden_size, 1))
        
        # Pre-compute production gradients for all timesteps if provided
        prod_grads = {}
        if production_targets is not None:
            for t in range(len(xs)):
                y_prod = self.production_layer.forward(hs[t])
                dprod = 2 * (y_prod - production_targets[t]) / production_targets[t].size
                dprod *= production_loss_weight
                prod_grads[t] = dprod
        elif production_target is not None:
            y_prod = self.production_layer.forward(hs[len(xs) - 1])
            dprod = 2 * (y_prod - production_target) / production_target.size
            dprod *= production_loss_weight
            prod_grads[len(xs) - 1] = dprod

        for t in reversed(range(len(xs))):
            dh = np.zeros((self.hidden_size, 1))
            
            if targets is not None:
                dy = probs[t] - targets[t]
                dWhy += np.dot(dy, hs[t].T)
                dby += dy
                dh += np.dot(self.output_layer.W.T, dy) + dh_next
            else:
                dh = dh_next.copy()

            # Add production gradient if available for this timestep
            if t in prod_grads:
                dprod = prod_grads[t]
                dWprod += np.dot(dprod, hs[t].T)
                dby_prod += dprod
                dh += np.dot(self.production_layer.W.T, dprod)

            dh_raw = (1 - hs[t] ** 2) * dh

            dbh += dh_raw
            dWhh += np.dot(dh_raw, hs[t - 1].T)
            dWxh += np.dot(dh_raw, xs[t].T)
            dbx += dh_raw

            dh_next = np.dot(self.hidden_layer.W_hh.T, dh_raw)

        # Clip gradients to stabilize training
        for d in [dWxh, dWhh, dWhy, dby, dby_prod, dbh, dbx]:
            np.clip(d, -5, 5, out=d)

        if not freeze_perception:
            self.input_layer.W -= lr * dWxh
            self.input_layer.b -= lr * dbx
            self.hidden_layer.W_hh -= lr * dWhh
            self.hidden_layer.b_h -= lr * dbh
            self.output_layer.W -= lr * dWhy
            self.output_layer.b -= lr * dby

        if not freeze_production and (production_target is not None or production_targets is not None):
            self.production_layer.W -= lr * dWprod
            self.production_layer.b -= lr * dby_prod

    def get_params(self):
        """Return a copy of the model parameters."""
        return {
            "Wxh": self.input_layer.W.copy(),
            "bx": self.input_layer.b.copy(),
            "Whh": self.hidden_layer.W_hh.copy(),
            "bh": self.hidden_layer.b_h.copy(),
            "Why": self.output_layer.W.copy(),
            "by": self.output_layer.b.copy(),
            "Wprod": self.production_layer.W.copy(),
            "bprod": self.production_layer.b.copy(),
        }

    def set_params(self, params):
        """Load model parameters from a saved dictionary."""
        self.input_layer.W = params["Wxh"].copy()
        self.input_layer.b = params["bx"].copy()
        self.hidden_layer.W_hh = params["Whh"].copy()
        self.hidden_layer.b_h = params["bh"].copy()
        self.output_layer.W = params["Why"].copy()
        self.output_layer.b = params["by"].copy()
        self.production_layer.W = params["Wprod"].copy()
        self.production_layer.b = params["bprod"].copy()

    def predict(self, inputs, h_prev=None):
        """Return softmax probabilities for a sequence without updating weights."""
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        _, _, _, probs = self.forward(inputs, h_prev)
        return probs
