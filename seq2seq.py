"""Sequence-to-sequence RNN implementation inheriting from the base RNN class."""

import numpy as np
from network import RNN


class Seq2Seq(RNN):
    """A sequence-to-sequence recurrent neural network.

    Inherits from the base RNN class and adds encoding and decoding capabilities
    with shared weights for both understanding and producing sequences.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def encode(self, inputs, h_prev=None):
        """Encode an input sequence into a context vector (final hidden state).

        Args:
            inputs: list of input column vectors, one per timestep.
            h_prev: initial hidden state of shape (hidden_size, 1). Defaults to zeros.

        Returns:
            Final hidden state after processing the sequence, shape (hidden_size, 1).
        """
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        _, hs, _, _ = self.forward(inputs, h_prev)
        return hs[len(inputs) - 1]

    def decode(self, context, start_token, max_length=50, end_token=None):
        """Decode a sequence from a context vector using greedy decoding.

        Args:
            context: context vector (final hidden state from encoder), shape (hidden_size, 1).
            start_token: one-hot vector for the start token, shape (input_size, 1).
            max_length: maximum length of generated sequence.
            end_token: index of end token to stop generation (optional).

        Returns:
            List of predicted token indices.
        """
        outputs = []
        h = context
        x = start_token
        for _ in range(max_length):
            x_proj = self.input_layer.forward(x)
            h = self.hidden_layer.forward(x_proj, h)
            y = self.output_layer.forward(h)
            probs = self.softmax(y)
            predicted_token_idx = np.argmax(probs)
            outputs.append(predicted_token_idx)
            if end_token is not None and predicted_token_idx == end_token:
                break
            # create one-hot for next input
            next_x = np.zeros((self.input_size, 1))
            next_x[predicted_token_idx] = 1
            x = next_x
        return outputs