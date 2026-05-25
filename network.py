"""Simple recurrent neural network layers and training utilities."""

import numpy as np


# ============================================================
# Layers
# ============================================================

class InputLayer:
    """Project input vectors into hidden state space."""

    def __init__(self, input_size, hidden_size):

        # Xavier-style initialization
        self.W = np.random.randn(
            hidden_size,
            input_size
        ) / np.sqrt(input_size)

        self.b = np.zeros((hidden_size, 1))

    def forward(self, X):

        return np.dot(self.W, X) + self.b


class HiddenLayer:
    """Shared recurrent hidden layer."""

    def __init__(self, hidden_size):

        # Stable recurrent initialization
        self.W_hh = np.random.randn(
            hidden_size,
            hidden_size
        ) / np.sqrt(hidden_size)

        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x_proj, h_prev):

        return np.tanh(
            np.dot(self.W_hh, h_prev)
            + x_proj
            + self.b_h
        )


class OutputLayer:
    """Linear output layer."""

    def __init__(self, hidden_size, output_size):

        self.W = np.random.randn(
            output_size,
            hidden_size
        ) / np.sqrt(hidden_size)

        self.b = np.zeros((output_size, 1))

    def forward(self, h):

        return np.dot(self.W, h) + self.b


# ============================================================
# Shared-state RNN
# ============================================================

class RNN:
    """
    Shared-state recurrent neural network.

    IMPORTANT:
    Perception and production use the SAME recurrent dynamics.

    The only separate components are the output heads.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
    ):

        # Shared encoder dynamics
        self.input_layer = InputLayer(
            input_size,
            hidden_size
        )

        self.hidden_layer = HiddenLayer(
            hidden_size
        )

        # Perception head
        self.output_layer = OutputLayer(
            hidden_size,
            output_size
        )

        # Production head
        self.production_output_layer = OutputLayer(
            hidden_size,
            input_size
        )

        self.hidden_size = hidden_size
        self.input_size = input_size

    # ========================================================
    # Utilities
    # ========================================================

    def softmax(self, x):

        x = x - np.max(x)

        exp = np.exp(x)

        return exp / np.sum(exp, keepdims=True)

    # ========================================================
    # Perception
    # ========================================================

    def forward(self, inputs, h_prev):
        """
        Perception forward pass.
        """

        xs = {}
        hs = {}
        ys = {}
        probs_dict = {}

        hs[-1] = h_prev

        for t in range(len(inputs)):

            x_proj = self.input_layer.forward(
                inputs[t]
            )

            h = self.hidden_layer.forward(
                x_proj,
                hs[t - 1]
            )

            y = self.output_layer.forward(h)

            probs = self.softmax(y)

            xs[t] = inputs[t]
            hs[t] = h
            ys[t] = y
            probs_dict[t] = probs

        return xs, hs, ys, probs_dict

    # ========================================================
    # Production
    # ========================================================

    def production_output(self, h):
        """
        Continuous production output.

        IMPORTANT:
        No softmax.
        Production targets are continuous spectra.
        """

        return self.production_output_layer.forward(h)

    def generate_production_sequence(
        self,
        final_perceptual_h,
        seq_length,
        teacher_forcing_inputs=None,
        teacher_forcing_ratio=1.0,
        generation_noise=0.01,
    ):
        """
        Shared-state autoregressive generation.

        Production reuses the SAME recurrent dynamics
        as perception.
        """

        productions = []

        prod_states = {
            "pre": {},
            "post": {}
        }

        h_prod = final_perceptual_h.copy()

        for t in range(seq_length):

            # Store pre-update hidden state
            prod_states["pre"][t] = h_prod.copy()

            # Generate continuous spectral vector
            output = self.production_output_layer.forward(
                h_prod
            )

            productions.append(output)

            # ====================================================
            # Scheduled sampling
            # ====================================================

            use_teacher = (
                teacher_forcing_inputs is not None
                and np.random.rand() < teacher_forcing_ratio
            )

            if use_teacher:

                feedback = teacher_forcing_inputs[t]

            else:

                feedback = output.copy()

                # Prevent exact fixed-point collapse
                feedback += (
                    np.random.randn(*feedback.shape)
                    * generation_noise
                )

            # ====================================================
            # Shared recurrent dynamics
            # ====================================================

            x_proj = self.input_layer.forward(
                feedback
            )

            # Slight amplification prevents
            # hidden-state contraction
            x_proj *= 1.2

            # IMPORTANT:
            # SAME recurrent unit as perception
            h_new = self.hidden_layer.forward(
                x_proj,
                h_prod
            )

            prod_states["post"][t] = h_new.copy()

            h_prod = h_new

        return productions, prod_states

    # ========================================================
    # Perception backprop
    # ========================================================

    def backward(
        self,
        xs,
        hs,
        probs,
        targets=None,
        lr=1e-3,
        freeze_perception=False,
    ):

        dWxh = np.zeros_like(self.input_layer.W)
        dbx = np.zeros_like(self.input_layer.b)

        dWhh = np.zeros_like(self.hidden_layer.W_hh)
        dbh = np.zeros_like(self.hidden_layer.b_h)

        dWhy = np.zeros_like(self.output_layer.W)
        dby = np.zeros_like(self.output_layer.b)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(xs))):

            dh = np.zeros((self.hidden_size, 1))

            if targets is not None:

                dy = probs[t] - targets[t]

                dWhy += np.dot(
                    dy,
                    hs[t].T
                )

                dby += dy

                dh += (
                    np.dot(
                        self.output_layer.W.T,
                        dy
                    )
                    + dh_next
                )

            else:

                dh = dh_next.copy()

            dh_raw = (
                1 - hs[t] ** 2
            ) * dh

            dbh += dh_raw

            dWhh += np.dot(
                dh_raw,
                hs[t - 1].T
            )

            dWxh += np.dot(
                dh_raw,
                xs[t].T
            )

            dbx += dh_raw

            dh_next = np.dot(
                self.hidden_layer.W_hh.T,
                dh_raw
            )

        # Gradient clipping
        for d in [
            dWxh,
            dWhh,
            dWhy,
            dby,
            dbh,
            dbx,
        ]:
            np.clip(d, -5, 5, out=d)

        if not freeze_perception:

            self.input_layer.W -= lr * dWxh
            self.input_layer.b -= lr * dbx

            self.hidden_layer.W_hh -= lr * dWhh
            self.hidden_layer.b_h -= lr * dbh

            self.output_layer.W -= lr * dWhy
            self.output_layer.b -= lr * dby

    # ========================================================
    # Production backprop
    # ========================================================

    def backward_production_only(
        self,
        final_h,
        production_targets,
        lr=1e-3,
        freeze_perception=False,
        teacher_forcing_ratio=0.5,
    ):
        """
        Backprop through autoregressive production.

        IMPORTANT:
        Gradients flow into the SHARED recurrent unit.
        """

        productions, prod_states = (
            self.generate_production_sequence(
                final_h,
                len(production_targets),
                teacher_forcing_inputs=production_targets,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
        )

        # ====================================================
        # Production head gradients
        # ====================================================

        dWprod = np.zeros_like(
            self.production_output_layer.W
        )

        dby_prod = np.zeros_like(
            self.production_output_layer.b
        )

        # ====================================================
        # Shared recurrent gradients
        # ====================================================

        dWxh = np.zeros_like(
            self.input_layer.W
        )

        dbx = np.zeros_like(
            self.input_layer.b
        )

        dWhh = np.zeros_like(
            self.hidden_layer.W_hh
        )

        dbh = np.zeros_like(
            self.hidden_layer.b_h
        )

        dh_next = np.zeros(
            (self.hidden_size, 1)
        )

        # ====================================================
        # BPTT
        # ====================================================

        for t in reversed(
            range(len(production_targets))
        ):

            output = productions[t]

            h_pre = prod_states["pre"][t]

            target = production_targets[t]

            # MSE gradient
            dy = (
                2
                * (output - target)
                / target.size
            )

            # ====================================================
            # Production output gradients
            # ====================================================

            dWprod += np.dot(
                dy,
                h_pre.T
            )

            dby_prod += dy

            # ====================================================
            # Backprop into shared hidden state
            # ====================================================

            dh = (
                np.dot(
                    self.production_output_layer.W.T,
                    dy
                )
                + dh_next
            )

            dh_raw = (
                1 - h_pre ** 2
            ) * dh

            # Shared recurrent gradients
            dbh += dh_raw

            prev_h = (
                prod_states["pre"][t - 1]
                if t > 0
                else final_h
            )

            dWhh += np.dot(
                dh_raw,
                prev_h.T
            )

            # Shared input gradients
            feedback_input = (
                production_targets[t]
            )

            dWxh += np.dot(
                dh_raw,
                feedback_input.T
            )

            dbx += dh_raw

            dh_next = np.dot(
                self.hidden_layer.W_hh.T,
                dh_raw
            )

        # ====================================================
        # Gradient clipping
        # ====================================================

        for d in [
            dWprod,
            dby_prod,
            dWxh,
            dbx,
            dWhh,
            dbh,
        ]:
            np.clip(d, -5, 5, out=d)

        # ====================================================
        # Update production head
        # ====================================================

        self.production_output_layer.W -= (
            lr * dWprod
        )

        self.production_output_layer.b -= (
            lr * dby_prod
        )

        # ====================================================
        # Update shared recurrent system
        # ====================================================

        if not freeze_perception:

            self.input_layer.W -= lr * dWxh
            self.input_layer.b -= lr * dbx

            self.hidden_layer.W_hh -= lr * dWhh
            self.hidden_layer.b_h -= lr * dbh

    # ========================================================
    # Utilities
    # ========================================================

    def get_params(self):

        return {

            "Wxh": self.input_layer.W.copy(),
            "bx": self.input_layer.b.copy(),

            "Whh": self.hidden_layer.W_hh.copy(),
            "bh": self.hidden_layer.b_h.copy(),

            "Why": self.output_layer.W.copy(),
            "by": self.output_layer.b.copy(),

            "Wprod": (
                self.production_output_layer.W.copy()
            ),

            "bprod": (
                self.production_output_layer.b.copy()
            ),
        }

    def set_params(self, params):

        self.input_layer.W = params["Wxh"].copy()
        self.input_layer.b = params["bx"].copy()

        self.hidden_layer.W_hh = params["Whh"].copy()
        self.hidden_layer.b_h = params["bh"].copy()

        self.output_layer.W = params["Why"].copy()
        self.output_layer.b = params["by"].copy()

        self.production_output_layer.W = (
            params["Wprod"].copy()
        )

        self.production_output_layer.b = (
            params["bprod"].copy()
        )

    def predict(self, inputs, h_prev=None):

        if h_prev is None:

            h_prev = np.zeros(
                (self.hidden_size, 1)
            )

        _, _, _, probs = self.forward(
            inputs,
            h_prev
        )

        return probs