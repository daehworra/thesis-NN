"""Training and evaluation utilities for the RNN word recognizer."""

import numpy as np
from network import RNN
from languages import main_language

erb_bins = np.linspace(4, 30, 30)


def gaussian(x, mu, sigma=1.0):
    """Generate a Gaussian response over ERB bins."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def formants_to_spectrum(f1, f2, erb_bins=erb_bins):
    """Convert F1/F2 formant values into a normalized spectral vector."""
    spec_f1 = gaussian(erb_bins, f1)
    spectrum = spec_f1.copy()
    if f2 != 0:
        spec_f2 = gaussian(erb_bins, f2)
        spectrum += spec_f2

    spectrum /= np.sum(spectrum)
    return spectrum.reshape(-1, 1)


def utterance_to_input(sequence, erb_bins=erb_bins):
    """Turn a sequence of formant pairs into RNN input vectors."""
    return [formants_to_spectrum(f1, f2, erb_bins) for (f1, f2) in sequence]


def onehot(label, language=main_language):
    """Create a one-hot label vector for a given word label."""
    vec = np.zeros((len(language.word_labels), 1))
    vec[language.word_labels.index(label)] = 1
    return vec


def prefix_target_distribution(word, step, language=main_language):
    """Build a prefix-aware target distribution for a given timestep.

    Args:
        word: Word object containing the true phoneme sequence.
        step: timestep index in the utterance.
        language: language object with vocabulary labels.

    Returns:
        A normalized distribution over all vocabulary labels, where only words
        consistent with the current phoneme prefix receive probability mass.
    """
    prefix = word.phonseq[: step + 1]
    vec = np.zeros((len(language.word_labels), 1))
    valid_indices = [
        i
        for i, label in enumerate(language.word_labels)
        if label.startswith(prefix)
    ]
    if not valid_indices:
        vec[language.word_labels.index(word.phonseq)] = 1
        return vec

    prob = 1.0 / len(valid_indices)
    for i in valid_indices:
        vec[i] = prob
    return vec


def production_target_vector(inputs):
    """Build a 30-dimensional target vector for the utterance.

    This returns the average spectral vector of the entire input sequence.
    The production head learns to predict this utterance-level representation
    from the final perceptual hidden state.
    """
    return np.mean(np.hstack(inputs), axis=1, keepdims=True)


def train_perception(rnn, epochs=10000, lr=1e-2, print_every=100):
    """Train only the perception head and shared recurrent state."""
    best_loss = np.inf
    best_params = rnn.get_params()
    best_epoch = -1

    for epoch in range(epochs):
        sequence, word = main_language.random_utterance(length=1)
        inputs = utterance_to_input(sequence)
        targets = [prefix_target_distribution(word, t) for t in range(len(inputs))]

        h0 = np.zeros((rnn.hidden_size, 1))
        xs, hs, ys, probs = rnn.forward(inputs, h0)

        loss = 0
        for t in range(len(probs)):
            loss += -np.sum(targets[t] * np.log(probs[t] + 1e-9))

        rnn.backward(
            xs,
            hs,
            probs,
            targets,
            lr=lr,
            production_target=None,
            freeze_production=True,
        )

        if loss < best_loss:
            best_loss = loss
            best_params = rnn.get_params()
            best_epoch = epoch

        if epoch % print_every == 0:
            print("perception", epoch, loss)

    rnn.set_params(best_params)
    print("best perception loss:", best_loss, "at epoch", best_epoch)
    return rnn, best_loss, best_epoch


def train_production(rnn, epochs=2000, lr=1e-2, freeze_perception=True, print_every=100):
    """Train the production head to predict each phoneme's spectral vector at each timestep.

    For the CORRECT word, we draw an actual utterance and convert it to 30-dimensional vectors.
    The target at each timestep is the actual spectral vector for that phoneme.
    freeze_perception=True keeps the recurrent representation fixed.
    """
    best_loss = np.inf
    best_params = rnn.get_params()
    best_epoch = -1

    for epoch in range(epochs):
        sequence, word = main_language.random_utterance(length=1)
        inputs = utterance_to_input(sequence)
        h0 = np.zeros((rnn.hidden_size, 1))
        xs, hs, ys, probs = rnn.forward(inputs, h0)

        # Create production targets: one for each timestep (the actual input at that timestep)
        production_targets = inputs  # List of 30-dimensional vectors, one per timestep

        loss = 0
        for t in range(len(inputs)):
            prod_output = rnn.production_output(hs[t])
            loss += np.mean((prod_output - inputs[t]) ** 2)

        rnn.backward(
            xs,
            hs,
            probs,
            targets=None,
            lr=lr,
            production_targets=production_targets,
            production_loss_weight=1.0,
            freeze_perception=freeze_perception,
            freeze_production=False,
        )

        if loss < best_loss:
            best_loss = loss
            best_params = rnn.get_params()
            best_epoch = epoch

        if epoch % print_every == 0:
            print("production", epoch, loss)

    rnn.set_params(best_params)
    print("best production loss:", best_loss, "at epoch", best_epoch)
    return rnn, best_loss, best_epoch


def train_model(
    perception_epochs=10000,
    production_epochs=10000,
    lr=1e-2,
):
    """Train perception first, then train the production head from the final state."""
    rnn = RNN(
        input_size=len(erb_bins),
        hidden_size=50,
        output_size=len(main_language.word_labels),
    )

    rnn, perception_loss, perception_epoch = train_perception(
        rnn, epochs=perception_epochs, lr=lr
    )

    rnn, production_loss, production_epoch = train_production(
        rnn, epochs=production_epochs, lr=lr, freeze_perception=True
    )

    return rnn, perception_loss, perception_epoch, production_loss, production_epoch


def test_single_random_word(rnn, top_k=None):
    """Evaluate the model on a single random word and print timestep probabilities.

    Args:
        rnn: trained RNN instance.
        top_k: if provided, only print the top-k predicted words at each timestep.

    Returns:
        sequence: the input utterance as a list of formant tuples.
        word: the Word object selected for testing.
        probs: dictionary of softmax probabilities at each timestep.
        production_outputs: list of 30-dimensional vectors produced by the production head.
        production_targets: list of 30-dimensional target vectors (actual inputs).
    """
    sequence, word = main_language.random_utterance(length=1)
    inputs = utterance_to_input(sequence)
    h0 = np.zeros((rnn.hidden_size, 1))
    xs, hs, ys, probs = rnn.forward(inputs, h0)

    print(f"Testing random word: {word.phonseq}")
    print(f"Utterance sequence: {sequence}")
    print()

    for t in range(len(inputs)):
        prob = probs[t].flatten()
        prefix = word.phonseq[: t + 1]
        print(f"Timestep {t}, prefix='{prefix}':")
        if top_k is None:
            for label, p in zip(main_language.word_labels, prob):
                print(f"  {label}: {p:.4f}")
        else:
            top_indices = np.argsort(prob)[::-1][:top_k]
            for i in top_indices:
                print(f"  {main_language.word_labels[i]}: {prob[i]:.4f}")
        print()

    production_outputs = [rnn.production_output(hs[t]).flatten() for t in range(len(inputs))]
    production_targets = [inp.flatten() for inp in inputs]

    print("Production head outputs (one per timestep):")
    for t in range(len(production_outputs)):
        print(f"Timestep {t}:")
        print(np.array2string(production_outputs[t], precision=4, separator=', '))
    print()

    print("Production target vectors (actual spectral inputs):")
    for t in range(len(production_targets)):
        print(f"Timestep {t}:")
        print(np.array2string(production_targets[t], precision=4, separator=', '))
    print()

    return sequence, word, probs, production_outputs, production_targets


def test_production_for_word(rnn, word_label=None):
    """Display the production head output for one chosen vocabulary word, timestep by timestep."""
    if word_label is None:
        sequence, word = main_language.random_utterance(length=1)
    else:
        try:
            word = next(w for w in main_language.words if w.phonseq == word_label)
        except StopIteration:
            raise ValueError(f"Unknown word label: {word_label}")
        sequence = word.utterance(length=1)

    inputs = utterance_to_input(sequence)
    h0 = np.zeros((rnn.hidden_size, 1))
    _, hs, _, _ = rnn.forward(inputs, h0)
    
    production_outputs = [rnn.production_output(hs[t]).flatten() for t in range(len(inputs))]
    production_targets = [inp.flatten() for inp in inputs]

    print(f"Production output for word: {word.phonseq}")
    print(f"Utterance sequence: {sequence}")
    print()
    
    print("Production head outputs (one per timestep):")
    for t in range(len(production_outputs)):
        print(f"Timestep {t}:")
        print(np.array2string(production_outputs[t], precision=4, separator=', '))
    print()

    print("Production target vectors (actual spectral inputs):")
    for t in range(len(production_targets)):
        print(f"Timestep {t}:")
        print(np.array2string(production_targets[t], precision=4, separator=', '))
    print()

    return word, production_outputs, production_targets


if __name__ == "__main__":
    best_rnn, best_loss, best_epoch, best_prod_loss, best_prod_epoch = train_model()
    test_single_random_word(best_rnn, top_k=3)
    test_production_for_word(best_rnn, word_label="pupupu")
