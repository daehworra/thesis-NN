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


def train_model(epochs=10000, lr=1e-2):
    """Train an RNN and return the best model found during training.

    The best model is selected by the lowest loss observed on the training
    examples during the run.

    Args:
        epochs: number of training iterations.
        lr: learning rate for parameter updates.

    Returns:
        best_rnn: RNN instance with parameters from the best observed loss.
        best_loss: lowest loss value observed.
        best_epoch: epoch index where the best loss occurred.
    """
    rnn = RNN(input_size=len(erb_bins), hidden_size=50, output_size=len(main_language.word_labels))
    best_loss = np.inf
    best_params = rnn.get_params()
    best_epoch = -1

    for epoch in range(epochs):
        # 1. sample a random word and use its matching utterance
        sequence, word = main_language.random_utterance(length=1)
        inputs = utterance_to_input(sequence)
        targets = [prefix_target_distribution(word, t) for t in range(len(inputs))]

        # 2. forward pass
        h0 = np.zeros((50, 1))
        xs, hs, ys, probs = rnn.forward(inputs, h0)

        # 3. compute loss using softmax probabilities
        loss = 0
        for t in range(len(probs)):
            loss += -np.sum(targets[t] * np.log(probs[t] + 1e-9))

        # 4. backward pass and parameter update
        rnn.backward(xs, hs, probs, targets, lr=lr)

        if loss < best_loss:
            best_loss = loss
            best_params = rnn.get_params()
            best_epoch = epoch

        if epoch % 100 == 0:
            print(epoch, loss)

    print("best loss:", best_loss, "at epoch", best_epoch)

    best_rnn = RNN(input_size=len(erb_bins), hidden_size=50, output_size=len(main_language.word_labels))
    best_rnn.set_params(best_params)
    return best_rnn, best_loss, best_epoch


def test_single_random_word(rnn, top_k=None):
    """Evaluate the model on a single random word and print timestep probabilities.

    Args:
        rnn: trained RNN instance.
        top_k: if provided, only print the top-k predicted words at each timestep.

    Returns:
        sequence: the input utterance as a list of formant tuples.
        word: the Word object selected for testing.
        probs: dictionary of softmax probabilities at each timestep.
    """
    sequence, word = main_language.random_utterance(length=1)
    inputs = utterance_to_input(sequence)
    probs = rnn.predict(inputs)

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

    return sequence, word, probs


if __name__ == "__main__":
    best_rnn, best_loss, best_epoch = train_model()
    test_single_random_word(best_rnn)
