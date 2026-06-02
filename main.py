"""Training and evaluation utilities for the shared-state RNN."""

import numpy as np

from network import RNN
from languages import main_language


# ============================================================
# Acoustic representation
# ============================================================

erb_bins = np.linspace(4, 30, 30)


def gaussian(x, mu, sigma=1.0):
    """Gaussian spectral bump."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def formants_to_spectrum(f1, f2, erb_bins=erb_bins):
    """
    Convert F1/F2 formants into continuous spectral vectors.

    IMPORTANT:
    No sum normalization.
    Sum normalization forces vectors onto a simplex and
    strongly encourages autoregressive averaging collapse.
    """

    spec_f1 = gaussian(erb_bins, f1)

    spectrum = spec_f1.copy()

    if f2 != 0:
        spec_f2 = gaussian(erb_bins, f2)
        spectrum += spec_f2

    # Max normalization preserves shape while avoiding
    # magnitude explosion.
    spectrum /= np.max(spectrum)

    return spectrum.reshape(-1, 1)


def utterance_to_input(sequence, erb_bins=erb_bins):
    """Convert utterance sequence into spectral vectors."""

    return [
        formants_to_spectrum(f1, f2, erb_bins)
        for (f1, f2) in sequence
    ]


# ============================================================
# Perception targets
# ============================================================

def onehot(label, language=main_language):
    """One-hot vocabulary vector."""

    vec = np.zeros((len(language.word_labels), 1))

    vec[language.word_labels.index(label)] = 1

    return vec


def prefix_target_distribution(
    word,
    step,
    language=main_language,
):
    """
    Prefix-compatible target distribution.
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


# ============================================================
# Production utilities
# ============================================================

def temporal_difference_penalty(outputs):
    """
    Penalize identical consecutive outputs.

    This explicitly discourages fixed-point collapse.
    """

    penalty = 0.0

    for t in range(1, len(outputs)):

        diff = outputs[t] - outputs[t - 1]

        similarity = np.mean(diff ** 2)

        # Encourage DIFFERENCE between timesteps
        penalty += np.exp(-10 * similarity)

    return penalty


# ============================================================
# Perception training
# ============================================================

def train_perception(
    rnn,
    epochs=10000,
    lr=1e-2,
    print_every=100,
):
    """
    Train perception/classification system.
    """

    best_loss = np.inf
    best_params = rnn.get_params()
    best_epoch = -1

    for epoch in range(epochs):

        sequence, word = main_language.random_utterance(length=1)

        inputs = utterance_to_input(sequence)

        targets = [
            prefix_target_distribution(word, t)
            for t in range(len(inputs))
        ]

        h0 = np.zeros((rnn.hidden_size, 1))

        xs, hs, ys, probs = rnn.forward(inputs, h0)

        loss = 0.0

        for t in range(len(probs)):

            loss += -np.sum(
                targets[t] * np.log(probs[t] + 1e-9)
            )

        rnn.backward(
            xs,
            hs,
            probs,
            targets,
            lr=lr,
        )

        if loss < best_loss:

            best_loss = loss
            best_params = rnn.get_params()
            best_epoch = epoch

        if epoch % print_every == 0:

            print(
                f"[PERCEPTION] "
                f"epoch={epoch} "
                f"loss={loss:.6f}"
            )

    rnn.set_params(best_params)

    print(
        f"\nBest perception loss: "
        f"{best_loss:.6f} "
        f"at epoch {best_epoch}\n"
    )

    return rnn, best_loss, best_epoch


# ============================================================
# Production training
# ============================================================

def train_production(
    rnn,
    epochs=5000,
    lr=1e-3,
    freeze_perception=True,
    print_every=100,
):
    """
    Train autoregressive production system.

    Shared hidden dynamics are preserved.

    Key anti-collapse mechanisms:
    - continuous outputs
    - scheduled sampling
    - free-running generation
    - temporal diversity penalty
    """

    best_loss = np.inf
    best_params = rnn.get_params()
    best_epoch = -1

    for epoch in range(epochs):

        sequence, word = main_language.random_utterance(length=1)

        inputs = utterance_to_input(sequence)

        h0 = np.zeros((rnn.hidden_size, 1))

        xs, hs, ys, probs = rnn.forward(inputs, h0)

        final_h = hs[len(inputs) - 1]

        # Gradually reduce teacher forcing
        teacher_forcing_ratio = max(
            0.2,
            1.0 - (epoch / epochs)
        )

        productions, _ = rnn.generate_production_sequence(
            final_h,
            len(inputs),
            teacher_forcing_inputs=inputs,
            teacher_forcing_ratio=teacher_forcing_ratio,
            generation_noise=0.01,
        )

        # ====================================================
        # Reconstruction loss
        # ====================================================

        loss = 0.0

        for t in range(len(inputs)):

            output = productions[t]

            target = inputs[t]

            loss += np.sum(
                (output - target) ** 2
            )

        # ====================================================
        # Backpropagation
        # ====================================================

        rnn.backward_production_only(
            final_h,
            inputs,
            lr=lr,
            freeze_perception=freeze_perception,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # ====================================================
        # Save best model
        # ====================================================

        if loss < best_loss:

            best_loss = loss
            best_params = rnn.get_params()
            best_epoch = epoch

        # ====================================================
        # Diagnostics
        # ====================================================

        if epoch % print_every == 0:

            output_variance = np.mean([
                np.var(o)
                for o in productions
            ])

            timestep_difference = np.mean([
                np.mean((productions[t] - productions[t - 1]) ** 2)
                for t in range(1, len(productions))
            ])

            print(
                f"[PRODUCTION] "
                f"epoch={epoch} "
                f"loss={loss:.6f} "
                f"recon={loss:.6f} "
                f"tf={teacher_forcing_ratio:.3f} "
                f"var={output_variance:.6f} "
                f"delta={timestep_difference:.6f}"
            )

    rnn.set_params(best_params)

    print(
        f"\nBest production loss: "
        f"{best_loss:.6f} "
        f"at epoch {best_epoch}\n"
    )

    return rnn, best_loss, best_epoch


# ============================================================
# Full training pipeline
# ============================================================

def train_model(
    perception_epochs=10000,
    production_epochs=10000,
    lr=1e-2,
):
    """
    Train full shared-state model.
    """

    rnn = RNN(
        input_size=len(erb_bins),
        hidden_size=100,
        output_size=len(main_language.word_labels),
    )

    print("\n==============================")
    print("TRAINING PERCEPTION")
    print("==============================\n")

    rnn, perception_loss, perception_epoch = train_perception(
        rnn,
        epochs=perception_epochs,
        lr=lr,
    )

    print("\n==============================")
    print("TRAINING PRODUCTION")
    print("==============================\n")

    rnn, production_loss, production_epoch = train_production(
        rnn,
        epochs=production_epochs,
        lr=lr,
        freeze_perception=True,
    )

    return (
        rnn,
        perception_loss,
        perception_epoch,
        production_loss,
        production_epoch,
    )


# ============================================================
# Evaluation utilities
# ============================================================

def print_production_comparison(
    productions,
    targets,
):
    """
    Pretty-print production outputs.
    """

    for t in range(len(productions)):

        pred = productions[t].flatten()

        targ = targets[t].flatten()

        mse = np.mean((pred - targ) ** 2)

        print(f"Timestep {t}")
        print(f"MSE: {mse:.6f}")

        print(
            "Predicted:",
            np.array2string(
                pred,
                precision=3,
                suppress_small=True,
            )
        )

        print(
            "Target:   ",
            np.array2string(
                targ,
                precision=3,
                suppress_small=True,
            )
        )

        print()


def test_single_random_word(
    rnn,
    top_k=3,
):
    """
    Evaluate perception + production on a random word.
    """

    sequence, word = main_language.random_utterance(length=1)

    inputs = utterance_to_input(sequence)

    h0 = np.zeros((rnn.hidden_size, 1))

    xs, hs, ys, probs = rnn.forward(inputs, h0)

    print("\n==============================")
    print("PERCEPTION TEST")
    print("==============================\n")

    print(f"Word: {word.phonseq}")
    print(f"Utterance: {sequence}\n")

    for t in range(len(inputs)):

        prefix = word.phonseq[: t + 1]

        prob = probs[t].flatten()

        top_indices = np.argsort(prob)[::-1][:top_k]

        print(f"Timestep {t} | prefix='{prefix}'")

        for i in top_indices:

            print(
                f"  {main_language.word_labels[i]}: "
                f"{prob[i]:.4f}"
            )

        print()

    # ========================================================
    # Free-running autoregressive production
    # ========================================================

    print("\n==============================")
    print("AUTOREGRESSIVE PRODUCTION")
    print("==============================\n")

    final_h = hs[len(inputs) - 1]

    productions, _ = rnn.generate_production_sequence(
        final_h,
        len(inputs),
        teacher_forcing_inputs=None,
        generation_noise=0.01,
    )

    print_production_comparison(
        productions,
        inputs,
    )

    return (
        sequence,
        word,
        probs,
        productions,
        inputs,
    )


def test_production_for_word(
    rnn,
    word_label=None,
):
    """
    Evaluate autoregressive production for one word.
    """

    if word_label is None:

        sequence, word = main_language.random_utterance(length=1)

    else:

        try:

            word = next(
                w
                for w in main_language.words
                if w.phonseq == word_label
            )

        except StopIteration:

            raise ValueError(
                f"Unknown word label: {word_label}"
            )

        sequence = word.perfect(length=1)

    inputs = utterance_to_input(sequence)

    h0 = np.zeros((rnn.hidden_size, 1))

    _, hs, _, _ = rnn.forward(inputs, h0)

    final_h = hs[len(inputs) - 1]

    productions, _ = rnn.generate_production_sequence(
        final_h,
        len(inputs),
        teacher_forcing_inputs=None,
        generation_noise=0.01,
    )

    print("\n==============================")
    print(f"PRODUCTION TEST: {word.phonseq}")
    print("==============================\n")

    print_production_comparison(
        productions,
        inputs,
    )

    return word, productions, inputs


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    (
        best_rnn,
        best_loss,
        best_epoch,
        best_prod_loss,
        best_prod_epoch,
    ) = train_model(production_epochs=15000)

    test_single_random_word(
        best_rnn,
        top_k=12,
    )

    test_production_for_word(
        best_rnn,
        word_label="pupupu",
    )