"""Training and evaluation utilities for the shared-state RNN."""

import numpy as np
import math
from network import RNN
from languages import main_language
import matplotlib.pyplot as plt


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

        sequence, word = main_language.random_utterance(length=LENGTH)

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

        sequence, word = main_language.random_utterance(length=LENGTH)

        inputs = utterance_to_input(sequence)

        h0 = np.zeros((rnn.hidden_size, 1))

        xs, hs, ys, probs = rnn.forward(inputs, h0)

        final_h = hs[len(inputs) - 1]

        # Gradually reduce teacher forcing
        teacher_forcing_ratio =  1.0 - (epoch / epochs)

        productions, _ = rnn.generate_production_sequence(
            final_h,
            len(inputs),
            teacher_forcing_inputs=inputs,
            teacher_forcing_ratio=teacher_forcing_ratio,
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


def test_single_word(
    rnn,
    top_k=3,
    phonseq=None,
):
    """
    Evaluate perception + production on a random word.
    """
    if phonseq is None:
        sequence, word = main_language.random_utterance(length=LENGTH)
    else:
        word_idx= main_language.word_labels.index(phonseq)
        word = main_language.words[word_idx]
        sequence = word.utterance(length=LENGTH)

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
    # Free-running production
    # ========================================================

    print("\n==============================")
    print("PRODUCTION")
    print("==============================\n")

    final_h = hs[len(inputs) - 1]

    productions, _ = rnn.generate_production_sequence(
        final_h,
        len(inputs),
        teacher_forcing_inputs=None,
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
        sequence, word = main_language.random_utterance(LENGTH)
    else:
        word_idx = main_language.word_labels.index(word_label)
        word = main_language.words[word_idx]
        sequence = word.utterance(LENGTH)

    perfect_sequence = word.perfect(LENGTH)
    perfect_inputs =utterance_to_input(perfect_sequence)
    inputs = utterance_to_input(sequence)

    h0 = np.zeros((rnn.hidden_size, 1))

    _, hs, _, _ = rnn.forward(inputs, h0)

    final_h = hs[len(inputs) - 1]

    productions, _ = rnn.generate_production_sequence(
        final_h,
        len(inputs),
        teacher_forcing_inputs=None,
    )

    print("\n==============================")
    print(f"PRODUCTION TEST: {word.phonseq}")
    print("==============================\n")

    print_production_comparison(
        productions,
        perfect_inputs,
    )

    return word, productions, inputs, perfect_inputs

def test_all_word_perc(rnn):
    """Test perception on every word in the vocabulary.

    Returns a list of 0/1 values indicating whether the network's
    top prediction at the final timestep matches the true word label.
    """

    results = []

    for word in main_language.words:

        sequence = word.utterance(LENGTH)

        inputs = utterance_to_input(sequence)

        h0 = np.zeros((rnn.hidden_size, 1))

        xs, hs, ys, probs = rnn.forward(inputs, h0)

        # Use final timestep prediction for correctness
        final_prob = probs[len(probs) - 1].flatten()

        pred_idx = int(np.argmax(final_prob))

        true_idx = main_language.word_labels.index(word.phonseq)

        results.append(1 if pred_idx == true_idx else 0)

    return results

def test_all_word_prod(rnn):
    """Test autoregressive production for every word in the vocabulary.

    Returns the summed per-word mean-squared error across the full
    production sequence for each word, as compared to the ideal utterance.
    """

    total_mse = 0.0

    for word in main_language.words:

        sequence = word.utterance(LENGTH)
        perfect_sequence = word.perfect(LENGTH)

        inputs = utterance_to_input(sequence)
        perfect_inputs = utterance_to_input(perfect_sequence)

        h0 = np.zeros((rnn.hidden_size, 1))

        _, hs, _, _ = rnn.forward(inputs, h0)

        final_h = hs[len(inputs) - 1]

        productions, _ = rnn.generate_production_sequence(
            final_h,
            len(inputs),
            teacher_forcing_inputs=None,
        )

        word_mse = np.mean([
            np.mean((productions[t] - perfect_inputs[t]) ** 2)
            for t in range(len(inputs))
        ])

        total_mse += word_mse

    return total_mse


def epoch_testing_perc(runs=10, epochs=[1000, 3000, 5000, 10000, 15000, 20000, 30000]):
    """Run multiple training runs for each epoch value and plot results.

    For each value in `epochs`, this trains `runs` independent models,
    evaluates perception accuracy across the vocabulary using
    `test_all_word_perc`, and then plots mean accuracy with error bars.

    Returns a dict mapping epoch -> list of accuracies (floats in [0,1]).
    """

    all_results = {}

    means = []
    stds = []

    epoch_values = list(epochs)

    for epoch in epoch_values:

        accuracies = []

        for i in range(runs):

            (
                best_rnn,
                best_loss,
                best_epoch,
                best_prod_loss,
                best_prod_epoch,
            ) = train_model(perception_epochs=epoch, production_epochs=0)

            res = test_all_word_perc(best_rnn)

            # accuracy = fraction of words classified correctly
            acc = float(np.mean(res)) if res else 0.0

            accuracies.append(acc)

            print(f"[EPOCH TEST] epochs={epoch} run={i+1}/{runs} acc={acc:.3f}")

        all_results[epoch] = accuracies

        means.append(float(np.mean(accuracies)))

        stds.append(float(np.std(accuracies)))

    # Plot mean accuracy with std dev error bars
    plt.figure()

    plt.errorbar(epoch_values, means, yerr=stds, marker="o", capsize=5)

    plt.xlabel("Training epochs")

    plt.ylabel("Perception accuracy")

    plt.title("Perception accuracy vs training epochs")

    plt.grid(True)

    plt.tight_layout()

    outpath = "perception_epoch_comparison.png"

    plt.savefig(outpath)

    print(f"Saved epoch comparison plot to {outpath}")

    try:
        plt.show()
    except Exception:
        # In headless environments, showing may fail; ignore.
        pass

    return all_results


def epoch_testing_prod(runs=10, epochs=[1000, 3000, 5000, 10000, 15000, 20000, 30000]):
    """Run multiple production training runs for each production epoch value.

    Perception is fixed at 10000 epochs, while production is trained for the
    varying values in `epochs`. The function evaluates production loss on the
    full vocabulary using `test_all_word_prod` and plots mean loss vs epochs.
    """

    all_results = {}

    means = []
    stds = []

    epoch_values = list(epochs)

    for epoch in epoch_values:

        losses = []

        for i in range(runs):

            (
                best_rnn,
                best_loss,
                best_epoch,
                best_prod_loss,
                best_prod_epoch,
            ) = train_model(perception_epochs=10000, production_epochs=epoch)

            loss = test_all_word_prod(best_rnn)

            losses.append(float(loss))

            print(f"[EPOCH TEST PROD] prod_epochs={epoch} run={i+1}/{runs} loss={loss:.6f}")

        all_results[epoch] = losses

        means.append(float(np.mean(losses)))

        stds.append(float(np.std(losses)))

    plt.figure()

    plt.errorbar(epoch_values, means, yerr=stds, marker="o", capsize=5)

    plt.xlabel("Production training epochs")

    plt.ylabel("Production loss (summed MSE)")

    plt.title("Production loss vs production training epochs")

    plt.grid(True)

    plt.tight_layout()

    outpath = "production_epoch_comparison.png"

    plt.savefig(outpath)

    print(f"Saved production epoch comparison plot to {outpath}")

    try:
        plt.show()
    except Exception:
        pass

    return all_results

def plot_vector_comparison(production, perfect_input,
                           production_label="Produced",
                           perfect_label="Ideal"):
    """
    Plot corresponding vectors from production and perfect_input.

    Parameters
    ----------
    production : list of array-like
        List of produced vectors.
    perfect_input : list of array-like
        List of ideal vectors.
    production_label : str
        Label for produced vectors.
    perfect_label : str
        Label for ideal vectors.
    """

    if len(production) != len(perfect_input):
        raise ValueError(
            f"Length mismatch: {len(production)} produced vectors "
            f"and {len(perfect_input)} ideal vectors."
        )

    n_vectors = len(production)

    # Create a roughly square grid
    ncols = math.ceil(math.sqrt(n_vectors))
    nrows = math.ceil(n_vectors / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    for i, (prod_vec, ideal_vec) in enumerate(zip(production, perfect_input)):
        prod_vec = np.asarray(prod_vec)
        ideal_vec = np.asarray(ideal_vec)

        if len(prod_vec) != len(ideal_vec):
            raise ValueError(
                f"Vector {i} has mismatched dimensions "
                f"({len(prod_vec)} vs {len(ideal_vec)})."
            )

        x = np.arange(len(prod_vec))

        axes[i].plot(
            x,
            ideal_vec,
            label=perfect_label,
            linewidth=2
        )

        axes[i].plot(
            x,
            prod_vec,
            '--',
            label=production_label,
            linewidth=2
        )

        axes[i].set_title(f"Utterance {i+1}")
        axes[i].set_xlabel("Feature")
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    # Hide unused axes
    for ax in axes[n_vectors:]:
        ax.set_visible(False)

    fig.suptitle("Produced vs Ideal Vectors", fontsize=16)
    fig.tight_layout()

    return fig, axes

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    LENGTH=1

    # epoch_testing_perc()

    #epoch_testing_prod()

    (
        best_rnn,
        best_loss,
        best_epoch,
        best_prod_loss,
        best_prod_epoch,
    ) = train_model(perception_epochs=10000, production_epochs=30000)

    # test_single_word(
    #     best_rnn,
    #     top_k=12,
    #     phonseq='pipiti'
    # )


    _, production, _, perfect_input = test_production_for_word(
        best_rnn,
        word_label="katupa",
    )

    plot_vector_comparison(production, perfect_input)
    plt.show()