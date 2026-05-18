"""Training and evaluation utilities for the RNN word recognizer."""
import matplotlib.pyplot as plt
import numpy as np
from network import RNN
from seq2seq import Seq2Seq
from languages import main_language, vowels, consonants

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


def canonical_utterance_to_input(word, length=1, erb_bins=erb_bins):
    """Turn a word into the canonical (mean) acoustic spectrum per frame."""
    sequence = []
    phonseq = word.phonseq
    for i in range(len(phonseq)):
        phon = phonseq[i]
        if phon in consonants:
            next_vowel = phonseq[i + 1]
            f1 = consonants[phon].bursts[next_vowel]
            sequence += [(f1, 0)] * length
        else:
            sequence += [(vowels[phon].F1, vowels[phon].F2)] * length
    return utterance_to_input(sequence, erb_bins)


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


def test_single_word(rnn, top_k=None, label="rand"):
    """Evaluate the model on a single (random) word and print timestep probabilities.

    Args:
        rnn: trained RNN instance.
        top_k: if provided, only print the top-k predicted words at each timestep.
        label: if provided, function uses the existing word with the label/phonseq provided.

    Returns:
        sequence: the input utterance as a list of formant tuples.
        word: the Word object selected for testing.
        probs: dictionary of softmax probabilities at each timestep.
    """
    if label == "rand":
        sequence, word = main_language.random_utterance(length=1)
    else:
        wrd_idx = main_language.word_labels.index(label)
        word = main_language.words[wrd_idx]
        sequence = word.utterance()

    inputs = utterance_to_input(sequence)
    probs = rnn.predict(inputs)
    

    print(f"Testing random word: {label}")
    print(f"Utterance sequence: {sequence}")
    print()

    for t in range(len(inputs)):
        prob = probs[t].flatten()
        prefix = label[: t + 1]
        print(f"Timestep {t}, prefix='{prefix}':")
        if top_k is None:
            for label2, p in zip(main_language.word_labels, prob):
                print(f"  {label2}: {p:.4f}")
        else:
            top_indices = np.argsort(prob)[::-1][:top_k]
            for i in top_indices:
                print(f"  {main_language.word_labels[i]}: {prob[i]:.4f}")
        print()

    return sequence, word, probs


def train_seq2seq_perception(input_size=None, hidden_size=50, epochs=5000, lr=1e-2, language=main_language):
    """Train a seq2seq model for perception (utterance -> word label).
    
    Uses prefix_target_distribution to teach the model to narrow down word
    possibilities as more of the utterance is revealed.
    
    Args:
        input_size: spectral dimension (defaults to len(erb_bins), usually 30).
        hidden_size: dimension of hidden state.
        epochs: number of training iterations.
        lr: learning rate.
        language: language object with vocabulary.
    
    Returns:
        model: trained Seq2Seq instance.
        best_loss: lowest loss observed.
        best_epoch: epoch where best loss occurred.
    """
    if input_size is None:
        input_size = len(erb_bins)
    
    output_size = len(language.word_labels)
    model = Seq2Seq(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    
    best_loss = np.inf
    best_params = model.get_params()
    best_epoch = -1
    
    for epoch in range(epochs):
        sequence, word = language.random_utterance(length=1)
        inputs = utterance_to_input(sequence)
        
        # Full forward/backward through the model to train all weights
        h0 = np.zeros((hidden_size, 1))
        xs, hs, ys, probs = model.forward(inputs, h0)
        targets = [prefix_target_distribution(word, t, language) for t in range(len(inputs))]
        
        total_loss = 0
        for t in range(len(inputs)):
            total_loss += -np.sum(targets[t] * np.log(probs[t] + 1e-9))
        
        model.backward(xs, hs, probs, targets, lr=lr)
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_params = model.get_params()
            best_epoch = epoch
        
        if epoch % 500 == 0:
            print(f"  Perception train epoch {epoch}: loss {total_loss:.4f}")
    
    model.set_params(best_params)
    return model, best_loss, best_epoch


def train_seq2seq_production(input_size=None, hidden_size=50, max_frames=6, epochs=5000, lr=1e-2, language=main_language):
    """Train a seq2seq model for production (word label + position -> frame).
    
    Trains using an RNN sequence of word+position inputs, allowing the model to
    learn frame dependencies across the word.
    
    Args:
        input_size: spectral dimension (defaults to len(erb_bins), usually 30).
        hidden_size: dimension of hidden state.
        max_frames: maximum number of frames per word in training.
        epochs: number of training iterations.
        lr: learning rate.
        language: language object with vocabulary.
    
    Returns:
        model: trained Seq2Seq instance.
        best_loss: lowest loss observed.
        best_epoch: epoch where best loss occurred.
    """
    if input_size is None:
        input_size = len(erb_bins)
    
    vocab_size = len(language.word_labels)
    model_input_size = vocab_size + max_frames
    model = Seq2Seq(input_size=model_input_size, hidden_size=hidden_size, output_size=input_size)
    
    best_loss = np.inf
    best_params = model.get_params()
    best_epoch = -1
    
    for epoch in range(epochs):
        _, word = language.random_utterance(length=1)
        targets = canonical_utterance_to_input(word, length=1)
        if len(targets) > max_frames:
            targets = targets[:max_frames]
        num_frames = len(targets)
        
        word_vec = onehot(word.phonseq, language)
        
        total_loss = 0
        for frame_idx in range(num_frames):
            position_vec = np.zeros((max_frames, 1))
            position_vec[frame_idx] = 1.0
            x = np.vstack([word_vec, position_vec])
            
            xs, hs, ys, probs = model.forward([x], np.zeros((hidden_size, 1)))
            total_loss += -np.sum(targets[frame_idx] * np.log(probs[0] + 1e-9))
            model.backward(xs, hs, probs, [targets[frame_idx]], lr=lr)
        
        avg_loss = total_loss / num_frames if num_frames > 0 else 0
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = model.get_params()
            best_epoch = epoch
        
        if epoch % 500 == 0:
            print(f"  Production train epoch {epoch}: avg frame cross-entropy {avg_loss:.4f}")
    
    model.set_params(best_params)
    return model, best_loss, best_epoch


def test_perception(model, input_size=None, label=None, top_k=3, language=main_language):
    """Test perception: progressively encode utterance and predict word at each timestep.
    
    Shows how the model narrows down word possibilities as more of the utterance
    is revealed, demonstrating the effect of prefix_target_distribution training.
    
    Args:
        model: trained Seq2Seq perception model.
        input_size: spectral dimension (defaults to len(erb_bins)).
        label: word label to test (None for random).
        top_k: show top-k predictions.
        language: language object.
    
    Returns:
        (sequence, word, final_prediction, final_confidence)
    """
    if input_size is None:
        input_size = len(erb_bins)
    
    if label is None:
        sequence, word = language.random_utterance(length=1)
        label = word.phonseq
    else:
        word_idx = language.word_labels.index(label)
        word = language.words[word_idx]
        sequence = word.utterance()
    
    inputs = utterance_to_input(sequence)
    hidden_size = model.hidden_layer.W_hh.shape[0]
    
    print(f"\n{'='*60}")
    print(f"PERCEPTION TEST: {label}")
    print(f"{'='*60}")
    print(f"Utterance frames: {len(inputs)}")
    print(f"\nTimestep-by-timestep predictions (top {top_k}):\n")
    
    # Progressive encoding
    h0 = np.zeros((hidden_size, 1))
    h_state = h0
    final_probs = None
    final_pred = None
    
    for t in range(len(inputs)):
        # Accumulate context frame-by-frame
        x_proj = model.input_layer.forward(inputs[t])
        h_state = model.hidden_layer.forward(x_proj, h_state)
        
        # Predict from accumulated context
        y = model.output_layer.forward(h_state)
        probs = model.softmax(y).flatten()
        final_probs = probs
        
        # Get prefix from true label
        true_prefix = label[:t+1]
        
        print(f"Timestep {t} (prefix='{true_prefix}'):")
        top_indices = np.argsort(probs)[::-1][:top_k]
        for i, idx in enumerate(top_indices, 1):
            marker = "→" if language.word_labels[idx] == label else " "
            print(f"  {i}. {marker} {language.word_labels[idx]}: {probs[idx]:.4f}")
        print()
    
    # Final prediction
    final_pred = language.word_labels[np.argmax(final_probs)]
    final_confidence = final_probs[np.argmax(final_probs)]
    
    print(f"Result: {'✓ CORRECT' if final_pred == label else '✗ INCORRECT'}")
    print(f"Final prediction: {final_pred} (confidence: {final_confidence:.4f})")
    
    return sequence, word, final_pred, final_confidence


def test_production(model, input_size=None, hidden_size=50, max_frames=6, label=None, language=main_language):
    """Test production: generate utterance frames from word label + position.
    
    Generates frames sequentially, where each frame is conditioned on both the
    word label and the frame position. This allows the model to generate
    context-dependent consonant frames.
    
    Args:
        model: trained Seq2Seq production model.
        input_size: spectral dimension (defaults to len(erb_bins)).
        hidden_size: dimension of hidden state.
        max_frames: maximum frames (must match training).
        label: word label to produce (None for random).
        language: language object.
    
    Returns:
        (word, generated_frames, ground_truth_frames)
    """
    if input_size is None:
        input_size = len(erb_bins)
    
    if label is None:
        _, word = language.random_utterance(length=1)
        label = word.phonseq
    else:
        word_idx = language.word_labels.index(label)
        word = language.words[word_idx]
    
    # One-hot word label
    word_vec = onehot(label, language)
    
    generated_frames = []
    h = np.zeros((hidden_size, 1))
    
    # Use the full canonical target length (or cap to max_frames)
    target_inputs = canonical_utterance_to_input(word, length=1)
    num_frames = min(len(target_inputs), max_frames)
    
    # Generate frame-by-frame using independent position-conditioned prediction
    for frame_idx in range(num_frames):
        position_vec = np.zeros((max_frames, 1))
        position_vec[frame_idx] = 1.0
        
        combined_input = np.vstack([word_vec, position_vec])
        xs, hs, ys, probs = model.forward([combined_input], np.zeros((hidden_size, 1)))
        generated_frames.append(probs[0].flatten())
    
    generated_frames = np.array(generated_frames)
    
    # Ground truth
    truth_inputs = target_inputs
    truth_frames = np.vstack([f.flatten() for f in truth_inputs[:num_frames]])

    print(f"\n{'='*60}")
    print(f"PRODUCTION TEST: {label}")
    print(f"{'='*60}")
    print(f"Ground truth canonical frames: {num_frames}")
    print(f"Generated frames: {generated_frames.shape[0]}")
    print(f"\nGenerated frame peaks:")
    for i in range(len(generated_frames)):
        gen_peak = np.argmax(generated_frames[i])
        gen_erb = erb_bins[gen_peak] if gen_peak < len(erb_bins) else 0
        print(f"  Frame {i}: peak at ERB {gen_erb:.2f}")
    
    print(f"\nGround truth frame peaks:")
    for i in range(len(truth_frames)):
        truth_peak = np.argmax(truth_frames[i])
        truth_erb = erb_bins[truth_peak] if truth_peak < len(erb_bins) else 0
        print(f"  Frame {i}: peak at ERB {truth_erb:.2f}")
    
    return word, generated_frames, truth_frames


def run_seq2seq_full_suite(hidden_size=50, input_size=None, max_frames=6, train_epochs=5000, test_words=None):
    """Complete test suite: train perception and production models, then test both.
    
    Args:
        hidden_size: dimension of hidden state (default 50).
        input_size: spectral dimension (defaults to 30).
        max_frames: max frames per word for production.
        train_epochs: number of training epochs.
        test_words: list of word labels to test (default: sample from vocabulary).
    """
    if input_size is None:
        input_size = len(erb_bins)
    
    if test_words is None:
        test_words = ["pitaku", "katupa", "pupupu", "tutapa"]
    
    print("\n" + "="*60)
    print("SEQ2SEQ FULL SUITE")
    print("="*60)
    print(f"Config: input_size={input_size}, hidden_size={hidden_size}, max_frames={max_frames}")
    
    # Train perception
    print("\n[1/4] Training PERCEPTION model (utterance -> word)...")
    perc_model, perc_loss, perc_epoch = train_seq2seq_perception(
        input_size=input_size, hidden_size=hidden_size, epochs=train_epochs
    )
    print(f"✓ Perception trained. Best loss: {perc_loss:.4f} at epoch {perc_epoch}")
    
    # Train production
    print("\n[2/4] Training PRODUCTION model (word -> utterance)...")
    prod_model, prod_loss, prod_epoch = train_seq2seq_production(
        input_size=input_size, hidden_size=hidden_size, max_frames=max_frames, epochs=train_epochs
    )
    print(f"✓ Production trained. Best loss: {prod_loss:.4f} at epoch {prod_epoch}")
    
    # Test perception
    print("\n[3/4] Testing PERCEPTION...")
    print("-" * 60)
    perc_correct = 0
    for label in test_words:
        _, _, pred, _ = test_perception(perc_model, input_size=input_size, label=label)
        if pred == label:
            perc_correct += 1
    print(f"\nPerception accuracy: {perc_correct}/{len(test_words)}")
    
    # Test production
    print("\n[4/4] Testing PRODUCTION...")
    print("-" * 60)
    for label in test_words:
        test_production(prod_model, input_size=input_size, hidden_size=hidden_size, 
                       max_frames=max_frames, label=label)
    
    print("\n" + "="*60)
    print("SUITE COMPLETE")
    print("="*60)
    
    return perc_model, prod_model


if __name__ == "__main__":
    # Run the full seq2seq test suite
    # Uncomment to run with custom parameters:
    perc_model, prod_model = run_seq2seq_full_suite(hidden_size=50, train_epochs=20000)
    
    # Original RNN tests (commented out)
    # best_rnn, best_loss, best_epoch = train_model(epochs=20000)
    # test_single_word(best_rnn, label="pitaku")
    # test_single_word(best_rnn, label="katupa")
    # test_single_word(best_rnn, label="pupupu")
    # test_single_word(best_rnn, label="tutapa")

    # test_vowel = vowels['a'].utter(1)
    # input = utterance_to_input(test_vowel)
    # plt.plot(erb_bins, np.squeeze(input))
    # plt.xlabel('ERB-step on basilar membrane')
    # plt.ylabel('activation')
    # plt.show()