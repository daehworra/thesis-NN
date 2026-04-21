import numpy as np
from network import RNN
from languages import main_language

erb_bins = np.linspace(4, 30, 30) 

def gaussian(x, mu, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def formants_to_spectrum(f1, f2, erb_bins=erb_bins):
    spec_f1 = gaussian(erb_bins, f1)
    spectrum = spec_f1.copy()
    if f2 != 0:
        spec_f2 = gaussian(erb_bins, f2)
        spectrum += spec_f2 

    spectrum /= np.sum(spectrum)
    return spectrum.reshape(-1, 1) 

def utterance_to_input(sequence, erb_bins=erb_bins):
    return [formants_to_spectrum(f1, f2, erb_bins) for (f1, f2) in sequence]

def onehot(label):
    vec = np.zeros((len(main_language.word_labels), 1))
    vec[main_language.word_labels.index(label)] = 1
    return vec
