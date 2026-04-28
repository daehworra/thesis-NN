"""Define the artificial language and utterance generation utilities."""

from scipy.stats import bernoulli, norm
import random


class Vowel:
    """Generate vowel formant values with Gaussian noise."""

    def __init__(self, F1, F2):
        self.F1 = F1
        self.F2 = F2

    def utter(self, length=1):
        """Generate a vowel utterance sequence of given length.

        Returns:
            A list of (F1, F2) tuples for each timestep.
        """
        rand_F1 = norm.rvs(loc=self.F1, scale=1, size=length)
        rand_F2 = norm.rvs(loc=self.F2, scale=1, size=length)

        return [(rand_F1[i], rand_F2[i]) for i in range(length)]


class Consonant:
    """Generate consonant formant bursts at vowel-dependent locations."""

    def __init__(self, a_burst, i_burst, u_burst):
        self.bursts = {
            "a": a_burst,
            "i": i_burst,
            "u": u_burst
        }

    def utter(self, vowel, length=1):
        """Generate a consonant utterance conditioned on a following vowel."""
        F1_list = norm.rvs(loc=self.bursts[vowel], scale=1.5, size=length)
        return [(F1, 0) for F1 in F1_list]


# Tone and vowel classes used by the artificial language.
vowels = {
    'a': Vowel(13, 19),
    'e': Vowel(10, 22),
    'i': Vowel(7, 25),
    'o': Vowel(10, 16),
    'u': Vowel(7, 13)
}

consonants = {
    'p': Consonant(9, 18, 14),
    't': Consonant(27, 29, 25),
    'k': Consonant(19, 23, 21)
}

class Word:
    """A word represented by a phoneme sequence and its generated utterance."""

    def __init__(self, phonseq):
        self.phonseq = phonseq

    def utterance(self, length=1):
        """Generate a noisy utterance sequence for this word.

        Args:
            length: number of frames per phoneme.

        Returns:
            A list of (F1, F2) tuples forming the full utterance.
        """
        sequence = []
        phonseq = self.phonseq

        for i in range(len(phonseq)):
            phon = phonseq[i]

            if phon in consonants:
                sequence += consonants[phon].utter(phonseq[i + 1], length)
            else:
                sequence += vowels[phon].utter(length)

        return sequence

# Cohort 1
pitaku = Word('pitaku')
pipiti = Word('pipiti')
pikita = Word('pikita')
pituka = Word('pituka')

# Cohort 2
katuka = Word('katuka')
katuti = Word('katuti')
katupu = Word('katupu')
katupa = Word('katupa')

# Unique
tutapa = Word('tutapa')
pupupu = Word('pupupu')
kipiti = Word('kipiti')
tapika = Word('tapika')

class Language:
    """Container for vocabulary words and random utterance sampling."""

    def __init__(self, words):
        self.words = words
        self.word_labels = [word.phonseq for word in self.words]

    def add_word(self, word):
        """Add a new word to the language vocabulary."""
        self.words.append(word)
        self.word_labels.append(word.phonseq)

    def random_utterance(self, length=1):
        """Sample a random vocabulary word and produce its utterance."""
        word = random.choice(self.words)
        return word.utterance(length), word


    
main_language = Language([pitaku, pituka, pikita, pipiti, katuka, katuti, katupu, katupa, tutapa, pupupu, kipiti, tapika])
