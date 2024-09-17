import re
import string
from collections import Counter


class WordTokenizer:
    def __init__(
        self,
        dataset:list[str],
        vocab_size:int=10000,
    ):
        """
        Take a list of strings, and make a word tokenizer using the most common
        words. The strings are split by the space character.

        Parameters
        ----------
        dataset: list[str]
            a lsit of strings, eadh string is one sentence ins the text dataset.
        vocab_size : int
            the top most frequent words will be chosen
        """

        self.vocab_size = vocab_size
        dataset = [self.clean_str(s) for s  in dataset]

        # vocab = PAD + UNK + most_frequent_words
        self.vocab = self.build_vocab(dataset=dataset)[:vocab_size - 2]

        # padding, unknown and end of sentence tokens
        self.vocab.insert(0, "UNK")
        self.vocab.insert(1, "EOS")
        self.vocab.append("PAD")  # padding is last token

        self.word_to_idx = {w:i for i, w in enumerate(self.vocab)}

        self.unk_token = self.word_to_idx["UNK"]
        self.eos_token = self.word_to_idx["EOS"]
        self.pad_token = self.word_to_idx["PAD"]

    def __call__(self, x: list[str]) -> list[list[int]]:
        result = list(
            map(
                lambda x_i:[
                    self.word_to_idx.get(w, self.unk_token)
                    for w in x_i.split(" ")
                ] + [self.eos_token],
                x
            )
        )
        return result

    def decode(self, x_idx: list[int]):
        return " ".join([self.vocab[i] for i in x_idx])

    @staticmethod
    def build_vocab(dataset: list[str]) -> list[str]:
        """ Get the most common words in the list of strings """
        word_freq = {}
        for x in dataset:
            for word, freq in Counter(x.split(" ")).items():
                word_freq[word] = word_freq.get(word, 0) + freq
        vocab = sorted(word_freq, key=lambda w: -word_freq[w])
        return vocab

    @staticmethod
    def clean_str(text:str) -> string:
        """ Insert spaces around punctuation so it counts as word """
        text = re.sub(f"([{string.punctuation}])", r" \1 ", text)
        text = re.sub(" +", " ", text) # remove multiple spaces
        return text.lower()
