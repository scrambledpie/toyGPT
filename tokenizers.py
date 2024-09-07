from collections import Counter

class WordTokenizer:
    def __init__(self, dataset:list[str], vocab_size:int=10000):

        self.vocab_size = vocab_size
        self.vocab = self.build_vocab(dataset=dataset)[:vocab_size]

        # padding, unknown and end of sentence tokens
        self.vocab.insert(0, "PAD")
        self.vocab.insert(1, "UNK")
        self.vocab.insert(2, "EOS")
        self.word_to_idx = {w:i for i, w in enumerate(self.vocab)}

        self.pad_token = self.word_to_idx["PAD"]
        self.unk_token = self.word_to_idx["UNK"]
        self.eos_token = self.word_to_idx["EOS"]

        self.final_tokens = [self.eos_token] + [self.pad_token] * 200

    def __call__(self, x: list[str]) -> list[list[int]]:
        x_idx = []
        for xi in x:
            x_idx.append(
                [self.word_to_idx.get(w, 1) for w in xi.split(" ")]
            )
        return x_idx

    def decode(self, x_idx: list[int]):
        return " ".join([self.vocab[i] for i in x_idx])

    @staticmethod
    def build_vocab(dataset: list[str]) -> list[str]:
        word_freq = {}
        for x in dataset:
            for word, freq in Counter(x.split(" ")).items():
                word_freq[word] = word_freq.get(word, 0) + freq
        vocab = sorted(word_freq, key=lambda w: -word_freq[w])
        return vocab
