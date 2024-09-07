import json
import re
from pathlib import Path
import string

from tokenizers import WordTokenizer


DATA_FILE = Path(__file__).parent / "winemag-data-130k-v2.json"

def clean_str(text:str):
    text = re.sub(f"([{string.punctuation}])", r" \1 ", text) # pad punctuation
    text = re.sub(" +", " ", text) # remove multiple spaces
    return text.lower()


def load_wine():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # 130k list[str]
    data = [x['description'] for x in data]
    data = [clean_str(x) for x in data]
    return data



class WineDataLoader:
    def __init__(
        self,
        context:int=200,
        vocab_size:int=10000,
        batchsize:int=128
    ):
        self.context = context
        self.vocab_size = vocab_size
        self.batchsize = batchsize
        self.tokenizer = None
        self.ds_tokens = None

        self.ds = load_wine()
        self.end_token_idx = self.vocab_size + 1
        self.pad_and_tokenize()
        self.make_minibatches()

    def pad_and_tokenize(self) -> None:
        self.tokenizer = WordTokenizer(dataset=self.ds, vocab_size=self.vocab_size)
        ds_tokens = self.tokenizer(self.ds)
        padding = [self.tokenizer.eos_token] + [self.tokenizer.pad_token]* (self.context + 1)
        output = []
        for tokens_i in ds_tokens:
            # add stop tokens and truncate
            tokens_i = tokens_i + padding
            tokens_i = tokens_i[:self.context + 1]
            output.append(tokens_i)

        self.ds_tokens = output
        n = self.batchsize - len(output) % self.batchsize
        self.ds_tokens += self.ds_tokens[:n]

    def make_minibatches(self) -> None:
        self.minibatches = []
        x_batch, y_batch = [], []
        for tokens in self.ds_tokens:
            if len(x_batch) == self.batchsize:
                self.minibatches.append([x_batch, y_batch])
                x_batch, y_batch = [], []

            x_idx, y_idx = tokens[:-1], tokens[1:]

            x_batch.append(x_idx)
            y_batch.append(y_idx)

        # pad last batch if too small
        if len(x_batch) > 0 and len(x_batch) < self.batchsize:
            n = self.batchsize - len(x_batch)
            x_batch += self.minibatches[0][0][:n]
            y_batch += self.minibatches[0][1][:n]
            self.minibatches.append([x_batch, y_batch])

    def __getitem__(self, idx) -> list[list[int]]:
        return self.minibatches[idx]
    
    def __len__(self):
        return len(self.minibatches)


if __name__=="__main__":
    ds = WineDataLoader()


