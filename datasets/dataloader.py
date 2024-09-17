
import pickle
from pathlib import Path
import hashlib

import jax.numpy as jnp

from .tokenizers import WordTokenizer


CACHE_DIR = Path(__file__).parent / "cached_datasets"
CACHE_DIR.mkdir(exist_ok=True)


class DataLoader:
    """
    Given a list of strings, create a dataloader that returns minibatches as
    integer token tensors of shape (batchsize, seq_len) using a word level
    tokenizer whose vocabulary is the most frequent words.

    NOTE: this class stores all tokenized strings in RAM as one large tensor,
    minibatches are slices of the large tensor. Hence this is not a good class
    for reaally big datasets!
    """
    def __init__(
        self,
        dataset: list[str],
        batchsize:int=128,
        seq_len: int=150,
        vocab_size:int=10000,
    ):
        self.tokenizer, self._data_tensor = self._initialize_dataset(
            dataset=dataset,
            vocab_size=vocab_size,
        )

        # call the property setters
        self._batchsize = batchsize
        self._seq_len = seq_len
        self._data_tensor_padded = None
        self._make_padded_tensor()

    @staticmethod
    def _initialize_dataset(dataset: list[str], vocab_size:int=10000):
        """
        Build the vocabulary and tokenizer. This function uses a cache to avoid
        rebuilding the same vocabularies + tokenizers.
        """
        key = "_".join(dataset) + str(vocab_size)
        key = hashlib.sha256(key.encode('utf-8')).hexdigest()

        filename = CACHE_DIR / key

        if not filename.exists():
            tokenizer = WordTokenizer(dataset=dataset, vocab_size=vocab_size)

            token_seqs = tokenizer(dataset)

            seq_len = max([len(s) for s in token_seqs])
            pad_seq = [tokenizer.pad_token] * seq_len

            token_seqs = [token_seq + pad_seq for token_seq in token_seqs]
            token_seqs = [token_seq[:seq_len] for token_seq in token_seqs]

            data_tensor = jnp.asarray(token_seqs)

            with open(filename, "wb") as f:
                pickle.dump([tokenizer, data_tensor], f)

        with open(filename, "rb") as f:
            tokenizer, data_tensor = pickle.load(f)

        return tokenizer, data_tensor

    @property
    def batchsize(self) -> int:
        return self._batchsize

    @batchsize.setter
    def batchsize(self, batchsize:int):
        """ Update batch size and the padding at the end of the last batch """
        self._batchsize = batchsize
        self._make_padded_tensor()

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @seq_len.setter
    def seq_len(self, seq_len:int):
        self._seq_len = seq_len
        self._make_padded_tensor()

    def _make_padded_tensor(self):
        """
        Given the batch size and sequence length, make the tensor for the full
        dataset that pads short sequences up to seq_len and pads the final batch
        up to full batch size.
        """
        # pad the final batch with sequencs of pad_tokens
        pad_v = self.batchsize - self._data_tensor.shape[0] % self.batchsize
        padding = jnp.ones(shape=(pad_v, self._data_tensor.shape[1]))
        padding = self.tokenizer.pad_token * padding
        data_tensor_padded = jnp.vstack([self._data_tensor, padding])

        pad_h = max(self.seq_len - data_tensor_padded.shape[1], 0)
        padding = jnp.ones(shape=(data_tensor_padded.shape[0], pad_h))
        padding = self.tokenizer.pad_token * padding
        data_tensor_padded = jnp.hstack([data_tensor_padded, padding])

        data_tensor_padded = data_tensor_padded[:, :self.seq_len]

        assert data_tensor_padded.shape[0] % self.batchsize == 0
        assert data_tensor_padded.shape[1] == self.seq_len

        self._data_tensor_padded = data_tensor_padded

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= len(self):
            raise StopIteration

        result = self[self._iter_index]
        self._iter_index += 1
        return result

    def __getitem__(self, index:int) -> jnp.ndarray:
        index = index * self.batchsize
        return self._data_tensor_padded[index:(index + self.batchsize), :]

    def __len__(self) -> int:
        return self._data_tensor_padded.shape[0] // self.batchsize
