import pickle
from pathlib import Path
from operator import getitem

import jax
import jax.numpy as jnp

from .transformer_block import transformer_block
from .randmat import Random


class GPTModel:
    def __init__(
        self,
        vocab_size:int,
        context_size:int,
        dtype=jnp.float32,
        num_layers:int=2,
        x_dim:int=256,
        qk_dim:int=128,
        eos_token:int=1,
        pad_token:int=None,
        num_heads:int=1,
    ):
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.x_dim = x_dim
        self.x_dim_head = x_dim // num_heads
        self.qk_dim = qk_dim
        self.qk_dim_head = qk_dim // num_heads
        self.num_heads = num_heads
        self.eos_token = eos_token

        if pad_token is None:
            # Assume the pad token is outside the vocab
            pad_token = vocab_size + 1
        self.pad_token = pad_token

        self._dtype = dtype

        # initialise embeddings token_id -> embedding vector
        embedding = Random.randmat(
            shape=(vocab_size, x_dim),
            dtype=dtype,
            minval=-1,
            maxval=1,
        )

        # initalise position embeddings token_index -> embedding vector
        self.pos_embeddings = self._init_pos_embeddings(
            context_size=context_size,
            x_dim=x_dim,
            dtype=dtype,
        )

        # initialise transformer model weights
        maxval_x = 1 / jnp.sqrt(x_dim)

        # make one big weights matrix for each layer that we unpacked for
        # (Q, K, V, fc1, fc2)
        n_cols = [(qk_dim * 2 + x_dim * 3)] * self.num_layers + [vocab_size]
        self.weights = [embedding]
        for n_col in n_cols:
            weights_i = Random.randmat(
                shape=(x_dim, n_col),
                dtype=dtype,
                minval=-maxval_x,
                maxval=maxval_x
            )
            self.weights.append(weights_i)

        # used for generation, compile just the forewad pass (no loss)
        self.gen_once = jax.jit(self._forward)

    @staticmethod
    def _init_pos_embeddings(context_size:int, x_dim:int, dtype) -> jnp.array:
        """" initialise positional embeddings """

        assert x_dim % 2 == 0, f"x_dim {x_dim} must be even"
        pos_embeddings_list = []
        for i in range(x_dim // 2):
            freq = 10000 ** (-2*i / x_dim)

            # (context_size)
            even_ids = jnp.sin(freq * jnp.arange(0, context_size, dtype=dtype))
            odd_ids = jnp.cos(freq * jnp.arange(0, context_size, dtype=dtype))
            pos_embeddings_list += [even_ids, odd_ids]

        # (context_sixze, x_dim)
        pos_embeddings = jnp.stack(pos_embeddings_list, axis=1)

        # (1, context_size, x_dim)
        return pos_embeddings[None, :, :].astype(dtype)

    def _forward(
        self,
        weights_list: list[jnp.array],
        x_idx:jnp.array,
    ) -> jnp.array:
        """ forward pass through full model """

        assert len(x_idx.shape) == 2, f" x_idx has wierd shape {x_idx.shape}"

        _, seq_len = x_idx.shape
        assert seq_len <= self.pos_embeddings.shape[1], (
            f"out of context! {seq_len} > {self.pos_embeddings.shape[1]}"
        )
        x = self.pos_embeddings[:, :seq_len, :]
        x = x + weights_list[0][x_idx, :]

        # pass through transformer blocks
        for w in weights_list[1:-1]:
            x = transformer_block(weights=w, x_emb=x, num_heads=self.num_heads)

        # pass through final inear layer
        x = jnp.matmul(x, weights_list[-1])

        return x

    def _loss(
        self,
        weights_list: jnp.array,
        xy_idx:jnp.array, # (batch_size, seq_len + 1)
    ) -> jnp.array:

        x_idx, y_idx = xy_idx[:, :-1], xy_idx[:, 1:]

        # (batch_size, seq_len, vocab_size)
        logits = self._forward(weights_list, x_idx)

        # (batch_size * seq_len, vocab_size)
        logits = logits.reshape(-1, self.vocab_size)

        y_idx = y_idx.reshape(-1,)  # (batch_size * seq_len,)

        # do not compute loss for padding tokens! (batch_size * seq_len,)
        pad_mask = jnp.clip(jnp.abs(y_idx - self.pad_token), min=0, max=1)
        pad_mask = pad_mask.astype(self.dtype)

        loss_1 = jax.vmap(getitem)(logits, y_idx)
        loss_1 = loss_1 * pad_mask
        loss_1 = loss_1.sum()

        #(batch_size * seq_len, vocab) -> (batch_size * seq_len, 1) -> (1)
        loss_2 = jax.scipy.special.logsumexp(logits, axis=1) * pad_mask

        loss_2 = loss_2.sum()

        # average over non-masked elements, avoid division by zero
        loss = (loss_1 - loss_2) * (1/(pad_mask.sum() + 0.01))

        return -loss

    def generate(self, x_idx: list[int]) -> list[int]:
        """
        From a list of token integer tokens, generate more tokens
        """
        # pad the tokens up to seq_len
        num_tokens = len(x_idx)
        num_pad = self.context_size - num_tokens

        x_idx = x_idx + [self.pad_token] * num_pad
        x_idx = jnp.asarray(x_idx).reshape(1, self.context_size)

        last_token = int(x_idx[0, num_tokens])

        while last_token != self.eos_token and num_tokens < self.context_size:
            logits = self.gen_once(self.weights, x_idx=x_idx)
            last_token = int(jnp.argmax(logits[0, num_tokens, :]))
            x_idx = x_idx.at[0, num_tokens].set(last_token)
            num_tokens += 1

        stop_tokens = [self.pad_token, self.eos_token]

        return  [int(x) for x in x_idx[0] if x not in stop_tokens]

    @property
    def dtype(self) -> None:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: jnp.dtype) -> None:
        """ Cast all local variables to the new type """
        self._dtype = dtype
        self.weights = [w.astype(dtype) for w in self.weights]
        self.pos_embeddings = self.pos_embeddings.astype(dtype)

    def save(self, filename:Path):

        print("Saving Model: ", end="")
        state = {
            "vocab_size":self.vocab_size,
            "context_size": self.context_size,
            "dtype": self.weights[0].dtype,
            "x_dim":self.x_dim,
            "qk_dim":self.qk_dim,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "weights": self.weights,
        }

        with open(filename, "wb") as f:
            pickle.dump(state, f)

        print(filename)

    @classmethod
    def load(cls, filename:Path):
        with open(filename, "rb") as f:
            stuff = pickle.load(f)
        weights = stuff.pop("weights")
        model = cls(**stuff)
        model.weights = weights
        return model
