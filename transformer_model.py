import pickle
from operator import getitem
from pathlib import Path

import jax
import jax.numpy as jnp

from debug_tools import nanprint


RNG_KEY = jax.random.key(0)


def randmat(
    shape:list[int],
    dtype,
    key,
    minval:float=0,
    maxval:float=1,
) -> jnp.ndarray:
    """ generate a random matrix with given shape """
    key, subkey = jax.random.split(key)
    return jax.random.uniform(
        minval=minval, maxval=maxval, shape=shape, key=subkey, dtype=dtype
    ), key


class Transformer:
    def __init__(
        self,
        vocab_size:int,
        context_size:int,
        dtype,
        dtype_batch,
        num_layers:int=2,
        x_dim:int=256,
        qk_dim:int=128,
        eos_token:int=2,
        pad_token:int=0,
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
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.dtype_batch = dtype_batch

        key =  RNG_KEY

        # initialise embeddings token_id -> embedding vector
        self.embedding, key = randmat(
            shape=(vocab_size, x_dim),
            dtype=dtype,
            key=key,
            minval=-1,
            maxval=1,
        )
        self.embedding_batch = self.embedding.astype(self.dtype_batch)

        #  initalise position embeddings token_index -> embedding vector
        self.pos_embeddings = self._init_pos_embeddings(
            context_size=context_size,
            x_dim=x_dim,
            dtype=dtype,
        )
        self.pos_embeddings_batch = self.pos_embeddings.astype(self.dtype_batch)
        # nanprint(x=self.pos_embeddings_batch, msg="pos_embeddings batch")

        # initialise transformer model weights
        maxval_x = 1 / jnp.sqrt(x_dim)

         # (1, 1, SEQ_LEN, SEQ_LEN)
        causal_mask = jnp.tri(N=context_size, k=-1, dtype=dtype_batch)
        causal_mask = causal_mask.transpose()
        self.causal_mask = -1e4 * causal_mask[None, None, : , :]

        # make one big weights matrix for each layer that we unpacked for
        # Q/K/V/ff1/ff2
        n_cols = [(qk_dim * 2 + x_dim * 3)] * self.num_layers + [vocab_size]
        self.weights = []
        for n_col in n_cols:
            weights_i, key = randmat(
                shape=(x_dim, n_col),
                dtype=dtype,
                key=key,
                minval=-maxval_x,
                maxval=maxval_x
            )
            self.weights.append(weights_i)

        self.grad_loss = jax.grad(self._loss)

        self.weights_batch = [w.astype(self.dtype_batch) for w in self.weights]

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

    def transformer_block(
        self,
        *,
        weights: list[jnp.array], # (X, QK), (X, QK) (X, X)
        x_emb:jnp.array, # (BATCH_SIZE, SEQ_LEN, X_DIM)
    ) -> jnp.array:  # (BATCH_SIZE, SEQ_LEN, X_DIM)
        """ forward pass through one transformer block """
        # w_q, w_k, w_v, w_f1, w_f2 = weights

        nanprint(x_emb, "trans block input x_emb")

        # unpack the weights
        i_1 = self.qk_dim
        i_2 = i_1 + self.qk_dim
        i_3 = i_2 + self.x_dim
        i_4 = i_3 + self.x_dim
        i_5 = i_4 + self.x_dim

        w_q = weights[:, :i_1]
        w_k = weights[:, i_1:i_2]
        w_v = weights[:, i_2:i_3]
        w_f1 = weights[:, i_3:i_4]
        w_f2 = weights[:, i_4:i_5]

        queries = jnp.matmul(x_emb, w_q)  # (BATCH_SIZE, SEQ_LEN, QK_DIM)
        keys = jnp.matmul(x_emb, w_k)     # (BATCH_SIZE, SEQ_LEN, QK_DIM)
        values = jnp.matmul(x_emb, w_v)   # (BATCH_SIZE, SEQ_LEN, X_DIM)

        batchsize, seq_len, _ = x_emb.shape

        # (BATCH_SIZE, HEADS, SEQ_LEN, QK_DIM_HEAD)
        shape = (batchsize, seq_len, self.num_heads)
        queries = queries.reshape(shape + (self.qk_dim_head,))
        keys = keys.reshape(shape + (self.qk_dim_head,))
        values = values.reshape(shape + (self.x_dim_head,))

        # (BATCH_SIZE, HEADS, SEQ_LEN, QK_DIM_HEAD)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3) # (bs, heads, context, x_head)

        # (BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN)
        normaliser = 1/jnp.sqrt(self.qk_dim_head)
        qk_prod = jnp.matmul(queries, keys.transpose((0, 1, 3, 2))) * normaliser

        nanprint(qk_prod, "qk_prod no mask")

        # apply causal mask
        # (BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN)
        qk_prod = jnp.triu(qk_prod) + self.causal_mask[:, :, :seq_len, :seq_len]

        nanprint(qk_prod, "qk_prod masked")

        # (BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN)
        attn = jax.nn.softmax(qk_prod, axis=3)

        nanprint(attn, "attn")

        # (BATCH_SIZE, HEADS, SEQ_LEN, x_dim_head)
        diff = jnp.matmul(attn, values)
        diff = diff.transpose(0, 2, 1, 3)
        diff = diff.reshape(
            batchsize, seq_len, self.num_heads * self.x_dim_head
        )

        nanprint(diff, "diff")

        x = x_emb + diff
        x = x + jnp.clip(jnp.matmul(x, w_f1), min=0) # relu
        x = x + jnp.matmul(x, w_f2) # no activation

        nanprint(x, "transformer block output")
        return x

    def _forward(
        self,
        weights_list: list[jnp.array],
        x_idx:jnp.array
    ) -> jnp.array:
        """ forward pass through full model """

        # {x_idx.device}"
        assert len(x_idx.shape) == 2, f" x_idx has wierd shape {x_idx.shape}"

        _, seq_len = x_idx.shape
        assert seq_len <= self.pos_embeddings_batch.shape[1], (
            f"out of context! {seq_len} > {self.pos_embeddings_batch.shape[1]}"
        )
        x = self.pos_embeddings_batch[:, :seq_len, :]
        x = x + self.embedding_batch[x_idx, :]

        for w in weights_list[:-1]:
            x = self.transformer_block(weights=w, x_emb=x)
        x = jnp.matmul(x, weights_list[-1])

        nanprint(x, "logits")
        return x

    def _loss(
        self,
        weights_list: jnp.array,
        x_idx:jnp.array, # (batch_size, seq_len)
        y_idx:jnp.array, # (batch_size, seq_len)
    ) -> jnp.array:

        # (batch_size, seq_len, vocab_size)
        logits = self._forward(weights_list, x_idx)

        # (batch_size * seq_len, vocab_size)
        logits = logits.reshape(-1, self.vocab_size)

        y_idx = y_idx.reshape(-1,)  # (batch_size * seq_len,)

        # do not compute loss for padding tokens! (batch_size * seq_len,)
        pad_mask = jnp.clip(jnp.abs(y_idx - self.pad_token), min=0, max=1)
        pad_mask = pad_mask.astype(self.dtype_batch)

        nanprint(pad_mask, "pad_mask")

        loss_1 = jax.vmap(getitem)(logits, y_idx)
        loss_1 = loss_1 * pad_mask
        loss_1 = loss_1.sum()

        nanprint(loss_1, "loss_1")


        #(batch_size * seq_len, vocab) -> (batch_size * seq_len, 1) -> (1)
        loss_2 = jax.scipy.special.logsumexp(logits, axis=1) * pad_mask

        # import pdb; pdb.set_trace()
        nanprint(loss_2, "loss_2 before sum")
        loss_2 = loss_2.sum()
        nanprint(loss_2, "loss_2 after sum")



        # average over non-masked elements
        loss = (loss_1 - loss_2) * (1/pad_mask.sum())


        nanprint(loss, "loss")

        return -loss

    def _generate_once(self, x_idx: jnp.array) -> jnp.array:
        # (1, len(prompt_tokens), VOCAB_SIZE)
        logits = self._forward(weights_list=self.weights, x_idx=x_idx)
        x_new = jnp.argmax(logits[0, -1, :]).reshape((1, 1))
        return x_new

    def generate(
        self,
        prompt_tokens:list[str],
        vocab:list[str]=None,
        max_tokens:int = 200,
    ) -> jnp.array:
        # assert isinstance(prompt_tokens, list)
        # assert all([isinstance(x, int) for x in prompt_tokens])

        generate_once = jax.jit(self._generate_once)
        # generate_once = self._generate_once


        x_idx = jnp.array(prompt_tokens).reshape(1, -1)

        if vocab:
            print(" ".join([vocab[i] for i in prompt_tokens]), end=" ")

        while (
            x_idx.shape[1] < self.context_size - 1 and
            x_idx[0][-1] != self.eos_token and
            x_idx.shape[1] < max_tokens
        ):
            x_new = generate_once(x_idx)

            x_new = x_new
            x_idx = jnp.concat([x_idx, x_new], axis=1)

            if vocab:
                print(vocab[x_idx[0, -1]], end=" ")
            else:
                print(x_idx.shape, x_idx)

        if vocab:
            return " ".join([vocab[i] for i in x_idx[0]])
        else:
            return x_idx[0, :]

    def save(self, filename:Path):

        print("Saving Model: ", end="")
        state = {
            "vocab_size":self.vocab_size,
            "context_size": self.context_size,
            "dtype": self.weights[0].dtype,
            "dtype_batch": self.weights_batch[0].dtype,
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




