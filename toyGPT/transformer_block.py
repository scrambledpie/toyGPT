import jax
import jax.numpy as jnp


def transformer_block(
    *,
    weights: jnp.array, # (X, QK), (X, QK), (X, X),  (X, X), (X, X)
    x_emb:jnp.array, # (BATCH_SIZE, SEQ_LEN, X_DIM)
    num_heads:int,
) -> jnp.array:  # (BATCH_SIZE, SEQ_LEN, X_DIM)
    """
    A forward pass through one transformer block consisting of a
    multi-head self-attenion ;ayer followed by 2 fully connected layers.

    There are a few minor quirks here that I need to "unquirk"
    TODO: the weight matrices can be specified to avoid unpacking
    TODO: use one matrix per attention head
    TODO: add layernorm layers
    TODO: currently require the concat of vals from heads == x_dim, remove this!

    Parameters
    ----------
    weights: jnp.array
        matrix of shape (2*qk_dim + 3*x_dim, x_dim) which is unpacked into
        5 matrices
            - w_q: queries (qk_dim, x_dim) get queries for all heads
            - w_k: keys (qk_dim, x_dim) get keys for all heads
            - w_v: values (x_dim, x_dim)  get values for all heads
            - w_f1: fully connected 1  (x_dim, x_dim)
            - w_f2: fully connected 2  (x_dim, x_dim)
    x_emb: jnp.array
        matrix of data (batch, seq_len, x_dim), a collection of embedded
        sequences.
    num_heads: int
        the number of attention heads

    Returns
    -------
    x : jnp.array
        output tensor the same shape as the input (batch, seq_len, x_dim)
    """
    _, seq_len, x_dim = x_emb.shape
    x_dim_head = x_dim // num_heads

    qk_dim = (weights.shape[1] - 3 * x_dim) // 2
    qk_dim_head = qk_dim // num_heads

    assert x_dim % num_heads == 0, f"wrong size multihead {x_dim} {num_heads}"
    assert qk_dim % num_heads == 0, f"wrong size multihead {qk_dim} {num_heads}"

    # w_q, w_k, w_v, w_f1, w_f2 = weights
    # unpack the weights
    i_1 = qk_dim
    i_2 = i_1 + qk_dim
    i_3 = i_2 + x_dim
    i_4 = i_3 + x_dim
    i_5 = i_4 + x_dim

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
    shape = (batchsize, seq_len, num_heads)
    queries = queries.reshape(shape + (qk_dim_head,))
    keys = keys.reshape(shape + (qk_dim_head,))
    values = values.reshape(shape + (x_dim_head,))

    # (BATCH_SIZE, HEADS, SEQ_LEN, QK_DIM_HEAD)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3) # (bs, heads, context, x_head)

    # (BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN)
    normaliser = 1/jnp.sqrt(qk_dim_head)
    qk_prod = jnp.matmul(queries, keys.transpose((0, 1, 3, 2))) * normaliser

    # apply causal mask
    # (BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN)
    causal_mask = jnp.tri(N=seq_len, k=-1, dtype=x_emb.dtype)
    causal_mask = causal_mask.transpose()
    causal_mask = -1e4 * causal_mask[None, None, : , :]

    qk_prod = jnp.triu(qk_prod) + causal_mask

    # (BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN)
    attn = jax.nn.softmax(qk_prod, axis=3)

    # (BATCH_SIZE, HEADS, SEQ_LEN, x_dim_head)
    diff = jnp.matmul(attn, values)
    diff = diff.transpose(0, 2, 1, 3)
    diff = diff.reshape(
        batchsize, seq_len, num_heads * x_dim_head
    )

    x = x_emb + diff
    x = x + jnp.clip(jnp.matmul(x, w_f1), min=0) # relu
    x = x + jnp.matmul(x, w_f2) # no activation

    return x
