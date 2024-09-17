from toyGPT.transformer_block import transformer_block
from toyGPT.randmat import Random


def test_transformer_block():
    """
    Run some data random data through, make sure it produces the right shape.
    """
    x_dim = 104
    batch_size = 1024
    qk_dim = 128
    seq_len = 321

    x_emb = Random.randmat(shape=(batch_size, seq_len, x_dim))
    weights = Random.randmat(shape=(x_dim, qk_dim * 2 + 3 *x_dim))

    for num_heads in [1, 2, 4, 8]:
        x_output = transformer_block(
            weights=weights,
            x_emb=x_emb,
            num_heads=num_heads
        )
        assert x_output.shape == x_emb.shape
