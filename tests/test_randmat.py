import jax

from toyGPT.randmat import Random


def test_random():
    """ subsequent calls are unique """
    mat_1 = Random.randmat((10, 10))
    mat_2 = Random.randmat((10, 10))
    assert (mat_1 == mat_2).sum() == 0


def test_random_fixed_key():
    """ subsequent calls with fixed key are the same """
    Random.key = jax.random.key(0)
    mat_1 = Random.randmat((10, 10))

    Random.key = jax.random.key(0)
    mat_2 = Random.randmat((10, 10))
    assert (mat_1 == mat_2).sum() == 100


