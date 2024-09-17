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
    mat_11 = Random.randint((10, 10))

    Random.key = jax.random.key(0)
    mat_2 = Random.randmat((10, 10))
    mat_22 = Random.randint((10, 10))

    assert (mat_1 == mat_2).sum() == 100
    assert (mat_11 == mat_22).sum() == 100


def test_randint():
    matrix = Random.randint(shape=(10, 10), minval=100, maxval=105)
    assert matrix.shape == (10, 10)
    assert (jax.numpy.round(matrix) == matrix).sum() == 100

    assert 100 <= matrix.min()
    assert matrix.max() <= 105
