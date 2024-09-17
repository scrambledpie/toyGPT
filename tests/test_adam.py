import jax
import jax.numpy as jnp

from toyGPT.adamoptimizer import AdamOptimizer


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


def make_random_matrices(key) -> list[jnp.array]:
    matrices = []
    for i in range(1, 5):
        w_matrix, key = randmat((1000, i * 100), dtype=jnp.float32, key=key)
        matrices.append(w_matrix)
    return matrices, key


def test_adam():
    """
    Initialise some matrices and perform a few optimizer iteraitons with
    randomly generated gradients.
    """
    # initilze some random parameters
    key = RNG_KEY
    params, key = make_random_matrices(key)

    # initialize the optimizer
    optimizer = AdamOptimizer(param_shapes=[w.shape for w in params])

    # perform a few iterations of optimization with random gradients
    for _ in range(10):
        grads, key = make_random_matrices(key = key)
        params_new = optimizer.update_params(params=params, grads=grads)

        # make sure shape match and parameters values are different
        for p1, p2 in zip(params, params_new):
            assert p1.shape == p2.shape
            assert jnp.mean(p1 == p2) < 1.0

        # prepare for next iteration
        params = params_new


if __name__=="__main__":
    test_adam()



