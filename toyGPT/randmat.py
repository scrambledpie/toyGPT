import jax
import jax.numpy as jnp


class Random:
    """
    Singleton class that manages the RNG key
    """
    key = jax.random.key(0)
    @classmethod
    def randmat(
        cls,
        shape:list[int],
        dtype=jnp.float32,
        minval:float=0,
        maxval:float=1,
    ) -> jnp.ndarray:
        cls.key, subkey = jax.random.split(cls.key)
        matrix = jax.random.uniform(
            minval=minval,
            maxval=maxval,
            shape=shape,
            key=subkey,
            dtype=dtype,
        )
        return matrix
