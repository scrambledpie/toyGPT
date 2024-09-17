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
        """
        Sample random uniform matrix
        """
        cls.key, subkey = jax.random.split(cls.key)
        matrix = jax.random.uniform(
            minval=minval,
            maxval=maxval,
            shape=shape,
            key=subkey,
            dtype=dtype,
        )
        return matrix

    @classmethod
    def randint(
        cls,
        shape:list[int],
        dtype=jnp.int32,
        minval:float=0,
        maxval:float=10000,
    ):
        """
        Sample random integer matrix
        """
        cls.key, subkey = jax.random.split(cls.key)
        shape = tuple(shape) + (maxval - minval,)
        matrix = minval + jax.random.categorical(
            key=subkey,
            logits=jnp.ones(shape=shape, dtype=jnp.float16),
            axis=-1,
        ).astype(dtype)
        return matrix

