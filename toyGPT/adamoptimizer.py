import jax
import jax.numpy as jnp


@jax.jit
def _increment_state(
    params : list[jnp.array],
    grads : list[jnp.array],
    ms : list[jnp.array],
    vs : list[jnp.array],
    t: int,
    lr_init: float,
    beta_1:float,
    beta_2:float,
    epsilon:float,
 ) -> tuple[jnp.array, jnp.array, jnp.array]:
    """
    A stateless function to compute and return updated the model parameters
    and optimizer state.
    """
    ms = [beta_1 * m + (1 - beta_1) * grad for m, grad in zip(ms, grads)]
    vs = [beta_2 * v + (1 - beta_2) * (grad**2) for v, grad in zip(vs, grads)]
    lr = lr_init * jnp.sqrt(1 - beta_2**t) / (1 - beta_1**t)
    updates = [-lr * m / (jnp.sqrt(v) + epsilon) for m, v in zip(ms, vs)]
    params = [p + g for p, g in zip(params, updates)]
    return params, ms, vs


class AdamOptimizer:
    """
    The Adam optimizer, parameters are updated according to
        theta <- theta + learning_rate * m / sqrt(v)
    where
        m = moving_average(gradients)
        v = moving_average(gradients**2)
    """
    def __init__(
        self,
        param_shapes: list[jnp.array],
        dtype=jnp.float32,
        learning_rate_init=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ) -> tuple[jnp.array]:
        """ Construct the inital optimizer state """
        # Constants
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Variables
        self.t = jnp.int16(0)
        self.ms = [jnp.zeros(shape, dtype=dtype) for shape in param_shapes]
        self.vs = [jnp.zeros(shape, dtype=dtype) for shape in param_shapes]

    def update_params(
        self,
        params: list[jnp.array],
        grads: list[jnp.array],
    ) -> list[jnp.array]:
        """
        Update the optimizer state and update + return the new model paramters.
        """
        self.t = self.t + 1
        params, self.ms, self.vs = _increment_state(
            params,
            grads,
            self.ms,
            self.vs,
            self.t,
            self.learning_rate_init,
            self.beta_1,
            self.beta_2,
            self.epsilon,
        )
        return params
