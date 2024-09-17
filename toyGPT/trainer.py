import time
from pathlib import Path

import jax
import jax.numpy as jnp

from datasets.dataloader import DataLoader
from toyGPT.gpt_model import GPTModel
from toyGPT.adamoptimizer import AdamOptimizer


def train_model(
    model:GPTModel,
    dataloader:DataLoader,
    epochs:int=100,
    checkpoint_dir: Path|None=None,
    log_dir: Path|None = None,
    dtype: jnp.dtype=jnp.float32,
) -> None:

    optimizer = AdamOptimizer(
        param_shapes=[w.shape for w in model.weights],
        dtype=dtype,
        learning_rate_init=0.004
    )

    loss_and_grad = jax.jit(jax.value_and_grad(model._loss, argnums=(0,)))
    # loss_and_grad = jax.value_and_grad(model._loss, argnums=(0,))


    for epoch in range(epochs):
        epoch_start = time.time()
        for i, x_idx in enumerate(dataloader):


            batch_start = time.time()

            loss, grads = loss_and_grad(
                model.weights,
                x_idx[:, :-1],
                x_idx[:, 1:],
            )
            grads = grads[0]

            model.weights = optimizer.update_params(model.weights, grads)

            batch_time = time.time() - batch_start

            prob_correct = float(jnp.exp(-loss))

            print(
                f"{epoch} {i}: {float(loss):.3f} {prob_correct:.3f} "
                f"{batch_time:.3f} seconds"
            )

        epoch_time = time.time() - epoch_start
        print(epoch, epoch_time)
        print()


