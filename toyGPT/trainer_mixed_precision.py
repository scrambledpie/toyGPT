import time
from pathlib import Path

import jax
import jax.numpy as jnp

from tensorboardX import SummaryWriter

from datasets.dataloader import DataLoader
from toyGPT.gpt_model import GPTModel
from toyGPT.adamoptimizer import AdamOptimizer


def train_model_mixed_precision(
    model:GPTModel,
    dataloader:DataLoader,
    epochs:int=100,
    checkpoint_dir: Path|None=None,
    log_dir: Path|None = None,
    dtype_lo:jnp.dtype=jnp.float16,
    dtype_hi:jnp.dtype=jnp.float32,
) -> None:
    """
    Train a GPT model on the given dataset for given number of epochs. This
    uses the Adam optimizer with mixed precision.

    Low precision
      - model weights (recast from high precision master weights)
      - all forward + backward pass compuations
      - gradients per batch element

    High Precision
      - model master weights
      - adam optimizer (gradient and squared gradient momentums)
      - recast gradients per batch element
      - gradients accumulated oevr batch elements

    Parameters
    ----------
    model : GPTModel
        a GPTModel instance, setup to accept data form the dataloader
    dataloader : DataLoader
        an iterable dataset returning minibatches of (batchsize, seq_len)
    epochs : int, optional
        numebr of epochs to train for, by default 100
    checkpoint_dir : Path | None, optional
        save model every epoch, by default None and nothing is saved.
    log_dir : Path | None, optional
        show loss in tensorboard, by default None and nothing is logged.
    dtype_lo : jnp.dtype, optional
        the datatype for all low precision operations, by default jnp.float16
    dtype_hi : jnp.dtype, optional
        the datatype for all high precision operations, by default jnp.float32
    """

    writer = None
    if log_dir is not None:
        writer = SummaryWriter(logdir=log_dir)
        print(f"Initialized tensorboard logging: {log_dir}")

    # weights high precision for optimization
    weights_hi = [w.astype(dtype_hi) for w in model.weights]


    # optimizer state high precision
    optimizer = AdamOptimizer(
        param_shapes=[w.shape for w in weights_hi],
        dtype=dtype_hi,
        learning_rate_init=0.004
    )

    # update all model parameters to float16
    model.dtype = dtype_lo

    # weight + minibatch -> param gradients for whole batch
    # (#params,) + (batchsize, seq_len) -> (#params,)

    # Compute gradient for each batch item, do not average the gradients
    # weight + minibatch -> param gradient for each batch item
    # (#params,) + (batchsize, 1, seq_len) -> (batch, #params)

    @jax.jit
    def loss_and_grad(weights_hi: list[jnp.ndarray], x_idx:jnp.ndarray):
        """
        Compute loss and gradients in dtype_lo and accumulate them in
        dtype_hi.
        """
        # compute gradient w.r.t. the 0th argument: weights
        foo = jax.value_and_grad(model._loss, argnums=(0,))

        # get FP16 gradient matrices for each batch item, map over batch axis
        foo = jax.vmap(foo, in_axes=(None, 0, 0))

        # (batchsize, 1), list[(batchsize, w.shape[0], w.shape[1]),....]
        loss, grads = foo(
                [w.astype(dtype_lo) for w in weights_hi],
                x_idx[:, None, :-1],
                x_idx[:, None, 1:],
        )
        grads = grads[0]  # unpack  grads w.r.t the 0th argument

        # cast to dtype_hi and accumulate.
        # NOTE: the accumulation does not seem to cause RAM usage issues!
        loss = loss.astype(dtype_hi).mean()
        for i in range(len(grads)):
            grads[i] = grads[i].mean(0, dtype=dtype_hi)

        return loss, grads


    for epoch in range(epochs):
        epoch_start = time.time()
        for i, x_idx in enumerate(dataloader):

            batch_start = time.time()

            loss, grads = loss_and_grad(weights_hi, x_idx)

            # take the gradient step
            weights_hi = optimizer.update_params(weights_hi, grads)

            # just the admin
            batch_time = time.time() - batch_start
            loss = float(loss)
            prob_correct = float(jnp.exp(-loss))

            print(
                f"{epoch} {i}: {loss:.3f} {prob_correct:.3f} "
                f"{batch_time:.3f} seconds"
            )
            if writer is not None:
                iter = epoch * len(dataloader) + i
                writer.add_scalar("loss", loss, iter)
                writer.add_scalar("prob(correct)", prob_correct, iter)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}:  {epoch_time:.2f} seconds")

        if writer is not None:
            writer.add_scalar("epoch time", epoch_time, epoch)

        if checkpoint_dir is not None:
            filename = checkpoint_dir / f"{epoch}.pt"
            model.save(filename=filename)

        print()
