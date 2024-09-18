import time
from pathlib import Path

import jax
import jax.numpy as jnp

from tensorboardX import SummaryWriter

from datasets.dataloader import DataLoader
from toyGPT.gpt_model import GPTModel
from toyGPT.adamoptimizer import AdamOptimizer


@jax.jit
def sharded_accumulation(
    tensor_lo:jnp.ndarray,
    shards:int=20,
    dtype_hi:jnp.dtype=jnp.float32,
) -> jnp.ndarray:
    """
    - split a tensor along itrs second axis
    - each split compute mean over first axis
    - concat the tensor over the split axis (was second, now)
    """
    shard_size = tensor_lo.shape[0] // shards
    output = jnp.zeros(dtype=dtype_hi, shape=tensor_lo.shape[1:])
    for i in range(shards):
        shard = tensor_lo[i*shard_size:(i+1)*shard_size, :]
        output = output + shard.astype(dtype_hi).sum(0)

    shard = tensor_lo[(i+1)*shard_size:, :]
    output = output + shard.astype(dtype_hi).sum(0)

    output = output * (1 / tensor_lo.shape[0])

    return output


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
    loss_and_grad = jax.value_and_grad(model._loss, argnums=(0, ))

    # Compute gradient for each batch item, do not average the gradients
    # weight + minibatch -> param gradient for each batch item
    # (#params,) + (batchsize, 1, seq_len) -> (batch, #params)
    loss_and_grad_batch = jax.jit(jax.vmap(loss_and_grad, in_axes=(None, 0, 0)))

    for epoch in range(epochs):
        epoch_start = time.time()
        for i, x_idx in enumerate(dataloader):

            batch_start = time.time()

            # compute gradients in low precision
            # expands the data to (batch, 1, seq_len) as the batch dimension
            # is used for jav.vmap.
            loss_lo, grads = loss_and_grad_batch(
                [w.astype(dtype_lo) for w in weights_hi],
                x_idx[:, None, :-1],
                x_idx[:, None, 1:],
            )
            grads = grads[0]

            # convert the grasdients to high precision and accumulate
            # NOTE: remove overwrite low precision tensors to save RAM
            for j in range(len(grads)):
                grads[j] = sharded_accumulation(grads[j])
            # grads = [g.astype(dtype_hi).mean(0) for g in grads]

            # grads = [sharded_accumulation(g) for g in grads]

            # take the gradient step
            weights_hi = optimizer.update_params(weights_hi, grads)

            batch_time = time.time() - batch_start

            # just the admin
            loss = float(loss_lo.astype(dtype_hi).mean())
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
