import time
from pathlib import Path

import jax
import jax.numpy as jnp

from tensorboardX import SummaryWriter

from datasets.dataloader import DataLoader
from toyGPT.gpt_model import GPTModel
from toyGPT.adamoptimizer import AdamOptimizer


def train_model(
    model:GPTModel,
    dataloader:DataLoader,
    epochs:int=100,
    checkpoint_dir: Path|None=None,
    log_dir: Path|None = None,
) -> None:
    """
    Train a GPT model on the given dataset for given number of epochs. This
    uses the Adam optimnizer.

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
    dtype : jnp.dtype, optional
        the datatype for all floats in the model, by default jnp.float32
    """

    writer = None
    if log_dir is not None:
        writer = SummaryWriter(logdir=log_dir)
        print(f"Initialized tensorboard logging: {log_dir}")

    optimizer = AdamOptimizer(
        param_shapes=[w.shape for w in model.weights],
        dtype=model.dtype,
        learning_rate_init=0.004
    )

    loss_and_grad = jax.jit(jax.value_and_grad(model._loss, argnums=(0,)))

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
            if writer is not None:
                iter = epoch * len(dataloader) + i
                writer.add_scalar("loss", float(loss), iter)
                writer.add_scalar("prob(correct)", prob_correct, iter)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}:  {epoch_time:.2f} seconds")

        if writer is not None:
            writer.add_scalar("epoch time", epoch_time, epoch)

        if checkpoint_dir is not None:
            filename = checkpoint_dir / f"{epoch}.pt"
            model.save(filename=filename)

        print()
