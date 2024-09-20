import time
from pathlib import Path

import jax
import jax.numpy as jnp
from tensorboardX import SummaryWriter


from datasets.dataloader import DataLoader
from toyGPT.gpt_model import GPTModel
from toyGPT.adamoptimizer import AdamOptimizer


class TrainModel:
    """
    Train a GPT model on the given dataset using the Adam optimnizer.

    Parameters
    ----------
    model : GPTModel
        a GPTModel instance, setup to accept data form the dataloader
    dataloader : DataLoader
        an iterable dataset returning minibatches of (batchsize, seq_len)
    epochs : int, optional
        numebr of epochs to train for, by default 100
    checkpoint_dir : Path
        save model every epoch, by default None and nothing is saved.
    prompts : list[str]
        a list of strings to start sequence generation, generation is performed
        once per epoch.
    save_every : int
        the number of epochs between saving checkpoint.
    mixed_precision : bool
        if True, forwared and backward passes are performed in FP16 for speed,
        however ths current implementation does not actually reduce VRAM usage!
    """
    def __init__(
        self,
        model:GPTModel,
        dataloader:DataLoader,
        checkpoint_dir: Path,
        prompts:list[str]|None=None,
        save_every:int=1,
        mixed_precision:bool=False,
    ):
        self.model = model
        self.dataloader = dataloader
        self._save_every = save_every

        self.optimizer = AdamOptimizer(
            param_shapes=[w.shape for w in model.weights],
            dtype=model.dtype,
            learning_rate_init=0.004
        )
        if prompts is None:
            prompts = []
        self.prompt_tokens = dataloader.tokenizer(prompts)

        if mixed_precision:
            self.loss_and_grad = self.build_loss_and_grad_mixed_precision(
                model=model
            )
        else:
            self.loss_and_grad = self.build_loss_and_grad(model=model)

        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(logdir=checkpoint_dir)

        self._epochs_trained = 0

    def train(self, epochs:int=1) -> None:
        """ Train the model for epochs """
        for _ in range(epochs):
            epoch_start = time.time()
            for i, x_idx in enumerate(self.dataloader):
                self._process_batch(
                    batch_idx=i,
                    x_idx=x_idx,
                )
            # Epoch statrstics reporting
            self._epochs_trained += 1
            epoch_time = time.time() - epoch_start
            completions = "\n\n".join(self._run_completions())
            self.writer.add_text(
                "Completions", completions, self._epochs_trained
            )

            if self._epochs_trained%self._save_every == 0:
                self.save()

            print(f"Epoch {self._epochs_trained}:  {epoch_time:.2f} seconds")
            print(completions)

    def _process_batch(
        self,
        batch_idx:int,
        x_idx:jnp.ndarray,
    ) -> None:
        """ Perform one weight update using the minibatch of tokens x_idx """
        batch_start = time.time()

        # forward pass, backward pass, update weights
        loss, grads = self.loss_and_grad(self.model.weights, x_idx)
        self.model.weights = self.optimizer.update_params(
            self.model.weights,
            grads
        )

        # Batch statistics reporting
        batch_time = time.time() - batch_start
        prob_correct = float(jnp.exp(-loss))
        epoch = self._epochs_trained
        print(
            f"{epoch} {batch_idx}: {float(loss):.3f} {prob_correct:.3f} "
            f"{batch_time:.3f} seconds"
        )
        iter = epoch * len(self.dataloader) + batch_idx
        self.writer.add_scalar("loss", float(loss), iter)
        self.writer.add_scalar("prob(correct)", prob_correct, iter)

    @staticmethod
    def build_loss_and_grad(model:GPTModel) -> callable:
        """ build loss and gradient functions """
        return jax.jit(jax.value_and_grad(model._loss))

    @staticmethod
    def build_loss_and_grad_mixed_precision(model:GPTModel) -> callable:
        """
        Perform forward and backward pass in FP16 and accumulate grads in FP32.

        NOTE: this uses more RAM! Because we cast full gradient tensors to FP32
        and then accumulate over batch dimension instead of incrementally cast
        and accumulate.
        """
        model.dtype = jnp.float32

        @jax.jit
        def loss_and_grad(weights, x_idx):

            # compute gradient w.r.t. the 0th argument: weights
            foo = jax.value_and_grad(model._loss)

            # get FP16 gradient matrices for each batch item
            foo = jax.vmap(foo, in_axes=(None, 0))

            # (batchsize, 1), list[(batchsize, w.shape[0], w.shape[1]),....]
            loss, grads = foo(
                [w.astype(jnp.float16) for w in weights],
                x_idx[:, None, :]
            )
            # cast to dtype_hi and accumulate.
            # NOTE: the accumulation does not seem to cause RAM usage issues!
            loss = loss.astype(jnp.float32).mean()
            for i in range(len(grads)):
                grads[i] = grads[i].astype(jnp.float32).mean(0)

            return loss, grads

        return loss_and_grad

    def _run_completions(self) -> list[str]:
        """ Pass the prompts through the model and generate new text """
        vocab = self.dataloader.tokenizer.vocab
        completions = [self.model.generate(p) for p in self.prompt_tokens]
        completions = [" ".join([vocab[i] for i in p]) for p in completions]
        return completions

    def save(self) -> None:
        """ Save the mode and the status of the trainer """
        model_file = self.checkpoint_dir / f"{self._epochs_trained}_model.pt"
        self.model.save(filename=model_file)

        # TODO: implement trainer saving
        # trainer_file = self.checkpoint_dir / f"{epoch}_trainer.pt"
        # self.optimizer.save(filename=optimizer_file)

    @classmethod
    def restore(cls):
        # TODO: implement training restoration
        pass

