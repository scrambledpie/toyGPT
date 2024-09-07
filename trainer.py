import datetime
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from dataloader import WineDataLoader
from debug_tools import get_size
from tensorboardX import SummaryWriter


from transformer_model import Transformer
from adamoptimizer import AdamOptimier

DTYPE = jnp.float32
DTYPE_BATCH = jnp.float32
X_DIM = 256
SEQ_LEN = 150
QK_DIM = 256
BATCH_SIZE = 1250
NUM_EPOCHS = 1000
VOCAB_SIZE = 10000
NUM_HEADS = 4


base_loss = jnp.log(VOCAB_SIZE)


ROOT_DIR = Path(__file__).parent
DATASET_FILE = ROOT_DIR / "wines_array_jnp.npy"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

num_folders = len(list(CHECKPOINTS_DIR.glob("*")))

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = f"{num_folders}_{timestamp}"

CHECKPOINT_DIR = CHECKPOINTS_DIR / model_name
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
COMPLETIONS_FILE =  CHECKPOINT_DIR / "0_completions.txt"

LOG_DIR = ROOT_DIR / "tensorboard_logs" / model_name

USE_PARALLEL = 1==1


NUM_DEVICES = jax.device_count()
DEVICES = jax.devices()

writer = SummaryWriter(log_dir=LOG_DIR)


def main():

    # Step 1. Load Dataset
    dataloader = WineDataLoader(
        batchsize=BATCH_SIZE,
        context=SEQ_LEN,
        vocab_size=VOCAB_SIZE
    )

    print("Transfering Dataset to GPU... ", end="")
    if not DATASET_FILE.exists():
        print("Saving dataset")
        dataset = jnp.array(dataloader.ds_tokens, dtype=jnp.int16)
        jnp.save(DATASET_FILE, dataset)

    print("Restoring dataset.... ", end="")
    dataset = jnp.load(DATASET_FILE) # (130,000, 201)

    # truncate to full batches and seq_len + 1
    dataset = dataset[:, :SEQ_LEN + 1]
    num_batches = dataset.shape[0] // BATCH_SIZE
    dataset = dataset[:num_batches * BATCH_SIZE]

    # (130000, 1, 150)
    dataset_x = dataset[:, None,  :-1]
    dataset_y = dataset[:, None, 1:]

    print(f"DONE size={get_size(dataset)}")

    # Step 2. Load Model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        context_size=SEQ_LEN,
        num_layers=4,
        x_dim=X_DIM,
        qk_dim=QK_DIM,
        dtype=DTYPE,
        dtype_batch=DTYPE_BATCH,
        eos_token=dataloader.tokenizer.eos_token,
        pad_token=dataloader.tokenizer.pad_token,
        num_heads=NUM_HEADS,
    )

    print(
        f"Found {NUM_DEVICES=} {jax.devices()} "
        f"weights = {get_size(model.weights)}"
    )

    # Step 3. Instantiate optimizxer
    optimizer = AdamOptimier(
        param_shapes=[w.shape for w in model.weights],
        dtype=DTYPE,
        learning_rate_init=0.004
    )


    # Step 4. build the gradient update opertations
    def grad_params_single_gpu(w_16, x_idx, y_idx):
        """
        w: list[jnp.array]
        x: jnp.array (batch_size_per_gpu, seq_len, x_dim)
        y: jnp.array (batch_size_per_gpu, seq_len, x_dim)
        """
        grad_params = jax.value_and_grad(model._loss, argnums=[0])
        grad_params = jax.vmap(grad_params, in_axes=[None, 0, 0])
        loss, grads = grad_params(w_16, x_idx, y_idx)
        grads = grads[0]
        loss_32 = loss.astype(DTYPE).mean(0)
        grads_32 = [g.astype(DTYPE).mean(0) for g in grads]
        return loss_32, grads_32

    if not USE_PARALLEL:
        grad_params = jax.jit(grad_params_single_gpu)
        # grad_params = grad_params_single_gpu
    else:
        # x_idx, y_idx shapes (num_devices, batchsize, 1, seq_len)
        grad_params_par = jax.pmap(grad_params_single_gpu, in_axes=(None, 0, 0))

        new_shape = (NUM_DEVICES, BATCH_SIZE // NUM_DEVICES, 1, SEQ_LEN)
        def grad_params(w_16, x_idx, y_idx):
            x_idx = x_idx.reshape(new_shape)
            y_idx = y_idx.reshape(new_shape)
            loss_32, grad_weights_32 = grad_params_par(w_16, x_idx, y_idx)

            # mean over GPU axis
            loss_32 = loss_32.mean()
            grad_weights_32 = [g.mean(axis=0) for g in grad_weights_32]
            return loss_32, grad_weights_32

    tick_epoch = time.time()
    first_gen = True

    # generator = jax.jit(model.generate)
    generator = model.generate

    for epoch in range(NUM_EPOCHS):
        # Save the model
        filename = CHECKPOINT_DIR / f"model_{str(epoch).zfill(3)}.ckpt"
        model.save(filename=filename)

        # Time the epoch
        epoch_duration = time.time() - tick_epoch
        tick_epoch = time.time()

        if epoch > 0:
            # show time
            print(f"\nEpoch: {epoch - 1} {epoch_duration} seconds\n\n")
            writer.add_scalar("epoch_time", epoch_duration, epoch)


        if epoch%20 == 0:
            # Do some generation
            print("\n\nStarting Generation")
            prompt_str = "this is tart and fruity , the flavours of"
            prompt_idx =  dataloader.tokenizer([prompt_str])[0]
            completion_str = generator(
                prompt_tokens=prompt_idx,
                max_tokens=len(prompt_idx) + 5 if first_gen else 200,
                vocab = dataloader.tokenizer.vocab,
            )
            print(epoch, completion_str)
            writer.add_text("completions", completion_str, epoch)

            with open(COMPLETIONS_FILE, "a") as f:
                f.write(f"\n{epoch=}\n {completion_str}\n")
            first_gen = False


        for i in range(num_batches):

            tick = time.time()

            # (BATCH_SIZE, 1, SEQ_LEN)
            x_idx = dataset_x[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :, :]
            y_idx = dataset_y[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :, :]

            loss_32, grads_32 = grad_params(
                w_16 = [w.astype(DTYPE_BATCH) for w in model.weights],
                x_idx=x_idx,
                y_idx=y_idx,
            )

            model.weights = optimizer.update_params(
                params=model.weights,
                grads=grads_32,
            )
            tock = time.time() - tick

            # nanprint(optimizer_state[-1][0], "ms[0]")
            loss_val = float(loss_32.mean())
            prob_y = float(jnp.exp(-loss_val))

            # record stuff in tensorboard
            time_step = epoch * num_batches + i
            writer.add_scalar("loss", loss_val, time_step)
            writer.add_scalar("prob_y", prob_y, time_step)
            writer.add_scalar("iteration_time", tock, time_step)

            print(
                f"{epoch=}, {i=} loss={loss_val:.3f} "
                f"prob={jnp.exp(-loss_val):.3f}"
                f" (baseloss {base_loss:.1f}) {tock:.3f} seconds"
            )


main()


