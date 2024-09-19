import jax

from toyGPT.trainer_mixed_precision import train_model_mixed_precision
from datasets.wine_dataset import WineDataLoader
from toyGPT.gpt_model import GPTModel

from folders import make_new_folders


def main():
    """
    Train a GPT model on the Wine reviews dataset. Use mixed FP32/TF32 precision
    for teh optimizer and master weights and FP16 for forward+backward passes.
    Uses a single GPU.
    """
    batchsize = 100
    seq_len = 150
    vocab_size = 10000
    dtype_lo = jax.numpy.float32
    dtype_hi = jax.numpy.float16

    num_layers = 3
    num_heads = 4
    x_dim = 256
    qk_dim = 256

    wine_dataloader = WineDataLoader(
        batchsize=batchsize,
        seq_len=seq_len+1,
        vocab_size=vocab_size,
    )

    model = GPTModel(
        vocab_size=vocab_size,
        eos_token=wine_dataloader.tokenizer.eos_token,
        pad_token=wine_dataloader.tokenizer.pad_token,
        context_size=seq_len,
        dtype=dtype_lo,
        num_layers=num_layers,
        x_dim=x_dim,
        qk_dim=qk_dim,
        num_heads=num_heads,
    )

    checkpoint_dir, log_dir = make_new_folders()

    train_model_mixed_precision(
        model=model,
        dataloader=wine_dataloader,
        epochs = 5,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        dtype_lo=dtype_lo,
        dtype_hi=dtype_hi,
    )


if __name__=="__main__":
    main()
