import jax

from toyGPT.trainer import train_model
from datasets.wine_dataset import WineDataLoader
from toyGPT.gpt_model import GPTModel

from folders import make_new_folders


def main():
    """
    Train a GPT model on the Wine reviews dataset. Use FP32/TF32 precision on
    a single GPU.
    """
    batchsize = 1125
    seq_len = 150
    vocab_size = 10000
    dtype = jax.numpy.float32

    num_layers = 3
    num_heads = 4
    x_dim = 256
    qk_dim = 256

    wine_dataloader = WineDataLoader(
        batchsize=batchsize,
        seq_len=seq_len+1,
        vocab_size=vocab_size,
    )

    prompts = [
        "fruity and spicy",
        "flavor of smoke like",
        "chocolate is very"
    ]

    model = GPTModel(
        vocab_size=vocab_size,
        eos_token=wine_dataloader.tokenizer.eos_token,
        pad_token=wine_dataloader.tokenizer.pad_token,
        context_size=seq_len,
        dtype=dtype,
        num_layers=num_layers,
        x_dim=x_dim,
        qk_dim=qk_dim,
        num_heads=num_heads,
    )

    checkpoint_dir, log_dir = make_new_folders()

    train_model(
        model=model,
        dataloader=wine_dataloader,
        epochs = 15,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        prompts=prompts,
    )


if __name__=="__main__":
    main()
