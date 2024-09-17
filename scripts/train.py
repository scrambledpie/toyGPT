import jax

from toyGPT.trainer import train_model
from datasets.wine_dataset import WineDataLoader
from toyGPT.gpt_model import GPTModel


def main():
    batchsize = 1000
    seq_len = 150
    vocab_size = 100
    dtype = jax.numpy.float32

    num_layers=2
    x_dim=256
    qk_dim=256

    wine_dataloader = WineDataLoader(
        batchsize=batchsize,
        seq_len=seq_len+1,
        vocab_size=vocab_size,
    )

    model = GPTModel(
        vocab_size=10000,
        eos_token=wine_dataloader.tokenizer.eos_token,
        pad_token=wine_dataloader.tokenizer.pad_token,
        context_size=seq_len,
        dtype=dtype,
        dtype_batch=dtype,
        num_layers=num_layers,
        x_dim=x_dim,
        qk_dim=qk_dim,
    )

    train_model(
        model=model,
        dataloader=wine_dataloader,
        epochs = 5,
    )


if __name__=="__main__":
    main()
