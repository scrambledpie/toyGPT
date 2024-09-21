import jax

from toyGPT.trainer import TrainModel
from datasets.wine_dataset import WineDataLoader
from toyGPT.gpt_model import GPTModel

from folders import make_new_folders


def main():
    """
    Train a GPT model on the Wine reviews dataset. Use FP32/TF32 precision on
    a single GPU.
    """
    batchsize = 100
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
        "fruity and spicy with a hint of",
        "chocolate like with a hint of",
        "an aftertaste of mushrooms with the scent of",
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

    checkpoint_dir, _ = make_new_folders()

    trainer = TrainModel(
        model=model,
        dataloader=wine_dataloader,
        checkpoint_dir=checkpoint_dir,
        prompts=prompts,
    )
    trainer.train(epochs=100)


if __name__=="__main__":
    main()
