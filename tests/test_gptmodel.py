from pathlib import Path

import jax

from toyGPT.gpt_model import GPTModel
from toyGPT.randmat import Random


def test_gpt_model():
    """ Instantiate and execute a GPT model """
    batchsize = 10
    seq_len = 150
    vocab_size = 100

    model = GPTModel(
        vocab_size=vocab_size,
        context_size=seq_len,
        eos_token=0,
        pad_token=vocab_size + 1
    )

    # include pad tokens
    xy_idx = Random.randint(
        shape=(batchsize, seq_len + 1),
        minval=0,
        maxval=vocab_size + 1,
    )

    logits = model._forward(
        weights_list=model.weights,
        x_idx=xy_idx[:, :seq_len],
    )

    assert logits.shape == (batchsize, seq_len, vocab_size)

    loss = model.compute_loss(weights_list=model.weights, xy_idx=xy_idx)

    # loss is a scalar
    loss = float(loss)

    # -log(prob) can only be positive
    assert loss > 0

    # check differentiability
    loss_and_grad = jax.value_and_grad(model.compute_loss, argnums=(0,))
    _, grads = loss_and_grad(model.weights, xy_idx)
    grads = grads[0]

    for w, grad in zip(model.weights, grads):
        assert w.shape == grad.shape


def test_save_load(tmp_path):
    """
    Instantiate, save and load the model, make sure all variables are restored.
    """
    seq_len = 150
    vocab_size = 100

    model = GPTModel(
        vocab_size=vocab_size,
        context_size=seq_len,
        eos_token=0,
        pad_token=vocab_size + 1
    )

    savefile = Path(tmp_path) / "gpt.pkl"
    model.save(savefile)
    model_new = GPTModel.load(savefile)

    # ensure all the weights are restored
    for w1, w2 in zip(model.weights, model_new.weights):
        assert (w1 == w2).mean() == 1

    # ensure all the other attributes are restored
    for attr in [
        "context_size",
        "x_dim",
        "qk_dim",
        "vocab_size",
        "eos_token",
        "pad_token",
        "num_heads",
        "num_layers",
    ]:
        assert getattr(model, attr) == getattr(model_new, attr)


def test_generation():
    """
    Prompt the model with arbitrary tokens, let it generate some more
    """
    seq_len = 150
    vocab_size = 100

    model = GPTModel(
        vocab_size=vocab_size,
        context_size=seq_len,
        eos_token=0,
        pad_token=vocab_size + 1
    )
    prompt_tokens = [2, 3, 4, 5, 6, 7]

    completion = model.generate(prompt_tokens)

    for x_i in completion:
        assert 0 <= x_i <= vocab_size


if __name__=="__main__":
    test_generation()
