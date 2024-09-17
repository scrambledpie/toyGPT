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
    x_idx = Random.randint(
        shape=(batchsize, seq_len),
        minval=0,
        maxval=vocab_size + 1,
    )

    logits = model._forward(weights_list=model.weights, x_idx=x_idx)

    assert logits.shape == (batchsize, seq_len, vocab_size)

    loss = model._loss(weights_list=model.weights, x_idx=x_idx, y_idx=x_idx)

    # loss is a scalar
    loss = float(loss)

    # -log(prob) can only be positive
    assert loss > 0

    # check differentiability
    loss_and_grad = jax.value_and_grad(model._loss, argnums=(0,))
    _, grads = loss_and_grad(model.weights, x_idx, x_idx)
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



