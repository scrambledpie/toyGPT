from toyGPT.gpt_model import GPTModel
from toyGPT.randmat import Random


def test_gpt_model(tmp_dir):
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


