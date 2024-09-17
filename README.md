# toyGPT
An implementation of a GPT model in plain jax/jax numpy trained on wine reviews

- Self attention layer implemented from scratch in JAX
- mixed precision support
- multi-GPU support (data parrallel)


[Kaggle wine reviews dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews)

# Setup
- install [JAX](https://jax.readthedocs.io/en/latest/quickstart.html)
- download the dataset and unzip into `datasets/`

# Usage
All entry points to run code are in `scripts/`
- `train.py` uses full precision training on a single GPU