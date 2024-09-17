import jax.numpy as jnp


def _get_bytes(x:jnp.array) -> int:
    if isinstance(x, list) or isinstance(x, tuple):
        return sum([0] + [_get_bytes(x_i) for x_i in x])

    num_elems = 1
    for d_i in x.shape:
        num_elems *= d_i
    return num_elems * x.dtype.itemsize


def _format_size(num_bytes:int) -> str:
    sizes = ["b", "kb", "Mb", "Gb"]

    for size in sizes:
        if num_bytes < 1024:
            break
        num_bytes = num_bytes / 1024

    return f"{num_bytes:.2f}{size}"


def get_size(x:list | jnp.ndarray) -> str:
    """ get memory of a tensor/list of tensors """
    return _format_size(_get_bytes(x))


def get_shape(x:list | jnp.ndarray):
    """
    Read shapes of lists of tensors, return shapes in same nested structure
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [get_shape(x_i) for x_i in x]
    return x.shape

