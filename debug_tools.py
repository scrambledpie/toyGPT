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
    return _format_size(_get_bytes(x))


def get_shape(x:list | jnp.ndarray):
    if isinstance(x, list) or isinstance(x, tuple):
        return [get_shape(x_i) for x_i in x]
    return x.shape


def nanprint(x: jnp.array, msg:str="") -> None:
    # return
    nans_exist = jnp.any(jnp.isnan(x))
    print(msg)
    print("    ", str(x.dtype), tuple(x.shape))
    # debug.print("    Nans {}, min {}, max {}", nans_exist, x.min(), x.max())
    print(f"    Nans {nans_exist}, min {x.min()}, max { x.max()}")
    print()




if __name__=="__main__":
    x = jnp.array([[0.1 for _ in range(1024)] for _ in range(1024)])

    print(f"x={get_shape(x)}")

    print(f"x={get_shape([x, x, [x, [x], [[x]]]])}")
