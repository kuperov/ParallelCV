import jax
from jax import tree_map
import jax.numpy as jnp
from jax.scipy.special import logsumexp


# stack arrays in pytrees
def tree_stack(trees):
    return tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)


# stack arrays in pytrees
def tree_concat(trees):
    return tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


def logvar(logx, axis=0, ddof=0):
    """Compute log variance of logx."""
    n = logx.shape[axis]
    logmean = logsumexp(logx, axis=axis) - jnp.log(n)
    logsumx2 = logsumexp(2 * logx, axis=axis)
    return (
        logsumx2
        + jnp.log1p(-jnp.exp(jnp.log(n) + 2 * logmean - logsumx2))
        - jnp.log(n - ddof)
    )


def logmean(logx, axis=0):
    """Compute log mean of logx."""
    n = logx.shape[axis]
    return logsumexp(logx, axis=axis) - jnp.log(n)


def print_devices():
    """Print summary of available devices to console."""
    device_list = [f"{d.device_kind} ({d.platform}{d.id})" for d in jax.devices()]
    if len(device_list) > 0:
        print(f'Detected devices: {", ".join(device_list)}')
    else:
        print("Only CPU is available. Check cuda/cudnn library versions.")
