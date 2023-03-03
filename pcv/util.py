import jax
import arviz as az
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp



def logvar(logx, axis=0, ddof=0):
    """Compute log variance of logx."""
    n = logx.shape[axis]
    logmean = logsumexp(logx, axis=axis) - jnp.log(n)
    logsumx2 = logsumexp(2*logx, axis=axis)
    return (
        logsumx2
        + jnp.log1p(-jnp.exp(jnp.log(n) + 2*logmean - logsumx2))
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


def to_arviz(theta, post_id: int) -> az.InferenceData:
    """Export a chain of draws to Arviz for visualization, etc.

    Args:
        theta: pytreee of draws
        post_id: zero-based posterior id
    """
    pos = tree_map(lambda x: x[post_id, ...], theta)
    theta_dict = az.convert_to_inference_data(
        dict(beta=pos.beta, sigsq=jax.vmap(sigsq_t.forward)(pos.sigsq))
    )
    return az.convert_to_inference_data(theta_dict)
