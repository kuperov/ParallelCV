"""diag is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""

import jax


def print_devices():
    """Print summary of available devices to console."""
    device_list = [f"{d.device_kind} ({d.platform}{d.id})" for d in jax.devices()]
    if len(device_list) > 0:
        print(f'Detected devices: {", ".join(device_list)}')
    else:
        print("Only CPU is available. Check cuda/cudnn library versions.")


def to_arviz(theta: Theta, post_id: int) -> az.InferenceData:
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
