import chex
import jax.numpy as jnp
import arviz as az


def online_cv():
    def make_fold(fold_id):
        results = fold_posterior(
            prng_key=inference_key,
            inference_loop=online_inference_loop,
            logjoint_density=lambda theta: logjoint_density(theta, fold_id),
            log_p=lambda theta: log_pred(theta, fold_id),
            make_initial_pos=make_initial_pos,
            num_chains=10,
            num_samples=2000,
            warmup_iter=1000,
        )
        return results

    online_fold_states = jax.vmap(make_fold)(jnp.arange(5))


def offline_cv_fold(fold_id):
    def replay_fold(fold_id, inference_key=inference_key):
        results, trace = fold_posterior(
            prng_key=inference_key,
            inference_loop=offline_inference_loop,
            logjoint_density=lambda theta: logjoint_density(theta, fold_id),
            log_p=lambda theta: log_pred(theta, fold_id),
            make_initial_pos=make_initial_pos,
            num_chains=10,
            num_samples=2000,
            warmup_iter=1000,
        )
        pos = trace.position
        theta_dict = az.convert_to_inference_data(
            dict(beta=pos.beta, sigsq=jax.vmap(sigsq_t.forward)(pos.sigsq))
        )
        trace_az = az.convert_to_inference_data(theta_dict)
        return results, trace_az


class CVScheme:
    """Generic CV scheme class

    Methods:
        name: name of the scheme suitable for plots and output
        n_folds: number of folds, always numbered from 0
        test_mask: boolean mask for test data for fold i
        train_mask: boolean mask for train data for fold i
        S_test: selection matrix for testing data for fold i
        S_train: selection matrix for training data for fold i
    """

    def __init__(self, T: int) -> None:
        super().__init__()
        self.T = T

    def name(self) -> str:
        raise NotImplementedError()

    def n_folds(self) -> int:
        raise NotImplementedError()

    def test_mask(self, i: int) -> chex.Array:
        raise NotImplementedError()

    def train_mask(self, i: int) -> chex.Array:
        raise NotImplementedError()

    def S_test(self, i: int) -> chex.Array:
        """Selection matrix for test data for this scheme

        Args:
            i:  fold number, 0-based
        """
        mask = self.test_mask(i)
        chex.assert_shape(mask, (self.T,))
        chex.assert_type(mask, jnp.bool_)
        S = jnp.diag(1.0 * mask)
        return S[mask, :]

    def S_train(self, i: int) -> chex.Array:
        """Selection matrix for training data for this scheme

        Args:
            i:  fold number, 0-based
        """
        mask = self.train_mask(i)
        chex.assert_shape(mask, (self.T,))
        chex.assert_type(mask, jnp.bool_)
        S = jnp.diag(1.0 * mask)
        return S[mask, :]


class LOOCVScheme(CVScheme):
    """Leave-one-out CV scheme"""

    def name(self) -> str:
        return "LOO"

    def n_folds(self) -> int:
        return self.T

    def train_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) != i  # type: ignore

    def test_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) == i  # type: ignore


class KFoldCVScheme(CVScheme):
    """K-fold CV scheme"""

    def __init__(self, T: int, k: int) -> None:
        self.k = k
        self.block_size = T // k
        super().__init__(T)

    def name(self) -> str:
        return f"{self.k}-fold"

    def n_folds(self) -> int:
        return self.k

    def train_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) // self.block_size != i

    def test_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) // self.block_size == i


class PointwiseKFoldCVScheme(CVScheme):
    """Pointwise K-fold scheme.

    This is an (inefficient) scheme for evaluating K-fold pointwise. It's like
    LOO but it uses the training sets from K-fold.

    Best use with lengths T that are multiples of the block size
    """

    def __init__(self, T: int, k: int) -> None:
        self.k = k
        self.block_size = T // k
        super().__init__(T)

    def name(self) -> str:
        return f"Pointwise {self.k}-fold"

    def n_folds(self) -> int:
        return self.T

    def train_mask(self, t: int) -> chex.Array:
        # The k-fold training set: missing the whole block
        block = t // self.block_size
        return jnp.arange(self.T) // self.block_size != block

    def test_mask(self, t: int) -> chex.Array:
        # The LOO testing set: just one variate
        return jnp.arange(self.T) == t


class HVBlockCVScheme(CVScheme):
    """H-block and HV-block CV schemes"""

    def __init__(self, T: int, h: int, v: int) -> None:
        super().__init__(T)
        self.h = h
        self.v = v

    @classmethod
    def from_delta(cls, T: int, delta: float) -> "HVBlockCVScheme":
        """Create HV-block scheme from delta hyperparameter"""
        h = jnp.floor(T**delta)
        v = jnp.floor(min(0.1 * T, (T - T ^ delta - 2 * h - 1) / 2))
        return cls(T, h, v)

    def name(self) -> str:
        return f"hv-block (h={self.h}, v={self.v})"

    def n_folds(self) -> int:
        return self.T

    def train_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return jnp.logical_or(idxs < i - self.v - self.h, idxs > i + self.v + self.h)

    def test_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return jnp.logical_and(idxs >= i - self.v, idxs <= i + self.v)


class LFOCVScheme(CVScheme):
    """LFO CV scheme

    Attrs:
        T: number of time points
        h: size of the halo
        v: size of the validation block
        m: size of the initial margin
    """

    def __init__(self, T: int, h: int, v: int, m: int) -> None:
        super().__init__(T)
        self.h = h
        self.v = v
        self.m = m

    def name(self) -> str:
        return f"LFO (h={self.h}, v={self.v}, m={self.m})"

    def n_folds(self) -> int:
        return self.T - self.m

    def train_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return idxs < i + self.m - self.v - self.h

    def test_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return jnp.logical_and(idxs >= i + self.m - self.v, idxs <= i + self.m + self.v)
