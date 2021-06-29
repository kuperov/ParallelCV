import unittest

from jax import numpy as jnp
from jax.scipy import stats as st

from ploo import CVModel, LogTransform, TransformedCVModel


class _GaussianVarianceModel(CVModel):
    r"""Test model: Gaussian with unknown variance and a single obs

    The prior for :math:`\sigma^2` is Gamma(a, b).

    Because :math:`\sigma^2` has only positive support, we need to transform
    it to cover the real line. One way is with the logarithmic transform.
    """

    def __init__(
        self,
        y: jnp.DeviceArray,
        mean: float = 0.0,
        prior_shape: float = 2.0,
        prior_rate: float = 2.0,
    ) -> None:
        self.y = y
        self.mean = mean
        self.prior_shape = prior_shape
        self.prior_rate = prior_rate

    def log_joint(self, cv_fold, sigma_sq):
        ll = st.norm.logpdf(self.y, loc=self.mean, scale=jnp.sqrt(sigma_sq))
        lp = st.gamma.logpdf(sigma_sq, a=self.prior_shape, scale=1.0 / self.prior_rate)
        return ll + lp

    def log_pred(self, y_tilde, param):
        return super().log_pred(y_tilde, param)


class _TransformedGaussianVarianceModel(TransformedCVModel):
    def __init__(self, model: CVModel) -> None:
        super().__init__(model)
        self.sigma_sq_transform = LogTransform()

    def to_unconstrained_coordinates(self, params):
        unconstrained = {"sigma_sq": self.sigma_sq_transform(params["sigma_sq"])}
        return unconstrained

    def to_constrained_coordinates(self, params):
        constrained = {"sigma_sq": self.sigma_sq_transform.reverse(params["sigma_sq"])}
        return constrained

    def log_det(self, params):
        return self.sigma_sq_transform.log_det(params["sigma_sq"])


class TestModelParam(unittest.TestCase):
    
    def setUp(self) -> None:
        y = jnp.array(5.)
        self.orig = _GaussianVarianceModel(y, mean=0., prior_shape=2., prior_rate=2.)
        self.tf = _TransformedGaussianVarianceModel(self.orig)

    def test_log_transform(self):
        lj_orig = self.orig.log_joint(-1, sigma_sq=2.5)
        lj_tf = self.tf.log_joint(-1, sigma_sq=jnp.log(2.5))
        ldet = self.tf.log_det(sigma_sq=2.5)
        self.assertEqual(lj_orig, lj_tf - ldet)


if __name__ == '__main__':
    unittest.main()
