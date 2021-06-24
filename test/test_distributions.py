import jax.numpy as jnp
from numpy.testing import assert_almost_equal
import jax.scipy.stats as st
from minimc import neg_log_normal, neg_log_mvnormal, mixture, neg_log_funnel


def test_neg_log_normal():
    neg_log_p = neg_log_normal(2, 0.1)
    true_rv = st.norm(2, 0.1)
    for x in jnp.random.randn(10):
        assert_almost_equal(neg_log_p(x), -true_rv.logpdf(x))


def test_neg_log_mvnormal():
    mu = jnp.arange(10)
    cov = 0.8 * jnp.ones((10, 10)) + 0.2 * jnp.eye(10)
    neg_log_p = neg_log_mvnormal(mu, cov)
    true_rv = st.multivariate_normal(mu, cov)
    for x in jnp.random.randn(10, mu.shape[0]):
        assert_almost_equal(neg_log_p(x), -true_rv.logpdf(x))


def test_mixture_1d():
    neg_log_probs = [neg_log_normal(1.0, 1.0), neg_log_normal(-1.0, 1.0)]
    probs = [0.2, 0.8]
    neg_log_p = mixture(neg_log_probs, probs)

    true_rvs = [st.norm(1.0, 1.0), st.norm(-1.0, 1)]
    true_log_p = lambda x: -jnp.log(sum(p * rv.pdf(x) for p, rv in zip(probs, true_rvs)))
    for x in jnp.random.randn(10):
        assert_almost_equal(neg_log_p(x), true_log_p(x))


def test_neg_log_funnel():
    neg_log_p = neg_log_funnel()
    true_scale = st.norm(0, 1)
    for x in jnp.random.randn(10, 2):
        print(x)
        true_log_p = true_scale.logpdf(x[0]) + st.norm(0, jnp.exp(2 * x[0])).logpdf(x[1])
        assert_almost_equal(neg_log_p(x), -true_log_p)
