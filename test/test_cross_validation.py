"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest

import arviz as az
import chex
import jax

from ploo import compare
from ploo.models import GaussianVarianceModel


class TestCrossValidation(unittest.TestCase):
    """Does model selection via cross-validation work?"""

    def test_compare_elpd(self):
        """Check cross-validation for model selection

        All in one big test so we only have to run one set of cross-validations.
        We aren't retaining draws here, just using the accumulated elpd.
        """
        gen_key = jax.random.PRNGKey(seed=42)
        y = GaussianVarianceModel.generate(N=50, mean=0, sigma_sq=10, rng_key=gen_key)
        model_1 = GaussianVarianceModel(y, mean=0.0)  # good
        model_2 = GaussianVarianceModel(y, mean=-10.0)  # bad
        model_3 = GaussianVarianceModel(y, mean=50.0)  # awful
        chex.clear_trace_counter()
        post_1 = model_1.inference(draws=1e3, chains=4)
        chex.clear_trace_counter()
        post_2 = model_2.inference(draws=1e3, chains=4)
        chex.clear_trace_counter()
        post_3 = model_3.inference(draws=1e3, chains=4)
        chex.clear_trace_counter()
        cv_1 = post_1.cross_validate()
        chex.clear_trace_counter()
        cv_2 = post_2.cross_validate()
        chex.clear_trace_counter()
        cv_3 = post_3.cross_validate()
        # check comparisons across CVs
        cmp_res = compare(cv_1, cv_2, cv_3)
        self.assertEqual(cmp_res.names(), ["model0", "model1", "model2"])
        for model in ["LOO", "model0", "model1", "model2"]:
            self.assertIn(model, repr(cmp_res))
        cmp_res = compare(cv_1, bad_model=cv_2, awful_model=cv_3)
        self.assertEqual(cmp_res.names(), ["model0", "bad_model", "awful_model"])
        for model in ["LOO", "model0", "bad_model", "awful_model"]:
            self.assertIn(model, repr(cmp_res))
        self.assertIs(cmp_res[0], cv_1)
        self.assertIs(cmp_res["model0"], cv_1)
        # can a cv with no draws be represented as string?
        cv1_repr = repr(cv_1)
        self.assertIsInstance(cv1_repr, str)

    def test_one_cv(self):
        """Check a single cross-validation object, retaining draws"""
        gen_key = jax.random.PRNGKey(seed=42)
        y = GaussianVarianceModel.generate(N=50, mean=0, sigma_sq=10, rng_key=gen_key)
        model_1 = GaussianVarianceModel(y, mean=0.0)
        post_1 = model_1.inference(draws=1e3, chains=4)
        # LOO
        cv_1 = post_1.cross_validate(thin=2)
        # 50 folds x 4 chains x 1e3/2 = 500 draws per chain
        self.assertEqual(cv_1.states["sigma_sq"].shape, (50, 4, 500))
        m1_av_f0 = cv_1.arviz(cv_fold=0)
        summ0 = az.summary(m1_av_f0)
        self.assertEqual(len(summ0), 1)  # should have 1 variable, sigma_sq
        self.assertIsInstance(m1_av_f0, az.data.inference_data.InferenceData)
        m1_av_f1 = cv_1.arviz(cv_fold=1)
        self.assertIsInstance(m1_av_f1, az.data.inference_data.InferenceData)
        cv_repr = repr(cv_1)
        self.assertIsInstance(cv_repr, str)
        # K-fold
        cv_2 = post_1.cross_validate(thin=2, scheme="KFold", k=5)
        self.assertEqual(cv_2.states["sigma_sq"].shape, (5, 4, 500))


if __name__ == "__main__":
    unittest.main()
