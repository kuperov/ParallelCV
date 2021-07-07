"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest

from jax import numpy as jnp
from jax import random
from jax.interpreters.xla import DeviceArray

from ploo.hmc import WarmupResults, cross_validate, full_data_inference, warmup
from ploo.models import GaussianModel


class TestInference(unittest.TestCase):
    """Check HMC runs okay - warmup, inference, CV"""

    def setUp(self):
        self.y = GaussianModel.generate(N=200, mu=0.5, sigma=2, seed=42)
        self.gauss = GaussianModel(self.y)
        self.rng_key = random.PRNGKey(42)
        # by passing in warmup results we can skip the warmup step
        self.warmup = WarmupResults(
            step_size=1.215579867362976,
            mass_matrix=jnp.array([0.01815528, 0.00198543]),
            starting_values={
                "mu": jnp.array([0.86416173, 0.7386246, 0.9446525, 0.7386246]),
                "sigma": jnp.array([0.5996679, 0.6574026, 0.56209683, 0.6574026]),
            },
            int_steps=3,
        )

    def test_warmup(self):
        """Test warmup. See also test_model.py"""
        initial = self.gauss.to_inference_params(self.gauss.initial_value())
        warmup_res = warmup(self.gauss.potential, initial, 600, 8, self.rng_key)
        self.assertIsInstance(warmup_res, WarmupResults)
        self.assertEqual(warmup_res.int_steps, 3)
        self.assertIsInstance(warmup_res.starting_values, dict)
        self.assertIsInstance(warmup_res.starting_values["mu"], DeviceArray)
        self.assertEqual(warmup_res.starting_values["mu"].shape, (8,))
        self.assertEqual(warmup_res.starting_values["sigma"].shape, (8,))
        self.assertEqual(warmup_res.step_size.shape, ())
        self.assertIsInstance(warmup_res.code, str)

    def test_full_data_inference(self):
        """Smoke test full-data inference. Better tests in test_model.py"""
        states = full_data_inference(
            self.gauss.cv_potential,
            self.warmup,
            draws=1000,
            chains=4,
            rng_key=self.rng_key,
        )
        self.assertEqual(states.position["mu"].shape, (1000, 4))

    def test_cross_validation(self):
        """Smoke test cross-validation. Better tests in test_model.py"""
        accumulator, states = cross_validate(
            cv_potential=self.gauss.cv_potential,
            cv_cond_pred=self.gauss.log_cond_pred,
            warmup_res=self.warmup,
            cv_folds=self.gauss.cv_folds(),
            draws=1e3,
            chains=2,
            rng_key=self.rng_key,
        )
        self.assertIsNotNone(states)
        self.assertEqual(accumulator.divergence_count, 0)


if __name__ == "__main__":
    unittest.main()
