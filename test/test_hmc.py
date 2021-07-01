from jax.interpreters.xla import DeviceArray
import unittest

from jax import numpy as jnp, random

from ploo import (
    GaussianModel,
    WarmupResults,
)
from ploo.hmc import warmup, full_data_inference, cross_validate


class TestInference(unittest.TestCase):
    def setUp(self):
        self.y = GaussianModel.generate(N=200, mu=0.5, sigma=2, seed=42)
        self.gauss = GaussianModel(self.y)
        self.rng_key = random.PRNGKey(42)
        # by passing in warmup results we can skip the warmup step
        self.wu = WarmupResults(
            step_size=1.215579867362976,
            mass_matrix=jnp.array([0.01815528, 0.00198543]),
            starting_values={
                "mu": jnp.array([0.86416173, 0.7386246, 0.9446525, 0.7386246]),
                "sigma": jnp.array([0.5996679, 0.6574026, 0.56209683, 0.6574026]),
            },
            int_steps=3,
        )

    def test_warmup(self):
        initial = self.gauss.to_inference_params(self.gauss.initial_value())
        wu = warmup(self.gauss.cv_potential, initial, 600, 8, self.rng_key)
        self.assertIsInstance(wu, WarmupResults)
        self.assertEqual(wu.int_steps, 3)
        self.assertIsInstance(wu.starting_values, dict)
        self.assertIsInstance(wu.starting_values["mu"], DeviceArray)
        self.assertEqual(wu.starting_values["mu"].shape, (8,))
        self.assertEqual(wu.starting_values["sigma"].shape, (8,))
        self.assertEqual(wu.step_size.shape, ())
        self.assertIsInstance(wu.code, str)

    def test_full_data_inference(self):
        states = full_data_inference(
            self.gauss.cv_potential, self.wu, draws=1000, chains=4, rng_key=self.rng_key
        )
        self.assertEqual(states.position["mu"].shape, (1000, 4))

    def test_cross_validation(self):
        accumulator, states = cross_validate(
            cv_potential=self.gauss.cv_potential,
            cv_cond_pred=self.gauss.log_cond_pred,
            warmup=self.wu,
            cv_folds=self.gauss.cv_folds(),
            draws=1e3,
            chains=2,
            rng_key=self.rng_key,
        )
        self.assertIsNotNone(states)


if __name__ == "__main__":
    unittest.main()
