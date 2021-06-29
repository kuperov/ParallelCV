from jax.interpreters.xla import DeviceArray
import unittest, os

from jax import numpy as jnp, random
from scipy import stats as st
import numpy as np
import pandas

from ploo import (
    run_hmc,
    warmup,
    full_data_inference,
    cross_validate,
    CVPosterior,
    DummyProgress,
    GaussianModel,
    WarmupResults,
)


class TestInference(unittest.TestCase):
    def setUp(self):
        self.y = GaussianModel.generate(N=200, mu=0.5, sigma=2, seed=42)
        self.gauss = GaussianModel(self.y)
        self.key = random.PRNGKey(42)
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
        wu = warmup(self.gauss, 600, 8, self.key)
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
            self.gauss, self.wu, draws=1000, chains=4, rng_key=self.key
        )
        self.assertEqual(states.position["mu"].shape, (1000, 4))

    def test_cross_validation(self):
        states = cross_validate(self.gauss, self.wu, draws=1e3, chains=2, rng_key=self.key)


if __name__ == "__main__":
    unittest.main()
