from jax.interpreters.xla import DeviceArray
from ploo.inference import WarmupResults, warmup
import unittest, os

from jax import numpy as jnp, random
from scipy import stats as st
import numpy as np
import pandas

from ploo import run_hmc, CVPosterior, DummyProgress, GaussianModel


class TestInference(unittest.TestCase):
    def test_warmup(self):
        y = GaussianModel.generate(N=200, mu=0.5, sigma=2, seed=42)
        gauss = GaussianModel(y)
        key = random.PRNGKey(42)
        wu = warmup(gauss, 600, 8, key)
        self.assertIsInstance(wu, WarmupResults)
        self.assertEqual(wu.int_steps, 3)
        self.assertIsInstance(wu.starting_values, dict)
        self.assertIsInstance(wu.starting_values["mu"], DeviceArray)
        self.assertEqual(wu.starting_values["mu"].shape, (8,))
        self.assertEqual(wu.starting_values["sigma"].shape, (8,))
        self.assertEqual(wu.step_size.shape, ())
