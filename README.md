ploo: Parallel Brute Force LOO-CV
=================================

This is an experiment to try and implement LOO-CV with parallel processing on a GPU.

Setting up
----------

You'll need python ≥ 3.7. It's probably easiest to use Anaconda but YMMV.

On a reasonable OS you should be able to create the virtual environment that contains runtime and development dependencies with:

    make venv

If you must run Windows, use WSL.

Check everything is working by running the unit tests:

    make test

Hint: at the time of writing, everything is not working.

Roadmap
-------

This list makes me feel tired just looking at it.

  * [x] Black-box HMC inference using Stan warmup
  * [x] Basic posterior summary output for experiments
  * [ ] LOO implementation using parallel chains
  * [ ] Functions for comparing models
  * [ ] Inference for multiple models in parallel
  * [ ] LOO for multiple models in parallel
  * [ ] PSIS implementation as benchmark
