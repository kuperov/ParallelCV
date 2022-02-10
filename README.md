pcv: Convergence diagnostic for parallel MCMC
=============================================

This is an experiment to try and implement an online Rhat on a GPU.
It enables brute force cross-validation with parallel processing.

Setting up
----------

You'll need python ≥ 3.7. Alex uses CPython 3.8, although it is probably easiest to use Anaconda. YMMV.

On a reasonable OS you should be able to create the virtual environment that contains runtime and development dependencies with:

    git clone git@github.com:kuperov/ParallelCV.git
    cd ParallelCV
    make config

This will add a pre-commit git hook that runs the linter. The Makefile probably won't work on Windows without WSL.

Check everything is working by running the unit tests:

    make test

Before commiting code, run the code formatter and linter (the pre-commit hook should lint automatically):

    make pretty
    make lint

If you don't already have a python development environment you prefer, VSCode is free and has a shallow learning curve.

Usage
-----

There are example notebooks in the `notebooks` directory. In short, you need to provide a log joint density function,
initial state generator, and a few other bits and bobs.
