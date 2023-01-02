pcv: Convergence diagnostic for parallel MCMC
=============================================

This repo implements an online convergence diagnostic (Rhat) on a GPU. It enables brute force cross-validation and posterior stacking with parallel processing.

Setting up
----------

You'll need python ≥ 3.7. Alex uses CPython 3.10, although it is probably easiest to use Anaconda. YMMV.

On a reasonable OS you should be able to create the virtual environment that contains runtime and development dependencies with:

    git clone git@github.com:kuperov/ParallelCV.git
    cd ParallelCV
    make config

If you don't already have a python development environment you prefer, VSCode is free and has a shallow learning curve.

