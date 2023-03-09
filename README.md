pcv: Convergence diagnostic for parallel MCMC
=============================================

This repo implements an online convergence diagnostic (Rhat) on a GPU. It enables brute force cross-validation and posterior stacking with parallel processing.

Setting up
----------

You'll need python ≥ 3.7. Alex uses CPython 3.10, although it is probably easiest to use Anaconda. YMMV.

You need to create a virtual environment and install the
dependencies in requirements/requirements.txt. Note that the
blackjax dependency has been held back to an older version 
because its API is changing. The latest version won't work.

On a Debian-ish linux box you can do:

    git clone git@github.com:kuperov/ParallelCV.git
    cd ParallelCV
    make config
