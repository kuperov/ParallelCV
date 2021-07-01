ploo: Parallel Brute Force LOO-CV
=================================

This is an experiment to try and implement LOO-CV with parallel processing on a GPU.

The project roadmap has been moved to [Github issues](https://github.com/kuperov/ploo/milestones)

Setting up
----------

You'll need python ≥ 3.7. It's probably easiest to use Anaconda but YMMV.

On a reasonable OS you should be able to create the virtual environment that contains runtime and development dependencies with:

    git clone git@github.com:kuperov/ploo.git
    cd ploo
    make config

This will add a pre-commit git hook that runs the linter. If you must use Windows, the makefiles probably won't work for you.

Check everything is working by running the unit tests:

    make test

Before commiting code, run the code formatter and linter (the pre-commit hook should lint automatically):

    make black
    make lint
