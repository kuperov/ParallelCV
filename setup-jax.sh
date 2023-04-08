#!/bin/bash

# module add cuda/12.0
module add python/3.9.10-linux-centos7-haswell-gcc10.2.0

virtualenv -p `which python` .venv
source .venv/bin/activate
python3 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

