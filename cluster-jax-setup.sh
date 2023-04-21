#!/bin/bash

# environment setup script for running jax on the cluster
# 21 April 2023, alexander.cooper@monash.edu

module add cuda/12.0
module add python/3.9.10-linux-centos7-haswell-gcc10.2.0

cd /home/acoo0002/fr51/parallel

# create virtual environment and install deps
virtualenv -p `which python` .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python setup.py develop
python -m pip install jupyter arviz
python -m pip install git+https://github.com/blackjax-devs/blackjax.git@7100bca3ea39def4bbeaa179a015f67abfa0b1f0
python -m pip install --upgrade git+https://github.com/kuperov/welford.git
