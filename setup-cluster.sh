#!/bin/sh
# This script creates the environment required to run the code on the Monash cluster

rm -rf .venv
module purge
# module add python/3.9.10-linux-centos7-haswell-gcc10.2.0
module add python/3.9.10-linux-centos7-cascadelake-gcc11.2.0
virtualenv .venv -p `which python3`
source .venv/bin/activate
# module add cudnn/8.2.4
pip3 install --upgrade pip
pip install --upgrade jax[cpu] tensorflow-probability blackjax chex pandas matplotlib click arviz
