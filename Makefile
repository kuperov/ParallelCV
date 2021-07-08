# ploo package 

PYTHON=venv/bin/python3
JUPYTER=venv/bin/jupyter-lab

define HELP
ploo package Makefile

The following targets are available:

  help     display this message
  nb       start a jupyter notebook
  test     run unit tests
  gpu      install version of jax for GPUs (CUDA 11x)
  cpu      install version of jax for CPUs
  venv     create the virtual environment in venv/
  pretty   format code
  lint     run code linters
  fixtures generate testing model fixtures
  config   first-time environment setup
  clean    remove generated files

You can invoke any of the above with:

  make <target>

endef
export HELP

.PHONY: help
help:
	@echo "$$HELP"

.PHONY: nb
nb: venv
	# start notebook server
	venv/bin/jupyter-lab &

.PHONY: test
test: venv
	# run unit tests
	$(PYTHON) -m unittest discover --buffer test

venv:
	# create python virtual environment and install deps
	# you'll need virtualenv and pip installed, obvs
	rm -rf venv
	python3 -m virtualenv --python=python3 venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/requirements.txt
	$(PYTHON) -m install -r requirements/requirements-dev.txt
	$(PYTHON) setup.py develop

pretty:
	venv/bin/isort --profile black ploo test
	venv/bin/black ploo test

lint:
	# check code formatting (fix with "make pretty")
	venv/bin/black --check ploo test
	venv/bin/isort --profile black --check-only ploo test
	# check for obvious bugs or code standard violations
	venv/bin/flake8 ploo test
	# venv/bin/pylint ploo test

gpu:
	# install gpu-enabled version of jax
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

cpu:
	# install cpu-only version of jax
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade "jax[cpu]"

.PHONY: fixtures
fixtures:
	venv/bin/python3 scripts/run_test_models.py

config: venv
	# Configure the development environment, if not already
	cp --no-clobber -r scripts/vscode .vscode
	# Set up git commit hooks, if not already
	cp --no-clobber scripts/git-hooks/* .git/hooks

clean:
	@echo Deleting build artefacts. You should manually remove venv.
	find ploo test -name __pycache__ | xargs rm -r
	rm -rf .ipynb_checkpoints notebooks/.ipynb_checkpoints .vscode build ploo.egg-info
