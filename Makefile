# ploo package 

define HELP
ploo package Makefile

The following targets are available:

  help   display this message
  nb     start a jupyter notebook
  test   run unit tests
  venv   create the virtual environment in venv/
  black  format code using the black package
  lint   run code linter
  config first-time environment setup
  clean  remove generated files

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
	venv/bin/python -m unittest discover test

venv:
	# create python virtual environment and install deps
	# you'll need virtualenv and pip installed, obvs
	rm -rf venv
	python3 -m virtualenv --python=python3 venv
	venv/bin/pip3 install --upgrade pip
	venv/bin/pip3 install -r requirements.txt
	venv/bin/pip3 install -r requirements-dev.txt
	venv/bin/python3 setup.py develop

notebook_keypair.ignore/notebook_keypair.rsa:
	ssh-keygen -f `pwd`/notebook_keypair.nogit/notebook_keypair.rsa -t rsa -N ''

black:
	# format code
	venv/bin/black ploo test

lint:
	venv/bin/black --check ploo test
	venv/bin/flake8 ploo test

config: venv
	# Configure the development environment, if not already
	cp --no-clobber -r scripts/vscode .vscode
	# Set up git commit hooks, if not already
	cp --no-clobber scripts/git-hooks/* .git/hooks

clean:
	@echo Deleting build artefacts. You should manually remove venv.
	find ploo test -name __pycache__ | xargs rm -r
	rm -rf .ipynb_checkpoints notebooks/.ipynb_checkpoints .vscode build ploo.egg-info
