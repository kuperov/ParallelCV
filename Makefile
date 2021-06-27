.PHONY: all
all:
	@echo "No 'all' target"

.PHONY: nb
nb: venv
	venv/bin/jupyter-lab &

.PHONY: test
test: venv
	venv/bin/python -m unittest discover test

venv:
	# create python virtual environment and install deps
	rm -rf venv
	virtualenv venv
	venv/bin/pip3 install -r requirements.txt
	venv/bin/pip3 install -r requirements-dev.txt
	venv/bin/python3 setup.py develop
