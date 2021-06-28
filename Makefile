.PHONY: all
all:
	@echo "No 'all' target"

.PHONY: nb
nb: venv
	# start notebook server
	venv/bin/jupyter-lab &

.PHONY: test
test: venv
	# run unti tests
	venv/bin/python -m unittest discover test

venv:
	# create python virtual environment and install deps
	rm -rf venv
	virtualenv venv
	venv/bin/pip3 install -r requirements.txt
	venv/bin/pip3 install -r requirements-dev.txt
	venv/bin/python3 setup.py develop

notebook_keypair.ignore/notebook_keypair.rsa:
	ssh-keygen -f `pwd`/notebook_keypair.nogit/notebook_keypair.rsa -t rsa -N ''

black:
	# format code
	venv/bin/black ploo test
