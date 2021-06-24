from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as buff:
    long_description = buff.read()


setup(
    name="ploo",
    version="0.0.1",
    description="Parallel leave-one-out CV",
    long_description=long_description,
    author="Alex Cooper, adapted from code by Colin Carroll",
    author_email="alex@acooper.org",
    url="https://github.com/kuperov/ploo",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["test"]),
    install_requires=["jax", "tqdm"],
    include_package_data=True,
)
