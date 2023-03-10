from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as buff:
    long_description = buff.read()


def _parse_requirements(req_path):
    with open(path.join(here, "requirements", req_path)) as req_file:
        return [
            line.rstrip()
            for line in req_file
            if not (line.isspace() or line.startswith("#"))
        ]


setup(
    name="pcv",
    version="0.0.9",
    description="Parallel MCMC diagnostic experiment",
    long_description=long_description,
    author="Alex Cooper",
    author_email="alex@acooper.org",
    url="https://github.com/kuperov/ParallelCV",
    license="LICENSE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['pcv','pcv.models'],
    package_data={'pcv.models': ['radon_all.json.zip']},
    install_requires=_parse_requirements("requirements.txt"),
    include_package_data=True,
)
