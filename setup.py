"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as buff:
    long_description = buff.read()


def _parse_requirements(req_path):
    with open(path.join(here, 'requirements', req_path)) as req_file:
        return [
            line.rstrip()
            for line in req_file
            if not (line.isspace() or line.startswith('#'))
        ]


setup(
    name="ploo",
    version="0.0.1",
    description="Parallel leave-one-out CV",
    long_description=long_description,
    author="Alex Cooper",
    author_email="alex@acooper.org",
    url="https://github.com/kuperov/ploo",
    license="LICENSE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["test"]),
    install_requires=_parse_requirements('requirements.txt'),
    tests_require=_parse_requirements('requirements-tests.txt'),
    include_package_data=True,
)
