#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "networkx",
    "numpy",
    "pandas",
    "plotnine",
    "scikit-learn",
    "scipy",
    "pygraphviz",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

setup(
    author="Ben Lindsay",
    author_email="benjlindsay@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="PBCluster simplifies cluster analysis of particles in boxes with periodic boundary conditions.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pbcluster",
    name="pbcluster",
    packages=find_packages(include=["pbcluster"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/benlindsay/pbcluster",
    version="0.1.0",
    zip_safe=False,
)
