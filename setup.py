#!/usr/bin/env python

import pathlib
from setuptools import setup, find_packages


HERE = pathlib.Path(__file__).parent


README = (HERE / "README.md").read_text()

packages = find_packages()
install_requires = [line.strip() for line in open('requirements.txt')]

# This call to setup() does all the work
setup(
    name="selector-ac",
    version="0.1.0.5",
    description="Selector: Ensemble-Based Automated Algorithm Configuration",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/DOTBielefeld/selector",
    author="Dimitri Weiß",
    author_email="dimitri-weiss@web.de",
    packages=find_packages(exclude=["*wrapper*", "*test"]),
    license="MIT License",
    python_requires=">=3.8",
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",  
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    install_requires=install_requires,
    entry_points={},
)
