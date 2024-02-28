#!/usr/bin/env python

long_description = """\
To be added
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    from setuptools import setup, find_packages

    setup(
        name="qnm_filter",
        version="0.1",
        description="QNM filters for black hole ringdowns.",
        long_description=long_description,
        url="https://github.com/Sizheng-Ma/qnm_filter",
        author="Sizheng Ma, Neil Lu",
        author_email="sma@caltech.edu, Neil.Lu@anu.edu.au",
        packages=find_packages(),
        install_requires=[
            "h5py",
            "numpy",
            "qnm",
            "lalsuite",
            "matplotlib",
            "joblib",
            "bilby",
            "sxs",
        ],  # TODO: change the order
    )
