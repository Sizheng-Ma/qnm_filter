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
        python_requires=">=3.9,<3.12",  # lalsuite compatibility
        install_requires=[
            "h5py",
            "numpy<2.0",  # qnm compatibility
            "scipy<1.15.0",
            "astropy<6.0.0",
            "qnm",
            "lalsuite",
            "pandas",
            "matplotlib",
            "joblib",
            "bilby",
            "sxs<2025.0.17",
            "joblib",
            "lalsuite",
            "seaborn",
        ],
    )
