from setuptools import setup, find_packages

setup(
    name="nsEVDx",
    version="0.1.0",
    author="Nischal Kafle",
    description="Modeling Non-stationary Extreme Value Distributions using Bayesian and Frequentist Approach",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "tqdm",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology"
    ]
)
