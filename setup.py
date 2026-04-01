from setuptools import find_packages, setup

setup(
    name="nsEVDx",
    version="0.1.1",
    author="Nischal Kafle",
    description="Modeling Non-stationary Extreme Value Distributions",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
)
