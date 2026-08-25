"""
Setup script for DQM Anomaly Detection Pipeline
"""

from setuptools import setup, find_packages

setup(
    name="dqm_pipeline",
    version="0.1.0",
    description="DQM anomaly detection pipeline for Mu2e CRV detector",
    author="Sophie",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0",
        "numpy>=1.19",
        "scikit-learn>=0.24",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "dqm_pipeline=dqm_pipeline.run_pipeline:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
