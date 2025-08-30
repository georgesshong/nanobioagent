from setuptools import setup, find_packages
import os

# Read README for long description  
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nanobioagent",
    version="1.0.0",
    author="George Hong",
    author_email="gehong@ethz.ch",
    description="Nano-Scale Language Model Agents for Bioinformatics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/georgesshong/nanobioagent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "nanobioagent": [
            "config/*.json",
            "config/tools/*.json",
            "config/examples/*.json"
        ],
    },
    entry_points={
        "console_scripts": [
            "nanobioagent=main:cli_main",  # Changed from main to cli_main
        ],
    },
    keywords="bioinformatics, llm, slm, agents, genomics, ncbi, nano-scale, genegpt",
)