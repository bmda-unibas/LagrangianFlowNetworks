from os import path
from setuptools import find_packages, setup
from enflows.version import VERSION

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="enflows",
    version=1,
    description="Normalizing flows in PyTorch. An extension of nflows.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
        "umnn"
    ],
    dependency_links=[],
)
