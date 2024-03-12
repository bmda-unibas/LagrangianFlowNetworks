from os import path
from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="lflows",
    description="Lagrangian Flow Networks.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
        "umnn",
        "enflows"
    ],
)
