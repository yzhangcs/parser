# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="supar",
    version="0.1.0",
    author="Yu Zhang",
    author_email="yzhang.cs@outlook.com",
    description="Syntactic Parsing Models",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yzhangcs/parser",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic"
    ],
    setup_requires=[
        'setuptools>=18.0',
    ],
    install_requires=["torch", "transformers"],
    python_requires='>=3.7',
    zip_safe=False
)
