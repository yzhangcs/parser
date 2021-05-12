# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='supar',
    version='1.1.1',
    author='Yu Zhang',
    author_email='yzhang.cs@outlook.com',
    description='Syntactic/Semantic Parsing Models',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yzhangcs/parser',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic'
    ],
    setup_requires=[
        'setuptools>=56.0',
    ],
    install_requires=[
        'torch>=1.7.1',
        'transformers>=4.0.0',
        'nltk',
        'stanza',
        'dill'],
    entry_points={
        'console_scripts': [
            'biaffine-dep=supar.cmds.biaffine_dep:main',
            'crf-dep=supar.cmds.crf_dep:main',
            'crf2o-dep=supar.cmds.crf2o_dep:main',
            'crf-con=supar.cmds.crf_con:main',
            'biaffine-sdp=supar.cmds.biaffine_sdp:main',
            'vi-sdp=supar.cmds.vi_sdp:main'
        ]
    },
    python_requires='>=3.7',
    zip_safe=False
)
