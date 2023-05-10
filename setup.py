# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='supar',
    version='1.1.4',
    author='Yu Zhang',
    author_email='yzhang.cs@outlook.com',
    license='MIT',
    description='State-of-the-art parsers for natural language',
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
        'setuptools',
    ],
    install_requires=[
        'numpy>1.21.6',
        'torch>=1.13.1',
        'transformers>=4.0.0',
        'hydra-core>=1.2',
        'nltk',
        'stanza',
        'omegaconf',
        'dill',
        'pathos',
        'opt_einsum'
    ],
    extras_require={
        'elmo': ['allennlp'],
        'bpe': ['subword-nmt']
    },
    entry_points={
        'console_scripts': [
            'dep-biaffine=supar.cmds.dep.biaffine:main',
            'dep-crf=supar.cmds.dep.crf:main',
            'dep-crf2o=supar.cmds.dep.crf2o:main',
            'con-aj=supar.cmds.const.aj:main',
            'con-crf=supar.cmds.const.crf:main',
            'con-tt=supar.cmds.const.tt:main',
            'sdp-biaffine=supar.cmds.sdp.biaffine:main',
            'sdp-vi=supar.cmds.sdp.vi:main'
        ]
    },
    python_requires='>=3.7',
    zip_safe=False
)
