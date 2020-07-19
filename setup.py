# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='supar',
    version='0.1.0',
    author='Yu Zhang',
    author_email='yzhang.cs@outlook.com',
    description='Syntactic Parsing Models',
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
        'setuptools>=18.0',
    ],
    install_requires=['torch>=1.4.0', 'transformers', 'nltk'],
    entry_points={
        'console_scripts': [
            'biaffine-dependency=supar.cmds.biaffine_dependency:main',
            'crf-dependency=supar.cmds.crf_dependency:main',
            'crf2o-dependency=supar.cmds.crf2o_dependency:main',
            'crfnp-dependency=supar.cmds.crfnp_dependency:main',
            'crf-constituency=supar.cmds.crf_constituency:main'
        ]
    },
    python_requires='>=3.7',
    zip_safe=False
)
