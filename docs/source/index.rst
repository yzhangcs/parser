.. SuPar documentation master file, created by
   sphinx-quickstart on Sun Jul 26 00:02:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SuPar
================================================================

.. image:: https://github.com/yzhangcs/parser/workflows/build/badge.svg
   :alt: build
   :target: https://github.com/yzhangcs/parser/actions
.. image:: https://readthedocs.org/projects/parser/badge/?version=latest
   :alt: docs
   :target: https://parser.readthedocs.io/en/latest
.. image:: https://img.shields.io/pypi/v/supar
   :alt: release
   :target: https://github.com/yzhangcs/parser/releases
.. image:: https://img.shields.io/github/downloads/yzhangcs/parser/total
   :alt: downloads
   :target: https://pypistats.org/packages/supar
.. image:: https://img.shields.io/github/license/yzhangcs/parser
   :alt: LICENSE
   :target: https://github.com/yzhangcs/parser/blob/master/LICENSE

A Python package designed for structured prediction, including reproductions of many state-of-the-art syntactic/semantic parsers (with pretrained models for more than 19 languages), and highly-parallelized implementations of several well-known structured prediction algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Content

   self
   parsers/index
   models/index
   structs/index
   modules/index
   utils/index
   refs

Indices and tables
================================================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledge
================================================================

The implementations of structured distributions and semirings are heavily borrowed from torchstruct_ with some tailoring.

.. _torchstruct: https://github.com/harvardnlp/pytorch-struct