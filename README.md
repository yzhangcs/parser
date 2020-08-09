# SuPar

[![build](https://github.com/yzhangcs/parser/workflows/build/badge.svg)](https://github.com/yzhangcs/parser/actions)
[![docs](https://readthedocs.org/projects/parser/badge/?version=latest)](https://parser.readthedocs.io/en/latest)
[![release](https://img.shields.io/github/v/release/yzhangcs/parser)](https://yzhangcs/parser/releases)
[![downloads](https://pepy.tech/badge/supar)](https://pepy.tech/project/supar)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/parser)](https://github.com/yzhangcs/parser/blob/master/LICENSE)

`SuPar` provides a collection of state-of-the-art syntactic parsing models with Biaffine Parser ([Dozat and Manning, 2017](#dozat-2017-biaffine)) as the basic architecture:
* Biaffine Dependency Parser ([Dozat and Manning, 2017](#dozat-2017-biaffine))
* CRFNP Dependency Parser ([Koo et al., 2007](#koo-2007-structured); [Ma and Hovy, 2017](#ma-2017-neural))
* CRF Dependency Parser ([Zhang et al., 2020a](#zhang-2020-efficient))
* CRF2o Dependency Parser ([Zhang et al, 2020a](#zhang-2020-efficient))
* CRF Constituency Parser ([Zhang et al, 2020b](#zhang-2020-fast))

You can load released pretrained models for the above parsers and obtain dependency/constituency parsing trees very conveniently, as detailed in [Usage](#Usage).

The implementations of several popular and well-known algorithms, like MST (ChuLiu/Edmods), Eisner, CKY, MatrixTree, TreeCRF, are also integrated in this package.

Besides POS Tag embeddings used by the vanilla Biaffine Parser as auxiliary inputs to the encoder, optionally, `SuPar` also allows to utilize CharLSTM/BERT layers to produce character/subword-level features.
Among them, CharLSTM is taken as the default option, which avoids additional requirements for generating POS tags, as well as the inefficiency of BERT.
The BERT module in `SuPar` extracts BERT representations from the pretrained model in [`transformers`](https://github.com/huggingface/transformers).
It is also compatiable with other language models like XLNet, RoBERTa and ELECTRA, etc.

The CRF models for Dependency/Constituency parsing are our recent works published in ACL 2020 and IJCAI 2020 respectively.
If you are interested in them, please cite:
```bib
@inproceedings{zhang-etal-2020-efficient,
  title     = {Efficient Second-Order {T}ree{CRF} for Neural Dependency Parsing},
  author    = {Zhang, Yu and Li, Zhenghua and Zhang Min},
  booktitle = {Proceedings of ACL},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.acl-main.302},
  pages     = {3295--3305}
}

@inproceedings{zhang-etal-2020-fast,
  title     = {Fast and Accurate Neural {CRF} Constituency Parsing},
  author    = {Zhang, Yu and Zhou, Houquan and Li, Zhenghua},
  booktitle = {Proceedings of IJCAI},
  year      = {2020},
  doi       = {10.24963/ijcai.2020/560},
  url       = {https://doi.org/10.24963/ijcai.2020/560},
  pages     = {4046--4053}
}
```

## Contents

* [Contents](#contents)
* [Installation](#installation)
* [Performance](#performance)
* [Usage](#usage)
  * [Training](#training)
  * [Evaluation](#evaluation)
* [References](#references)

## Installation

`SuPar` can be installed via pip:
```sh
pip install -U supar
```
Or installing from source is also permitted:
```sh
git clone https://github.com/yzhangcs/parser && cd parser
python setup.py install
```

As a prerequisite, the following requirements should be satisfied:
* `python`: 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): 1.4
* [`transformers`](https://github.com/huggingface/transformers): 3.0

## Performance

Currently, `SuPar` provides pretrained models for English and Chinese.
English models are trained on Penn Treebank (PTB) with 39,832 training sentences, while Chinese models are trained on Penn Chinese Treebank version 7 (CTB7) with 46,572 training sentences.

The performance and parsing speed of these models are listed in the following table.
Notably, punctuation is ignored in all evaluation metrics for PTB, but reserved for CTB7.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th align="center">Type</th>
      <th align="center">Name</th>
      <th align="center">Metric</th>
      <th align="center" colspan=2>Performance</th>
      <th align="right">Speed (Sents/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=5>PTB</td>
      <td rowspan=4>Dependency</td>
      <td><code>biaffine-dep-en</code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.03</td><td align="center">94.37</td>
      <td align="right">1826.77</td>
    </tr>
    <tr>
      <td><code>crfnp-dep-en</code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.01</td><td align="center">94.42</td>
      <td align="right">2197.15</td>
    </tr>
    <tr>
      <td><code>crf-dep-en</code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.12</td><td align="center">94.50</td>
      <td align="right">652.41</td>
    </tr>
    <tr>
      <td><code>crf2o-dep-en</a></code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.14</td><td align="center">94.55</td>
      <td align="right">465.64</td>
    </tr>
    <tr>
      <td>Constituency</td>
      <td><code>crf-con-en</a></code></td>
      <td align="center">F<sub>1</sub></td>
      <td align="center" colspan=2>94.18</td><td align="right">923.74</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan=5>CTB7</td>
      <td rowspan=4>Dependency</td>
      <td><code>biaffine-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>88.77</td><td>85.63</td><td align="right">1155.50</td>
    </tr>
    <tr>
      <td><code>crfnp-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>88.78</td><td>85.64</td><td align="right">1323.75</td>
    </tr>
    <tr>
      <td><code>crf-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>88.98</td><td>85.84</td><td align="right">354.65</td>
    </tr>
    <tr>
      <td><code>crf2o-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>89.35</td><td>86.25</td><td align="right">217.09</td>
    </tr>
    <tr>
      <td>Constituency</td>
      <td><code>crf-con-zh</code></td>
      <td align="center">F<sub>1</sub></td>
      <td align="center" colspan=2>88.67</td>
      <td align="right">639.27</td>
    </tr>
  </tbody>
</table>

All results are tested on the machine with Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz and Nvidia GeForce GTX 1080 Ti GPU.

## Usage

`SuPar` is very easy to use. You can download the pretrained model and run syntactic parsing over sentences with a few lines of code:
```py
>>> from supar import Parser
>>> parser = Parser.load('biaffine-dep-en')
>>> dataset = parser.predict([['She', 'enjoys', 'playing', 'tennis', '.']], prob=True, verbose=False)
100%|####################################| 1/1 00:00<00:00, 85.15it/s
```
The call to `parser.predict` will return an instance of `supar.utils.Dataset` containing the predicted syntactic trees.
For dependency parsing, you can either access each sentence held in `dataset` or an individual field of all the trees.
```py
>>> print(dataset.sentences[0])
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _

>>> print(f"arcs:  {dataset.arcs[0]}\n"
          f"rels:  {dataset.rels[0]}\n"
          f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")
arcs:  [2, 0, 2, 3, 2]
rels:  ['nsubj', 'root', 'xcomp', 'dobj', 'punct']
probs: tensor([1.0000, 1.0000, 0.9650, 0.9687, 0.9996])
```
Probabilities can be returned along with the results if `prob=True`.
As for CRF parsers, marginals are available if `mbr=True`, i.e., using mbr decoding.

Note that `SuPar` requires pre-tokenized sentences as inputs.
If you'd like to parse un-tokenized raw texts, you can call `nltk.word_tokenize` to do the tokenization first:
```py
>>> import nltk
>>> text = nltk.word_tokenize('She enjoys playing tennis.')
>>> print(parser.predict([text], verbose=False).sentences[0])
100%|####################################| 1/1 00:00<00:00, 74.20it/s
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _

```

If there are a plenty of sentences to parse, `SuPar` also supports for loading them from file, and save to the `pred` file if specified.
```py
>>> dataset = parser.predict('data/ptb/test.conllx', pred='pred.conllx')
2020-07-25 18:13:50 INFO Load the data
2020-07-25 18:13:52 INFO
Dataset(n_sentences=2416, n_batches=13, n_buckets=8)
2020-07-25 18:13:52 INFO Make predictions on the dataset
100%|####################################| 13/13 00:01<00:00, 10.58it/s
2020-07-25 18:13:53 INFO Save predicted results to pred.conllx
2020-07-25 18:13:54 INFO 0:00:01.335261s elapsed, 1809.38 Sents/s
```

Please make sure the file is in CoNLL-X format. If some fields are missing, you can use underscores as placeholders.
An interface is provided for the transformation from text to CoNLL-X format string.
```py
>>> from supar.utils import CoNLL
>>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
1       She     _       _       _       _       _       _       _       _
2       enjoys  _       _       _       _       _       _       _       _
3       playing _       _       _       _       _       _       _       _
4       tennis  _       _       _       _       _       _       _       _
5       .       _       _       _       _       _       _       _       _

```

For Universial Dependencies (UD), the CoNLL-U file is also allowed, while comment lines in the file can be reserved before prediction and recovered during post-processing.
```py
>>> import os
>>> import tempfile
>>> text = '''# text = But I found the location wonderful and the neighbors very kind.
1\tBut\t_\t_\t_\t_\t_\t_\t_\t_
2\tI\t_\t_\t_\t_\t_\t_\t_\t_
3\tfound\t_\t_\t_\t_\t_\t_\t_\t_
4\tthe\t_\t_\t_\t_\t_\t_\t_\t_
5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_
6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_
7\tand\t_\t_\t_\t_\t_\t_\t_\t_
7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_
8\tthe\t_\t_\t_\t_\t_\t_\t_\t_
9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_
10\tvery\t_\t_\t_\t_\t_\t_\t_\t_
11\tkind\t_\t_\t_\t_\t_\t_\t_\t_
12\t.\t_\t_\t_\t_\t_\t_\t_\t_

'''
>>> path = os.path.join(tempfile.mkdtemp(), 'data.conllx')
>>> with open(path, 'w') as f:
...     f.write(text)
...
>>> print(parser.predict(path, verbose=False).sentences[0])
100%|####################################| 1/1 00:00<00:00, 68.60it/s
# text = But I found the location wonderful and the neighbors very kind.
1       But     _       _       _       _       3       cc      _       _
2       I       _       _       _       _       3       nsubj   _       _
3       found   _       _       _       _       0       root    _       _
4       the     _       _       _       _       5       det     _       _
5       location        _       _       _       _       6       nsubj   _       _
6       wonderful       _       _       _       _       3       xcomp   _       _
7       and     _       _       _       _       6       cc      _       _
7.1     found   _       _       _       _       _       _       _       _
8       the     _       _       _       _       9       det     _       _
9       neighbors       _       _       _       _       11      dep     _       _
10      very    _       _       _       _       11      advmod  _       _
11      kind    _       _       _       _       6       conj    _       _
12      .       _       _       _       _       3       punct   _       _

```

Constituency trees can be parsed in a similar manner.
The returned `dataset` holds all predicted trees represented using `nltk.Tree` objects.
```py
>>> parser = Parser.load('crf-con-en')
>>> dataset = parser.predict([['She', 'enjoys', 'playing', 'tennis', '.']], verbose=False)
100%|####################################| 1/1 00:00<00:00, 75.86it/s
>>> print(f"trees:\n{dataset.trees[0]}")
trees:
(TOP
  (S
    (NP (_ She))
    (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
    (_ .)))
>>> dataset = parser.predict('data/ptb/test.pid', pred='pred.pid')
2020-07-25 18:21:28 INFO Load the data
2020-07-25 18:21:33 INFO
Dataset(n_sentences=2416, n_batches=13, n_buckets=8)
2020-07-25 18:21:33 INFO Make predictions on the dataset
100%|####################################| 13/13 00:02<00:00,  5.30it/s
2020-07-25 18:21:36 INFO Save predicted results to pred.pid
2020-07-25 18:21:36 INFO 0:00:02.455740s elapsed, 983.82 Sents/s
```

Analogous to dependency parsing, a sentence can be transformed to an empty `nltk.Tree` conveniently:
```py
>>> from supar.utils import Tree
>>> print(Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], root='TOP'))
(TOP (_ She) (_ enjoys) (_ playing) (_ tennis) (_ .))
```

### Training

To train a model from scratch, it is preferred to use the command-line option, which is more flexible and customizable.
Here are some training examples:
```sh
# Biaffine Dependency Parser
# some common and default arguments are stored in config.ini
$ python -m supar.cmds.biaffine_dependency train -b -d 0  \
    -p exp/ptb.biaffine.dependency.char/model  \
    -f char  \
    -c config.ini
# to use BERT, `-f` and `--bert` (default to bert-base-cased) should be specified
# if you'd like to use XLNet, you can type `--bert xlnet-base-cased`
$ python -m supar.cmds.biaffine_dependency train -b -d 0  \
    -p exp/ptb.biaffine.dependency.bert/model  \
    -f bert  \
    --bert bert-base-cased

# CRF Dependency Parser
# for CRF dependency parsers, you should use `--proj` to discard all non-projective training instances
# optionally, you can use `--mbr` to perform mbr decoding
$ python -m supar.cmds.crf_dependency train -b -d 0  \
    -p exp/ptb.crf.dependency.char/model  \
    -f char  \
    --mbr  \
    --proj

# CRF Constituency Parser
# the training of CRF constituency parser behaves like dependency parsers
$ python -m supar.cmds.crf_constituency train -b -d 0 -p exp/ptb.crf.constituency.char/model -f char  \
    --mbr
```

For more instructions on training, please type `python -m supar.cmds.<parser> train -h`.

Alternatively, `SuPar` provides some equivalent command entry points registered in `setup.py`:
`biaffine-dependency`, `crfnp-dependency`, `crf-dependency`, `crf2o-dependency` and `crf-constituency`.
```sh
$ biaffine-dependency train -b -d 0 -p exp/ptb.biaffine.dependency.char/model -f char -c config.ini
```

To accommodate large models, distributed training is also supported:
```sh
python -m torch.distributed.launch --nproc_per_node=4 --master_port=10000  \
    -m supar.cmds.biaffine_dependency train -b -d 0,1,2,3 -p exp/ptb.biaffine.dependency.char/model -f char
```
You can consult the PyTorch [documentation](https://pytorch.org/docs/stable/notes/ddp.html) and [tutorials](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

### Evaluation

The evaluation process resembles prediction:
```py
>>> parser = Parser.load('biaffine-dep-en')
>>> loss, metric = parser.evaluate('data/ptb/test.conllx')
2020-07-25 20:59:17 INFO Load the data
2020-07-25 20:59:19 INFO
Dataset(n_sentences=2416, n_batches=11, n_buckets=8)
2020-07-25 20:59:19 INFO Evaluate the dataset
2020-07-25 20:59:20 INFO loss: 0.2326 - UCM: 61.34% LCM: 50.21% UAS: 96.03% LAS: 94.37%
2020-07-25 20:59:20 INFO 0:00:01.253601s elapsed, 1927.25 Sents/s
```

## References

* <a id="koo-2007-structured"></a>
Terry Koo, Amir Globerson, Xavier Carreras and Michael Collins. 2007. [Structured Prediction Models via the Matrix-Tree Theorem](https://www.aclweb.org/anthology/D07-1015/).
* <a id="dozat-2017-biaffine"></a>
Timothy Dozat and Christopher D. Manning. 2017. [Deep Biaffine Attention for Neural Dependency Parsing](https://openreview.net/pdf?id=Hk95PK9le).
* <a id="ma-2017-neural"></a>
Xuezhe Ma and Eduard Hovy. 2017. [Neural Probabilistic Model for Non-projective MST Parsing](https://www.aclweb.org/anthology/I17-1007/).
* <a id="zhang-2020-fast"></a>
Yu Zhang, Houquan Zhou and Zhenghua Li. 2020. [Fast and Accurate Neural CRF Constituency Parsing](https://www.ijcai.org/Proceedings/2020/560/).
* <a id="zhang-2020-efficient"></a>
Yu Zhang, Zhenghua Li and Min Zhang. 2020. [Efficient Second-Order TreeCRF for Neural Dependency Parsing](https://www.aclweb.org/anthology/2020.acl-main.302/).
