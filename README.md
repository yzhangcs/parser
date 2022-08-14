# SuPar

[![build](https://github.com/yzhangcs/parser/workflows/build/badge.svg)](https://github.com/yzhangcs/parser/actions)
[![docs](https://readthedocs.org/projects/parser/badge/?version=latest)](https://parser.readthedocs.io/en/latest)
[![release](https://img.shields.io/github/v/release/yzhangcs/parser)](https://github.com/yzhangcs/parser/releases)
[![downloads](https://img.shields.io/github/downloads/yzhangcs/parser/total)](https://pypistats.org/packages/supar)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/parser)](https://github.com/yzhangcs/parser/blob/master/LICENSE)

A Python package designed for structured prediction, including reproductions of many state-of-the-art syntactic/semantic parsers (with pretrained models for more than 19 languages),

* Dependency Parser
  * Biaffine ([Dozat and Manning, 2017](https://openreview.net/forum?id=Hk95PK9le))
  * CRF/CRF2o ([Zhang et al., 2020a](https://aclanthology.org/2020.acl-main.302))
* Constituency Parser
  * CRF ([Zhang et al., 2020b](https://www.ijcai.org/Proceedings/2020/560/))
  * AttachJuxtapose ([Yang and Deng, 2020](https://papers.nips.cc/paper/2020/hash/f7177163c833dff4b38fc8d2872f1ec6-Abstract.html))
* Semantic Dependency Parser
  * Biaffine ([Dozat and Manning, 2018](https://aclanthology.org/P18-2077))
  * MFVI/LBP ([Wang et al, 2019](https://aclanthology.org/P18-2077))

and highly-parallelized implementations of several well-known structured prediction algorithms.[^1]

* Chain:
  * LinearChainCRF ([Lafferty et al., 2001](http://www.aladdin.cs.cmu.edu/papers/pdfs/y2001/crf.pdf))
* Tree
  * MatrixTree ([Koo et al., 2007](https://www.aclweb.org/anthology/D07-1015); [Ma and Hovy, 2017](https://aclanthology.org/I17-1007))
  * DependencyCRF ([Eisner et al., 2000](https://www.cs.jhu.edu/~jason/papers/eisner.iwptbook00.pdf); [Zhang et al., 2020](https://aclanthology.org/2020.acl-main.302))
  * Dependency2oCRF ([McDonald et al., 2006](https://www.aclweb.org/anthology/E06-1011); [Zhang et al., 2020](https://aclanthology.org/2020.acl-main.302))
  * ConstituencyCRF ([Stern et al. 2017](https://aclanthology.org/P17-1076); [Zhang et al., 2020b](https://www.ijcai.org/Proceedings/2020/560/))
  * BiLexicalizedConstituencyCRF ([Eisner et al. 1999](https://aclanthology.org/P99-1059/); [Yang et al., 2021](https://aclanthology.org/2021.acl-long.209/))

## Installation

`SuPar` can be installed via pip:
```sh
$ pip install -U supar
```
Or installing from source is also permitted:
```sh
$ pip install -U git+https://github.com/yzhangcs/parser
```

As a prerequisite, the following requirements should be satisfied:
* `python`: >= 3.8
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.8
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Usage

You can download the pretrained model and parse sentences with just a few lines of code:
```py
>>> from supar import Parser
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
>>> parser = Parser.load('biaffine-dep-en')
>>> dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
```
By default, we use [`stanza`](https://github.com/stanfordnlp/stanza) internally to tokenize plain texts for parsing.
You only need to specify the language code `lang` for tokenization.

The call to `parser.predict` will return an instance of `supar.utils.Dataset` containing the predicted results.
You can either access each sentence held in `dataset` or an individual field of all results.
Probabilities can be returned along with the results if `prob=True`.
```py
>>> dataset[0]
1       I       _       _       _       _       2       nsubj   _       _
2       saw     _       _       _       _       0       root    _       _
3       Sarah   _       _       _       _       2       dobj    _       _
4       with    _       _       _       _       2       prep    _       _
5       a       _       _       _       _       6       det     _       _
6       telescope       _       _       _       _       4       pobj    _       _
7       .       _       _       _       _       2       punct   _       _

>>> print(f"arcs:  {dataset.arcs[0]}\n"
          f"rels:  {dataset.rels[0]}\n"
          f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")
arcs:  [2, 0, 2, 2, 6, 4, 2]
rels:  ['nsubj', 'root', 'dobj', 'prep', 'det', 'pobj', 'punct']
probs: tensor([1.0000, 0.9999, 0.9966, 0.8944, 1.0000, 1.0000, 0.9999])
```

`SuPar` also supports parsing from tokenized sentences or from file.
For BiLSTM-based semantic dependency parsing models, lemmas and POS tags are needed.

```py
>>> import os
>>> import tempfile
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
>>> dep = Parser.load('biaffine-dep-en')
>>> dep.predict(['I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.'], verbose=False)[0]
1       I       _       _       _       _       2       nsubj   _       _
2       saw     _       _       _       _       0       root    _       _
3       Sarah   _       _       _       _       2       dobj    _       _
4       with    _       _       _       _       2       prep    _       _
5       a       _       _       _       _       6       det     _       _
6       telescope       _       _       _       _       4       pobj    _       _
7       .       _       _       _       _       2       punct   _       _

>>> path = os.path.join(tempfile.mkdtemp(), 'data.conllx')
>>> with open(path, 'w') as f:
...     f.write('''# text = But I found the location wonderful and the neighbors very kind.
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

''')
...
>>> dep.predict(path, pred='pred.conllx', verbose=False)[0]
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

>>> con = Parser.load('crf-con-en')
>>> con.predict(['I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.'], verbose=False)[0].pretty_print()
              TOP                       
               |                         
               S                        
  _____________|______________________   
 |             VP                     | 
 |    _________|____                  |  
 |   |    |         PP                | 
 |   |    |     ____|___              |  
 NP  |    NP   |        NP            | 
 |   |    |    |     ___|______       |  
 _   _    _    _    _          _      _ 
 |   |    |    |    |          |      |  
 I  saw Sarah with  a      telescope  . 

>>> sdp = Parser.load('biaffine-sdp-en')
>>> sdp.predict([[('I','I','PRP'), ('saw','see','VBD'), ('Sarah','Sarah','NNP'), ('with','with','IN'),
                  ('a','a','DT'), ('telescope','telescope','NN'), ('.','_','.')]],
                verbose=False)[0]
1       I       I       PRP     _       _       _       _       2:ARG1  _
2       saw     see     VBD     _       _       _       _       0:root|4:ARG1   _
3       Sarah   Sarah   NNP     _       _       _       _       2:ARG2  _
4       with    with    IN      _       _       _       _       _       _
5       a       a       DT      _       _       _       _       _       _
6       telescope       telescope       NN      _       _       _       _       4:ARG2|5:BV     _
7       .       _       .       _       _       _       _       _       _

```

### Training

To train a model from scratch, it is preferred to use the command-line option, which is more flexible and customizable.
Below is an example of training Biaffine Dependency Parser:
```sh
$ python -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-en -p model -f char
```

Alternatively, `SuPar` provides some equivalent command entry points registered in [`setup.py`](setup.py):
`biaffine-dep`, `crf2o-dep`, `crf-con` and `biaffine-sdp`, etc.
```sh
$ biaffine-dep train -b -d 0 -c biaffine-dep-en -p model -f char
```

To accommodate large models, distributed training is also supported:
```sh
$ python -m supar.cmds.biaffine_dep train -b -c biaffine-dep-en -d 0,1,2,3 -p model -f char
```
You can consult the PyTorch [documentation](https://pytorch.org/docs/stable/notes/ddp.html) and [tutorials](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

### Evaluation

The evaluation process resembles prediction:
```py
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
>>> Parser.load('biaffine-dep-en').evaluate('ptb/test.conllx', verbose=False)
loss: 0.2393 - UCM: 60.51% LCM: 50.37% UAS: 96.01% LAS: 94.41%
```

See [EXAMPLES](EXAMPLES.md) for more instructions on training and evaluation.

## Performance

`SuPar` provides pretrained models for English, Chinese and 17 other languages.
The tables below list the performance and parsing speed of pretrained models for different tasks.
All results are tested on the machine with Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz and Nvidia GeForce GTX 1080 Ti GPU.

### Dependency Parsing

English and Chinese dependency parsing models are trained on PTB and CTB7 respectively.
For each parser, we provide pretrained models that take BiLSTM as encoder.
We also provide models trained by finetuning pretrained language models from [Huggingface Transformers](https://github.com/huggingface/transformers).
We use [`robert-large`](https://huggingface.co/roberta-large) for English and [`hfl/chinese-electra-180g-large-discriminator`](https://huggingface.co/hfl/chinese-electra-180g-large-discriminator) for Chinese.
During evaluation, punctuation is ignored in all metrics for PTB.

| Name                      |  UAS  |   LAS | Sents/s |
| ------------------------- | :---: | ----: | :-----: |
| `biaffine-dep-en`         | 96.01 | 94.41 | 1831.91 |
| `crf2o-dep-en`            | 96.07 | 94.51 | 531.59  |
| `biaffine-dep-roberta-en` | 97.33 | 95.86 | 271.80  |
| `biaffine-dep-zh`         | 88.64 | 85.47 | 1180.57 |
| `crf2o-dep-zh`            | 89.22 | 86.15 | 237.40  |
| `biaffine-dep-electra-zh` | 92.45 | 89.55 | 160.56  |

The multilingual dependency parsing model, named `biaffine-dep-xlmr`, is trained on merged 12 selected treebanks from Universal Dependencies (UD) v2.3 dataset by finetuning [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large).
The following table lists results of each treebank.
Languages are represented by [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

| Language |  UAS  |  LAS  | Sents/s |
| -------- | :---: | :---: | ------: |
| `bg`     | 96.95 | 94.24 |  343.96 |
| `ca`     | 95.57 | 94.20 |  184.88 |
| `cs`     | 95.79 | 93.83 |  245.68 |
| `de`     | 89.74 | 85.59 |  283.53 |
| `en`     | 93.37 | 91.27 |  269.16 |
| `es`     | 94.78 | 93.29 |  192.00 |
| `fr`     | 94.56 | 91.90 |  219.35 |
| `it`     | 96.29 | 94.47 |  254.82 |
| `nl`     | 96.04 | 93.76 |  268.57 |
| `no`     | 95.64 | 94.45 |  318.00 |
| `ro`     | 94.59 | 89.79 |  216.45 |
| `ru`     | 96.37 | 95.24 |  243.56 |

### Constituency Parsing

We use PTB and CTB7 datasets to train English and Chinese constituency parsing models.
Below are the results.

| Name                 |   P   |   R   | F<sub>1 | Sents/s |
| -------------------- | :---: | :---: | :-----: | ------: |
| `crf-con-en`         | 94.16 | 93.98 |  94.07  |  841.88 |
| `crf-con-roberta-en` | 96.42 | 96.13 |  96.28  |  233.34 |
| `crf-con-zh`         | 88.82 | 88.42 |  88.62  |  590.05 |
| `crf-con-electra-zh` | 92.18 | 91.66 |  91.92  |  140.45 |

The multilingual model `crf-con-xlmr` is trained on SPMRL dataset by finetuning [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large).
We follow instructions of [Benepar](https://github.com/nikitakit/self-attentive-parser) to preprocess the data.
For simplicity, we then directly merge train/dev/test treebanks of all languages in SPMRL into big ones to train the model.
The results of each treebank are as follows.

| Language |   P   |   R   | F<sub>1 | Sents/s |
| -------- | :---: | :---: | :-----: | ------: |
| `eu`     | 93.40 | 94.19 |  93.79  |  266.96 |
| `fr`     | 88.77 | 88.84 |  88.81  |  149.34 |
| `de`     | 93.68 | 92.18 |  92.92  |  200.31 |
| `he`     | 94.65 | 95.20 |  94.93  |  172.50 |
| `hu`     | 96.70 | 96.81 |  96.76  |  186.58 |
| `ko`     | 91.75 | 92.46 |  92.11  |  234.86 |
| `pl`     | 97.33 | 97.27 |  97.30  |  310.86 |
| `sv`     | 92.51 | 92.50 |  92.50  |  235.49 |

### Semantic Dependency Parsing

English semantic dependency parsing models are trained on [DM data introduced in SemEval-2014 task 8](https://catalog.ldc.upenn.edu/LDC2016T10), while Chinese models are trained on [NEWS domain data of corpora from SemEval-2016 Task 9](https://github.com/HIT-SCIR/SemEval-2016).
Our data preprocessing steps follow [Second_Order_SDP](https://github.com/wangxinyu0922/Second_Order_SDP).

| Name                |   P   |   R   | F<sub>1 | Sents/s |
| ------------------- | :---: | :---: | :-----: | ------: |
| `biaffine-sdp-en`   | 94.35 | 93.12 |  93.73  | 1067.06 |
| `vi-sdp-en`         | 94.36 | 93.52 |  93.94  |  821.73 |
| `vi-sdp-roberta-en` | 95.18 | 95.20 |  95.19  |  264.13 |
| `biaffine-sdp-zh`   | 72.93 | 66.29 |  69.45  |  523.36 |
| `vi-sdp-zh`         | 72.05 | 67.97 |  69.95  |  411.94 |
| `vi-sdp-electra-zh` | 73.29 | 70.53 |  71.89  |  139.52 |

## Citation

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

[^1]: The implementations of structured distributions and semirings are heavily borrowed from [torchstruct](https://github.com/harvardnlp/pytorch-struct) with some tailoring.
