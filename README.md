# SuPar

[![Travis](https://img.shields.io/travis/yzhangcs/parser.svg)](https://travis-ci.org/yzhangcs/parser)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/parser.svg)](https://github.com/yzhangcs/parser/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yzhangcs/parser.svg)](https://github.com/yzhangcs/parser/stargazers)		
[![GitHub forks](https://img.shields.io/github/forks/yzhangcs/parser.svg)](https://github.com/yzhangcs/parser/network/members)

An implementation of "Deep Biaffine Attention for Neural Dependency Parsing".

Details and [hyperparameter choices](#Hyperparameters) are almost identical to those described in the paper, 
except that we provide the Eisner rather than MST algorithm to ensure well-formedness. 
Practically, projective decoding like Eisner is the best choice since PTB contains mostly (99.9%) projective trees.

Besides the basic implementations, we also provide other features to replace the POS tags (TAG), 
e.g., character-level embeddings (CHAR) and BERT.

## Requirements

* `python`: 3.7.0
* [`pytorch`](https://github.com/pytorch/pytorch): 1.4.0
* [`transformers`](https://github.com/huggingface/transformers): 3.0.0

## Datasets

The model is evaluated on the Stanford Dependency conversion ([v3.3.0](https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip)) of the English Penn Treebank with POS tags predicted by [Stanford POS tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip).

For all datasets, we follow the conventional data splits.

## Performance

<table>
  <thead>
    <tr>
      <th rowspan=2>Dataset</th>
      <th rowspan=2>Parser</th>
      <th colspan=2 align="center">Performance</th>
      <th rowspan=2 align="right">Speed (Sents/s)</th>
    </tr>
    <tr>
      <th align="center">UAS</th>
      <th align="center">LAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=4>PTB</td>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/biaffine_dependency.py">Biaffine</a></code></td>
      <td align="center">96.03</td><td align="center">94.37</td><td align="right">1826.77</td>
    </tr>
    <tr>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crfnp_dependency.py">CRFNP</a></code></td>
      <td align="center">96.01</td><td align="center">94.42</td><td align="right">2197.15</td>
    </tr>
    <tr>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crf_dependency.py">CRF</a></code></td>
      <td align="center">96.12</td><td align="center">94.50</td><td align="right">652.41</td>
    </tr>
    <tr>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crf2o_dependency.py">CRF2o</a></code></td>
      <td align="center">96.14</td><td align="center">94.55</td><td align="right">465.64</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan=4>CTB7</td>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/biaffine_dependency.py">Biaffine</a></code></td>
      <td>88.77</td><td>85.63</td><td align="right">1155.50</td>
    </tr>
    <tr>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crfnp_dependency.py">CRFNP</a></code></td>
      <td>88.78</td><td>85.64</td><td align="right">1323.75</td>
    </tr>
    <tr>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crf_dependency.py">CRF</a></code></td>
      <td>88.98</td><td>85.84</td><td align="right">354.65</td>
    </tr>
    <tr>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crf2o_dependency.py">CRF2o</a></code></td>
      <td>89.35</td><td>86.25</td><td align="right">217.09</td>
    </tr>
  </tbody>
</table>


<table>
  <thead>
    <tr>
      <th rowspan=2>Dataset</th>
      <th rowspan=2>Parser</th>
      <th align="center">Performance</th>
      <th rowspan=2 align="right">Speed (Sents/s)</th>
    </tr>
    <tr>
      <th align="center">F<sub>1</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=4>PTB</td>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crf_constituency.py">CRF</a></code></td>
      <td align="center">94.18</td>
      <td align="right">923.74</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan=4>CTB7</td>
      <td><code><a href="https://github.com/yzhangcs/parser/blob/release/supar/parsers/crf_constituency.py">CRF</a></code></td>
      <td align="center">88.67</td>
      <td align="right">639.27</td>
    </tr>
    </tr>
  </tbody>
</table>

Note that punctuation is ignored in all evaluation metrics for PTB. 

Aside from using consistent hyperparameters, there are some keypoints that significantly affect the performance:

- Dividing the pretrained embedding by its standard-deviation
- Applying the same dropout mask at every recurrent timestep
- Jointly dropping the word and additional feature representations

For the above reasons, we may have to give up some native modules in pytorch (e.g., `LSTM` and `Dropout`), 
and use custom ones instead.

As shown above, our results have outperformed the [offical implementation](https://github.com/tdozat/Parser-v1) (95.74 and 94.08). 
Incorporating character-level features or external embeddings like BERT can further improve the performance of the model. 

## Usage

You can start the training, evaluation and prediction process by using subcommands registered in `parser.cmds`.

```sh
$ python run.py -h
usage: run.py [-h] {evaluate,predict,train} ...

Create the Biaffine Parser model.

optional arguments:
  -h, --help            show this help message and exit

Commands:
  {evaluate,predict,train}
    evaluate            Evaluate the specified parser and dataset.
    predict             Use a trained parser to make predictions.
    train               Train a parser.
```

Before triggering the subcommands, please make sure that the data files must be in CoNLL-X format. 
If some fields are missing, you can use underscores as placeholders.
Below are some examples:

```sh
$ python run.py train -d=0 -p=exp/ptb.char --feat=char  \
      --train=data/ptb/train.conllx  \
      --dev=data/ptb/dev.conllx  \
      --test=data/ptb/test.conllx  \
      --embed=data/glove.6B.100d.txt  \
      --unk=unk

$ python run.py evaluate -d=0 -f=exp/ptb.char --feat=char --tree  \
      --data=data/ptb/test.conllx

$ cat data/naive.conllx 
1       Too     _       _       _       _       _       _       _       _
2       young   _       _       _       _       _       _       _       _
3       too     _       _       _       _       _       _       _       _
4       simple  _       _       _       _       _       _       _       _
5       ,       _       _       _       _       _       _       _       _
6       sometimes       _       _       _       _       _       _       _       _
7       naive   _       _       _       _       _       _       _       _
8       .       _       _       _       _       _       _       _       _

$ python run.py predict -d=0 -p=exp/ptb.char --feat=char --tree  \
      --data=data/naive.conllx  \
      --pred=naive.conllx

# support for outputting the probabilities of predicted arcs, triggered by `--prob`
$ cat naive.conllx
1	Too	_	_	_	_	2	advmod	0.8894	_
2	young	_	_	_	_	0	root	0.9322	_
3	too	_	_	_	_	4	advmod	0.8722	_
4	simple	_	_	_	_	2	dep	0.8948	_
5	,	_	_	_	_	2	punct	0.8664	_
6	sometimes	_	_	_	_	7	advmod	0.8406	_
7	naive	_	_	_	_	2	dep	0.971	_
8	.	_	_	_	_	2	punct	0.9741	_

```

All the optional arguments of the subcommands are as follows:

```sh
$ python run.py train -h
usage: run.py train [-h] [--path PATH] [--device DEVICE]
                    [--seed SEED] [--threads THREADS]
                    [--batch-size BATCH_SIZE] [--buckets BUCKETS] [--partial]
                    [--mbr] [--tree] [--proj] [--feat {tag,char,bert}]
                    [--build] [--punct] [--max-len MAX_LEN] [--train TRAIN]
                    [--dev DEV] [--test TEST] [--embed EMBED] [--unk UNK]
                    [--bert BERT_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  path to model file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --batch-size BATCH_SIZE
                        batch size
  --buckets BUCKETS     max num of buckets to use
  --partial             whether partial annotation is included
  --mbr                 whether to use mbr decoding
  --tree                whether to ensure well-formedness
  --proj                whether to projectivise the data
  --feat {tag,char,bert}, -f {tag,char,bert}
                        choices of additional features
  --build, -b           whether to build the model first
  --punct               whether to include punctuation
  --max-len MAX_LEN     max length of the sentences
  --train TRAIN         path to train file
  --dev DEV             path to dev file
  --test TEST           path to test file
  --embed EMBED         path to pretrained embeddings
  --unk UNK             unk token in pretrained embeddings
  --bert BERT_MODEL
                        which bert model to use

$ python run.py evaluate -h
usage: run.py evaluate [-h] [--batch-size BATCH_SIZE] [--buckets BUCKETS]
                       [--punct] [--fdata FDATA] [--file FILE]
                       [--preprocess] [--device DEVICE] [--seed SEED]
                       [--threads THREADS] [--tree] [--feat {tag,char,bert}]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --buckets BUCKETS     max num of buckets to use
  --punct               whether to include punctuation
  --fdata FDATA         path to dataset
  --file FILE, -f FILE  path to saved files
  --preprocess, -p      whether to preprocess the data first
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --tree                whether to ensure well-formedness
  --feat {tag,char,bert}
                        choices of additional features

$ python run.py predict -h
usage: run.py predict [-h] [--path PATH] [--device DEVICE]
                      [--seed SEED] [--threads THREADS]
                      [--batch-size BATCH_SIZE] [--buckets BUCKETS]
                      [--partial] [--mbr] [--tree] [--proj] [--prob]
                      [--data DATA] [--pred PRED]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  path to model file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --batch-size BATCH_SIZE
                        batch size
  --buckets BUCKETS     max num of buckets to use
  --partial             whether partial annotation is included
  --mbr                 whether to use mbr decoding
  --tree                whether to ensure well-formedness
  --proj                whether to projectivise the data
  --prob                whether to output probs
  --data DATA           path to dataset
  --pred PRED           path to predicted result
```

## Hyperparameters

| Param         | Description                                                  |                                 Value                                  |
| :------------ | :----------------------------------------------------------- | :--------------------------------------------------------------------: |
| n_embed       | dimension of embeddings                                      |                                  100                                   |
| n_char_embed  | dimension of char embeddings                                 |                                   50                                   |
| n_bert_layers | number of bert layers to use                                 |                                   4                                    |
| embed_dropout | dropout ratio of embeddings                                  |                                  0.33                                  |
| n_lstm_hidden | dimension of lstm hidden states                              |                                  400                                   |
| n_lstm_layers | number of lstm layers                                        |                                   3                                    |
| lstm_dropout  | dropout ratio of lstm                                        |                                  0.33                                  |
| n_mlp_arc     | arc mlp size                                                 |                                  500                                   |
| n_mlp_rel     | label mlp size                                               |                                  100                                   |
| mlp_dropout   | dropout ratio of mlp                                         |                                  0.33                                  |
| lr            | starting learning rate of training                           |                                  2e-3                                  |
| betas         | hyperparameters of momentum and L2 norm                      |                               (0.9, 0.9)                               |
| epsilon       | stability constant                                           |                                 1e-12                                  |
| annealing     | formula of learning rate annealing                           | <img src="https://latex.codecogs.com/gif.latex?.75^{\frac{t}{5000}}"/> |
| batch_size    | approximate number of tokens per training update             |                                  5000                                  |
| epochs        | max number of epochs                                         |                                 50000                                  |
| patience      | patience for early stop                                      |                                  100                                   |
| min_freq      | minimum frequency of words in the training set not discarded |                                   2                                    |
| fix_len       | fixed length of a word                                       |                                   20                                   |

## References

* [Deep Biaffine Attention for Neural Dependency Parsing](https://openreview.net/pdf?id=Hk95PK9le)
<!-- * [Stack-Pointer Networks for Dependency Parsing](https://www.aclweb.org/anthology/P18-1130.pdf) -->