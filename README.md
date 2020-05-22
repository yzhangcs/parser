# Biaffine Parser

[![Travis](https://img.shields.io/travis/yzhangcs/biaffine-parser.svg)](https://travis-ci.org/yzhangcs/biaffine-parser)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/biaffine-parser.svg)](https://github.com/yzhangcs/biaffine-parser/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yzhangcs/biaffine-parser.svg)](https://github.com/yzhangcs/biaffine-parser/stargazers)		
[![GitHub forks](https://img.shields.io/github/forks/yzhangcs/biaffine-parser.svg)](https://github.com/yzhangcs/biaffine-parser/network/members)

An implementation of "Deep Biaffine Attention for Neural Dependency Parsing".

Details and [hyperparameter choices](#Hyperparameters) are almost identical to those described in the paper, 
except that we provide the Eisner rather than MST algorithm to ensure well-formedness. 
Practically, projective decoding like Eisner is the best choice since PTB contains mostly (99.9%) projective trees.

Besides the basic implementations, we also provide other features to replace the POS tags (TAG), 
e.g., character-level embeddings (CHAR) and BERT.

## Requirements

* `python`: 3.7.0
* [`pytorch`](https://github.com/pytorch/pytorch): 1.3.0
* [`transformers`](https://github.com/huggingface/transformers): 2.1.1

## Datasets

The model is evaluated on the Stanford Dependency conversion ([v3.3.0](https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip)) of the English Penn Treebank with POS tags predicted by [Stanford POS tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip).

For all datasets, we follow the conventional data splits:

* Train: 02-21 (39,832 sentences)
* Dev: 22 (1,700 sentences)
* Test: 23 (2,416 sentences)

## Performance

<table>
  <thead>
    <tr>
      <th rowspan=2>Model</th>
      <th rowspan=2>FEAT</th>
      <th colspan=2 style="text-align:center">Performance</th>
      <th rowspan=2 style="text-align:right">Speed (Sents/s)</th>
    </tr>
    <tr>
      <th style="text-align:center">UAS</th>
      <th style="text-align:center">LAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=3>Biaffine Parser</td>
      <td>TAG</td>
      <td>95.83</td><td>94.14</td><td style="text-align:right">1340.87</td>
    </tr>
    <tr>
      <td>CHAR</td>
      <td>96.06</td><td>94.46</td><td style="text-align:right">1073.13</td>
    </tr>
    <tr>
      <td>BERT</td>
      <td>96.64</td><td>95.11</td><td style="text-align:right">438.72</td>
    </tr>
    <tr>
      <td>Stack Pointer</td>
      <td>CHAR</td>
      <td style="color:white">00.00</td><td style="color:white">00.00</td><td style="text-align:right;color:white">0</td>
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
    evaluate            Evaluate the specified model and dataset.
    predict             Use a trained model to make predictions.
    train               Train a model.
```

Before triggering the subcommands, please make sure that the data files must be in CoNLL-X format. 
If some fields are missing, you can use underscores as placeholders.
Below are some examples:

```sh
$ python run.py train -p -d=0 -f=exp/ptb.char --feat=char  \
      --ftrain=data/ptb/train.conllx  \
      --fdev=data/ptb/dev.conllx  \
      --ftest=data/ptb/test.conllx  \
      --fembed=data/glove.6B.100d.txt  \
      --unk=unk

$ python run.py evaluate -d=0 -f=exp/ptb.char --feat=char --tree  \
      --fdata=data/ptb/test.conllx

$ cat data/naive.conllx 
1       Too     _       _       _       _       _       _       _       _
2       young   _       _       _       _       _       _       _       _
3       too     _       _       _       _       _       _       _       _
4       simple  _       _       _       _       _       _       _       _
5       ,       _       _       _       _       _       _       _       _
6       sometimes       _       _       _       _       _       _       _       _
7       naive   _       _       _       _       _       _       _       _
8       .       _       _       _       _       _       _       _       _

$ python run.py predict -d=0 -f=exp/ptb.char --feat=char --tree  \
      --fdata=data/naive.conllx  \
      --fpred=naive.conllx

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
usage: run.py train [-h] [--buckets BUCKETS] [--punct] [--ftrain FTRAIN]
                    [--fdev FDEV] [--ftest FTEST] [--fembed FEMBED]
                    [--unk UNK] [--conf CONF] [--file FILE] [--preprocess]
                    [--device DEVICE] [--seed SEED] [--threads THREADS]
                    [--tree] [--feat {tag,char,bert}]

optional arguments:
  -h, --help            show this help message and exit
  --buckets BUCKETS     max num of buckets to use
  --punct               whether to include punctuation
  --ftrain FTRAIN       path to train file
  --fdev FDEV           path to dev file
  --ftest FTEST         path to test file
  --fembed FEMBED       path to pretrained embeddings
  --unk UNK             unk token in pretrained embeddings
  --conf CONF, -c CONF  path to config file
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

$ python run.py evaluate -h
usage: run.py evaluate [-h] [--batch-size BATCH_SIZE] [--buckets BUCKETS]
                       [--punct] [--fdata FDATA] [--conf CONF] [--file FILE]
                       [--preprocess] [--device DEVICE] [--seed SEED]
                       [--threads THREADS] [--tree] [--feat {tag,char,bert}]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --buckets BUCKETS     max num of buckets to use
  --punct               whether to include punctuation
  --fdata FDATA         path to dataset
  --conf CONF, -c CONF  path to config file
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
usage: run.py predict [-h] [--batch-size BATCH_SIZE] [--fdata FDATA]
                      [--fpred FPRED] [--conf CONF] [--file FILE]
                      [--preprocess] [--device DEVICE] [--seed SEED]
                      [--threads THREADS] [--tree] [--feat {tag,char,bert}]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --fdata FDATA         path to dataset
  --fpred FPRED         path to predicted result
  --conf CONF, -c CONF  path to config file
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

* [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
