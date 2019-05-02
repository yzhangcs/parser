# Biaffine Parser

[![Travis](https://img.shields.io/travis/zysite/biaffine-parser.svg)](https://travis-ci.org/zysite/biaffine-parser)
[![LICENSE](https://img.shields.io/github/license/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/stargazers)		
[![GitHub forks](https://img.shields.io/github/forks/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/network/members)

An implementation of "Deep Biaffine Attention for Neural Dependency Parsing".

Details and [hyperparameter choices](#Hyperparameters) are almost identical to those described in the paper, except that we do not provide a decoding algorithm to ensure well-formedness, which does not seriously affect the results.

Another version of the implementation is available on [char](https://github.com/zysite/biaffine-parser/tree/char) branch, which replaces the tag embedding with char lstm and achieves better performance.

## Requirements

```txt
python == 3.7.0
pytorch == 1.0.0
```

## Datasets

The model is evaluated on the Stanford Dependency conversion ([v3.3.0](https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip)) of the English Penn Treebank with POS tags predicted by [Stanford POS tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip).

For all datasets, we follow the conventional data splits:

* Train: 02-21 (39,832 sentences)
* Dev: 22 (1,700 sentences)
* Test: 23 (2,416 sentences)

## Performance

|               |  UAS  |  LAS  |
| ------------- | :---: | :---: |
| tag embedding | 95.85 | 94.14 |
| char lstm     | 96.02 | 94.38 |

Note that punctuation is excluded in all evaluation metrics. 

Aside from using consistent hyperparameters, there are some keypoints that significantly affect the performance:

- Dividing the pretrained embedding by its standard-deviation
- Applying the same dropout mask at every recurrent timestep
- Jointly dropping the words and tags

For the above reasons, we may have to give up some native modules in pytorch (e.g., `LSTM` and `Dropout`), and use self-implemented ones instead.

As shown above, our results, especially on char lstm version, have outperformed the [offical implementation](https://github.com/tdozat/Parser-v1) (95.74 and 94.08).

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

Before triggering the subparser, please make sure that the data files must be in CoNLL-X format. If some fields are missing, you can use underscores as placeholders.

Optional arguments of the subparsers are as follows:

```sh
$ python run.py train -h
usage: run.py train [-h] [--buckets BUCKETS] [--punct] [--ftrain FTRAIN]
                    [--fdev FDEV] [--ftest FTEST] [--fembed FEMBED]
                    [--unk UNK] [--conf CONF] [--model MODEL] [--vocab VOCAB]
                    [--device DEVICE] [--seed SEED] [--threads THREADS]

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
  --model MODEL, -m MODEL
                        path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocab file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads

$ python run.py evaluate -h
usage: run.py evaluate [-h] [--batch-size BATCH_SIZE] [--buckets BUCKETS]
                       [--punct] [--fdata FDATA] [--conf CONF] [--model MODEL]
                       [--vocab VOCAB] [--device DEVICE] [--seed SEED]
                       [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --buckets BUCKETS     max num of buckets to use
  --punct               whether to include punctuation
  --fdata FDATA         path to dataset
  --conf CONF, -c CONF  path to config file
  --model MODEL, -m MODEL
                        path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocab file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads

$ python run.py predict -h
usage: run.py predict [-h] [--batch-size BATCH_SIZE] [--fdata FDATA]
                      [--fpred FPRED] [--conf CONF] [--model MODEL]
                      [--vocab VOCAB] [--device DEVICE] [--seed SEED]
                      [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --fdata FDATA         path to dataset
  --fpred FPRED         path to predicted result
  --conf CONF, -c CONF  path to config file
  --model MODEL, -m MODEL
                        path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocab file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
```

## Hyperparameters

| Param         | Description                                      |                                 Value                                  |
| :------------ | :----------------------------------------------- | :--------------------------------------------------------------------: |
| n_embed       | dimension of word embedding                      |                                  100                                   |
| n_tag_embed   | dimension of tag embedding                       |                                  100                                   |
| embed_dropout | dropout ratio of embeddings                      |                                  0.33                                  |
| n_lstm_hidden | dimension of lstm hidden state                   |                                  400                                   |
| n_lstm_layers | number of lstm layers                            |                                   3                                    |
| lstm_dropout  | dropout ratio of lstm                            |                                  0.33                                  |
| n_mlp_arc     | arc mlp size                                     |                                  500                                   |
| n_mlp_rel     | label mlp size                                   |                                  100                                   |
| mlp_dropout   | dropout ratio of mlp                             |                                  0.33                                  |
| lr            | starting learning rate of training               |                                  2e-3                                  |
| betas         | hyperparameter of momentum and L2 norm           |                               (0.9, 0.9)                               |
| epsilon       | stability constant                               |                                 1e-12                                  |
| annealing     | formula of learning rate annealing               | <img src="https://latex.codecogs.com/gif.latex?.75^{\frac{t}{5000}}"/> |
| batch_size    | approximate number of tokens per training update |                                  5000                                  |
| epochs        | max number of epochs                             |                                 50000                                  |
| patience      | patience for early stop                          |                                  100                                   |

## References

* [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
 