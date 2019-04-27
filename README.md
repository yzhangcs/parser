# Biaffine Parser

[![Travis](https://img.shields.io/travis/zysite/biaffine-parser.svg)](https://travis-ci.org/zysite/biaffine-parser)
[![LICENSE](https://img.shields.io/github/license/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/stargazers)		
[![GitHub forks](https://img.shields.io/github/forks/zysite/biaffine-parser.svg)](https://github.com/zysite/biaffine-parser/network/members)

An implementation of "Deep Biaffine Attention for Neural Dependency Parsing".

Details and [hyperparameter choices](#Hyperparameters) are almost the same as those described in the paper. 

## Requirements

```txt
python == 3.7.0
pytorch == 1.0.0
```

## Usage

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

$ python run.py train -h
usage: run.py train [-h] [--ftrain FTRAIN] [--fdev FDEV] [--ftest FTEST]
                    [--fembed FEMBED] [--device DEVICE] [--seed SEED]
                    [--threads THREADS] [--file FILE] [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --ftrain FTRAIN       path to train file
  --fdev FDEV           path to dev file
  --ftest FTEST         path to test file
  --fembed FEMBED       path to pretrained embedding file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file

$ python run.py evaluate -h
usage: run.py evaluate [-h] [--batch-size BATCH_SIZE] [--include-punct]
                       [--fdata FDATA] [--device DEVICE] [--seed SEED]
                       [--threads THREADS] [--file FILE] [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --include-punct       whether to include punctuation
  --fdata FDATA         path to dataset
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file

$ python run.py predict -h
usage: run.py predict [-h] [--batch-size BATCH_SIZE] [--fdata FDATA]
                      [--fpred FPRED] [--device DEVICE] [--seed SEED]
                      [--threads THREADS] [--file FILE] [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --fdata FDATA         path to dataset
  --fpred FPRED         path to predicted result
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file
```

## Hyperparameters

| Param         | Description                             |                                 Value                                  |
| :------------ | :-------------------------------------- | :--------------------------------------------------------------------: |
| n_embed       | dimension of word embedding             |                                  100                                   |
| n_tag_embed   | dimension of tag embedding              |                                  100                                   |
| embed_dropout | dropout ratio of embeddings             |                                  0.33                                  |
| n_lstm_hidden | dimension of lstm hidden state          |                                  400                                   |
| n_lstm_layers | number of lstm layers                   |                                   3                                    |
| lstm_dropout  | dropout ratio of lstm                   |                                  0.33                                  |
| n_mlp_arc     | arc mlp size                            |                                  500                                   |
| n_mlp_rel     | label mlp size                          |                                  100                                   |
| mlp_dropout   | dropout ratio of mlp                    |                                  0.33                                  |
| lr            | starting learning rate of training      |                                  2e-3                                  |
| betas         | hyperparameter of momentum and L2 norm  |                               (0.9, 0.9)                               |
| epsilon       | stability constant                      |                                 1e-12                                  |
| annealing     | formula of learning rate annealing      | <img src="https://latex.codecogs.com/gif.latex?.75^{\frac{t}{5000}}"/> |
| batch_size    | number of sentences per training update |                                  200                                   |
| epochs        | max number of epochs                    |                                  1000                                  |
| patience      | patience for early stop                 |                                  100                                   |

Aside from using consistent hyperparameters, there are some keypoints that significantly affect the performance of the model:

- Dividing the pretrained embedding by its standard-deviation
- Applying the same dropout mask at every recurrent timestep
- Jointly dropping the words and tags

For the above reasons, we may have to give up using some native modules in pytorch (e.g., `LSTM` and `Dropout`), and use self-implemented ones instead.

## References

* [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
