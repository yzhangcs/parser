# Dependency Parsing

Below are examples of training `biaffine`  and `crf2o` dependency parsers on PTB.

```sh
# biaffine
$ python -u -m supar.cmds.dep.biaffine train -b -d 0 -c dep-biaffine-en -p model -f char  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \
    --embed glove-6b-100
# crf2o
$ python -u -m supar.cmds.dep.crf2o train -b -d 0 -c dep-crf2o-en -p model -f char  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \
    --embed glove-6b-100  \
    --mbr  \
    --proj
```
The option `-c` controls where to load predefined configs, you can either specify a local file path or the same short name as a pretrained model.
For CRF models, you ***must*** specify `--proj` to remove non-projective trees.

Specifying `--mbr` to perform MBR decoding often leads to consistent improvement.

The model trained by finetuning [`robert-large`](https://huggingface.co/roberta-large) achieves nearly state-of-the-art performance in English dependency parsing.
Here we provide some recommended hyper-parameters (not the best, but good enough).
You are allowed to set values of registered/unregistered parameters in command lines to suppress default configs in the file.
```sh
$ python -u -m supar.cmds.dep.biaffine train -b -d 0 -c dep-biaffine-roberta-en -p model  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \
    --encoder=bert  \
    --bert=roberta-large  \
    --lr=5e-5  \
    --lr-rate=20  \
    --batch-size=5000  \
    --epochs=10  \
    --update-steps=4
```
The pretrained multilingual model `dep-biaffine-xlmr` is finetuned on [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large).
The training command is:
```sh
$ python -u -m supar.cmds.dep.biaffine train -b -d 0 -c dep-biaffine-xlmr -p model  \
    --train ud2.3/train.conllx  \
    --dev ud2.3/dev.conllx  \
    --test ud2.3/test.conllx  \
    --encoder=bert  \
    --bert=xlm-roberta-large  \
    --lr=5e-5  \
    --lr-rate=20  \
    --batch-size=5000  \
    --epochs=10  \
    --update-steps=4
```

To evaluate:
```sh
# biaffine
python -u -m supar.cmds.dep.biaffine evaluate -d 0 -p dep-biaffine-en --data ptb/test.conllx --tree  --proj
# crf2o
python -u -m supar.cmds.dep.crf2o evaluate -d 0 -p dep-crf2o-en --data ptb/test.conllx --mbr --tree --proj
```
`--tree` and `--proj` ensure that the output trees are well-formed and projective, respectively.

The commands for training and evaluating Chinese models are similar, except that you need to specify `--punct` to include punctuation.
