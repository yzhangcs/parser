## Constituency Parsing

Command for training `crf` constituency parser is simple.
We follow instructions of [Benepar](https://github.com/nikitakit/self-attentive-parser) to preprocess the data.

To train a BiLSTM-based model:
```sh
$ python -u -m supar.cmds.const.crf train -b -d 0 -c con-crf-en -p model -f char --mbr
    --train ptb/train.pid  \
    --dev ptb/dev.pid  \
    --test ptb/test.pid  \
    --embed glove-6b-100  \
    --mbr
```

To finetune [`robert-large`](https://huggingface.co/roberta-large):
```sh
$ python -u -m supar.cmds.const.crf train -b -d 0 -c con-crf-roberta-en -p model  \
    --train ptb/train.pid  \
    --dev ptb/dev.pid  \
    --test ptb/test.pid  \
    --encoder=bert  \
    --bert=roberta-large  \
    --lr=5e-5  \
    --lr-rate=20  \
    --batch-size=5000  \
    --epochs=10  \
    --update-steps=4
```

The command for finetuning [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large) on merged treebanks of 9 languages in SPMRL dataset is:
```sh
$ python -u -m supar.cmds.const.crf train -b -d 0 -c con-crf-roberta-en -p model  \
    --train spmrl/train.pid  \
    --dev spmrl/dev.pid  \
    --test spmrl/test.pid  \
    --encoder=bert  \
    --bert=xlm-roberta-large  \
    --lr=5e-5  \
    --lr-rate=20  \
    --batch-size=5000  \
    --epochs=10  \
    --update-steps=4
```

Different from conventional evaluation manner of executing `EVALB`, we internally integrate python code for constituency tree evaluation.
As different treebanks do not share the same evaluation parameters, it is recommended to evaluate the results in interactive mode.

To evaluate English and Chinese models:
```py
>>> Parser.load('con-crf-en').evaluate('ptb/test.pid',
                                       delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                                       equal={'ADVP': 'PRT'},
                                       verbose=False)
(0.21318972731630007, UCM: 50.08% LCM: 47.56% UP: 94.89% UR: 94.71% UF: 94.80% LP: 94.16% LR: 93.98% LF: 94.07%)
>>> Parser.load('con-crf-zh').evaluate('ctb7/test.pid',
                                       delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                                       equal={'ADVP': 'PRT'},
                                       verbose=False)
(0.3994724107416053, UCM: 24.96% LCM: 23.39% UP: 90.88% UR: 90.47% UF: 90.68% LP: 88.82% LR: 88.42% LF: 88.62%)
```

To evaluate the multilingual model:
```py
>>> Parser.load('con-crf-xlmr').evaluate('spmrl/eu/test.pid',
                                         delete={'TOP', 'ROOT', 'S1', '-NONE-', 'VROOT'},
                                         equal={},
                                         verbose=False)
(0.45620645582675934, UCM: 53.07% LCM: 48.10% UP: 94.74% UR: 95.53% UF: 95.14% LP: 93.29% LR: 94.07% LF: 93.68%)
```
