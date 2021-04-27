# Examples

This file provides instructions on how to train parsing models from scratch and evaluate them.
Some information has been given in [`README`](README.md).
Here we describe in detail the commands and other settings.

## Dependency Parsing

Below are examples of training `biaffine`  and `crf2o` dependency parsers on PTB.

```sh
# biaffine
$ python -u -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-en -p model -f char  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \
    --embed glove.6B.100d.txt  \
    --unk
# crf2o
$ python -u -m supar.cmds.crf2o_dep train -b -d 0 -c crf2o-dep-en -p model -f char  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \
    --embed glove.6B.100d.txt  \
    --unk unk  \
    --mbr  \
    --proj
```
The option `-c` controls where to load predefined configs, you can either specify a local file path or the same short name as a pretrained model.
For CRF models, you need to specify `--proj` to remove non-projective trees.
Specifying `--mbr` to perform MBR decoding often leads to consistent improvement.

The model trained by finetuning [`robert-large`](https://huggingface.co/roberta-large) achieves nearly state-of-the-art performance in English dependency parsing.
Here we provide some recommended hyper-parameters (not the best, but good enough).
You are allowed to set values of registered/unregistered parameters in bash to suppress default configs in the file.
```sh
$ python -u -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-roberta-en -p model  \
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
The pretrained multilingual model `biaffine-dep-xlmr` takes [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large) as backbone architecture and finetunes it.
The training command is as following:
```sh
$ python -u -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-xlmr -p model  \
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
python -u -m supar.cmds.biaffine_dep evaluate -d 0 -p biaffine-dep-en --data ptb/test.conllx --tree  --proj
# crf2o
python -u -m supar.cmds.crf2o_dep evaluate -d 0 -p crf2o-dep-en --data ptb/test.conllx --mbr --tree --proj
```
`--tree` and `--proj` ensures to output well-formed and projective trees respectively.

The commands for training and evaluating Chinese models are similar, except that you need to specify `--punct` to include punctuation.

## Constituency Parsing

Command for training `crf` constituency parser is simple.
We follow instructions of [Benepar](https://github.com/nikitakit/self-attentive-parser) to preprocess the data.

To train a BiLSTM-based model:
```sh
$ python -u -m supar.cmds.crf_con train -b -d 0 -c crf-con-en -p model -f char --mbr
    --train ptb/train.pid  \
    --dev ptb/dev.pid  \
    --test ptb/test.pid  \
    --embed glove.6B.100d.txt  \
    --unk unk  \
    --mbr
```

To finetune [`robert-large`](https://huggingface.co/roberta-large):
```sh
$ python -u -m supar.cmds.crf_con train -b -d 0 -c crf-con-roberta-en -p model  \
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
$ python -u -m supar.cmds.crf_con train -b -d 0 -c crf-con-roberta-en -p model  \
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
>>> Parser.load('crf-con-en').evaluate('ptb/test.pid',
                                       delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                                       equal={'ADVP': 'PRT'},
                                       verbose=False)
(0.21318972731630007, UCM: 50.08% LCM: 47.56% UP: 94.89% UR: 94.71% UF: 94.80% LP: 94.16% LR: 93.98% LF: 94.07%)
>>> Parser.load('crf-con-zh').evaluate('ctb7/test.pid',
                                       delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                                       equal={'ADVP': 'PRT'},
                                       verbose=False)
(0.3994724107416053, UCM: 24.96% LCM: 23.39% UP: 90.88% UR: 90.47% UF: 90.68% LP: 88.82% LR: 88.42% LF: 88.62%)
```

To evaluate the multilingual model:
```py
>>> Parser.load('crf-con-xlmr').evaluate('spmrl/eu/test.pid',
                                         delete={'TOP', 'ROOT', 'S1', '-NONE-', 'VROOT'},
                                         equal={},
                                         verbose=False)
(0.45620645582675934, UCM: 53.07% LCM: 48.10% UP: 94.74% UR: 95.53% UF: 95.14% LP: 93.29% LR: 94.07% LF: 93.68%)
```

## Semantic Dependency Parsing

The raw semantic dependency parsing datasets are not in line with the `conllu` format.
We follow [Second_Order_SDP](https://github.com/wangxinyu0922/Second_Order_SDP) to preprocess the data into the format shown in the following example.
```txt
#20001001
1	Pierre	Pierre	_	NNP	_	2	nn	_	_
2	Vinken	_generic_proper_ne_	_	NNP	_	9	nsubj	1:compound|6:ARG1|9:ARG1	_
3	,	_	_	,	_	2	punct	_	_
4	61	_generic_card_ne_	_	CD	_	5	num	_	_
5	years	year	_	NNS	_	6	npadvmod	4:ARG1	_
6	old	old	_	JJ	_	2	amod	5:measure	_
7	,	_	_	,	_	2	punct	_	_
8	will	will	_	MD	_	9	aux	_	_
9	join	join	_	VB	_	0	root	0:root|12:ARG1|17:loc	_
10	the	the	_	DT	_	11	det	_	_
11	board	board	_	NN	_	9	dobj	9:ARG2|10:BV	_
12	as	as	_	IN	_	9	prep	_	_
13	a	a	_	DT	_	15	det	_	_
14	nonexecutive	_generic_jj_	_	JJ	_	15	amod	_	_
15	director	director	_	NN	_	12	pobj	12:ARG2|13:BV|14:ARG1	_
16	Nov.	Nov.	_	NNP	_	9	tmod	_	_
17	29	_generic_dom_card_ne_	_	CD	_	16	num	16:of	_
18	.	_	_	.	_	9	punct	_	_
```

By default, BiLSTM-based semantic dependency parsing models take POS tag, lemma, and character embeddings as model inputs.
Below are examples of training `biaffine` and `vi` semantic dependency parsing models:
```sh
# biaffine
$ python -u -m supar.cmds.biaffine_sdp train -b -c biaffine-sdp-en -d 0 -f tag char lemma -p model  \
    --train dm/train.conllu  \
    --dev dm/dev.conllu  \
    --test dm/test.conllu  \
    --embed glove.6B.100d.txt  \
    --unk unk
# vi
$ python -u -m supar.cmds.vi_sdp train -b -c vi-sdp-en -d 1 -f tag char lemma -p model  \
    --train dm/train.conllu  \
    --dev dm/dev.conllu  \
    --test dm/test.conllu  \
    --embed glove.6B.100d.txt  \
    --unk unk  \
    --inference mfvi
```

To finetune [`robert-large`](https://huggingface.co/roberta-large):
```sh
$ python -u -m supar.cmds.biaffine_sdp train -b -d 0 -c biaffine-sdp-roberta-en -p model  \
    --train dm/train.conllu  \
    --dev dm/dev.conllu  \
    --test dm/test.conllu  \
    --encoder=bert  \
    --bert=roberta-large  \
    --lr=5e-5  \
    --lr-rate=1  \
    --batch-size=500  \
    --epochs=10  \
    --update-steps=1
```

To evaluate:
```sh
python -u -m supar.cmds.biaffine_sdp evaluate -d 0 -p biaffine-sdp-en --data dm/test.conllu
```