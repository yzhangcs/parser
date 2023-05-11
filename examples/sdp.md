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
$ python -u -m supar.cmds.sdp.biaffine train -b -c sdp-biaffine-en -d 0 -f tag char lemma -p model  \
    --train dm/train.conllu  \
    --dev dm/dev.conllu  \
    --test dm/test.conllu  \
    --embed glove-6b-100
# vi
$ python -u -m supar.cmds.sdp.vi train -b -c sdp-vi-en -d 1 -f tag char lemma -p model  \
    --train dm/train.conllu  \
    --dev dm/dev.conllu  \
    --test dm/test.conllu  \
    --embed glove-6b-100  \
    --inference mfvi
```

To finetune [`robert-large`](https://huggingface.co/roberta-large):
```sh
$ python -u -m supar.cmds.sdp.biaffine train -b -d 0 -c sdp-biaffine-roberta-en -p model  \
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
python -u -m supar.cmds.sdp.biaffine evaluate -d 0 -p sdp-biaffine-en --data dm/test.conllu
```