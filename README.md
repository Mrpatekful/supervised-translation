# Neural machine translation

Supervised neural machine translation on english-german multi30k
dataset with seq2seq architecture from *[Sutskever et al. (2014)](https://arxiv.org/pdf/1409.3215.pdf)*, luong-style general attention
mechanism *[Luong et al. (2015)](https://arxiv.org/pdf/1508.04025.pdf)* and beam search in pytorch. I employ *label smoothing [Szegedy et al. (2015)](https://arxiv.org/pdf/1512.00567.pdf)*, *weight dropout* *[Wan et al. (2013)](https://cs.nyu.edu/~wanli/dropc/dropc.pdf)*, *locked dropout [Merity et al. (2017)](https://arxiv.org/pdf/1708.02182.pdf)*, *embedding dropout [Gal & Ghahramani (2016)](https://arxiv.org/pdf/1512.05287.pdf)*, *shared embedding [Press & Wolf (2017)](https://arxiv.org/pdf/1608.05859.pdf)* and
*mixture of softmaxes [Yang et al. (2018)](https://arxiv.org/pdf/1711.03953.pdf)* with several other smaller tweaks for better results. The model uses automatic mixed precision from nvidia/apex.

## Usage

I rely on my own forked version of torchtext, which can be installed from my git repository with pip. The main differences from the original repo is the addition of the faster bucket iterator from harvard-nlp [article](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and the reordering of the special tokens like UNK in the vocabulary for minor technical reasons.

```bash
pip install git+https://github.com/Mrpatekful/text
```

The initial learning rate for *SGD* is determined with "*LR range test*" *[Smith (2017)](https://arxiv.org/pdf/1506.01186.pdf)*. During training I set the learning rate by cosine annealing schedule with warm restarts *[Loshchilov & Hutter (2017)](https://arxiv.org/pdf/1608.03983.pdf)*. I also use my fork of pytorch-nlp, which includes my implementation of several techniques mentioned earlier.

```bash
pip install git+https://github.com/Mrpatekful/PyTorch-NLP
```

The model can be trained with the following command.
Note that `<data_dir>` and `<model_dir>` are optional,
as they are provided by default. Training with different hyperparameters can be done by running the `train.py` script and passing the desired options as command line arguments.

```bash
./run.sh "train" "<data_dir>" "<model_dir>"
```

An interactive evaluation mode is available on the trained model by
switching the `train` to the `eval` flag. During this interactive mode the model uses beam search decoding. Beam width parameter can be modified in the `beam.py` file.

```bash
./run.sh "eval" "<data_dir>" "<model_dir>"
```

## Results

The displayed results are obtained from a model, which was trained with default parameters. These sentences were cherry picked from the interactive evaluation with the default beam width. The model was trained on multi30k dataset, and none of these sentences are present in the training data.

```text
A man in jeans at the beach playing with a red ball.
ein mann in jeans am am strand mit einem roten ball
```

```text
I am playing tennis.
ich spiele tennis .
```

```text
Do you want to play basketball?
willst sie basketball spielen spielen
```

```text
How old are you?
wie alt sind du ?
```

```text
A man cooking food on the stove.
ein mann kocht essen dem dem herd .
```
