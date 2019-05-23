# Neural machine translation

Supervised neural machine translation on english-german multi30k dataset with seq2seq architecture, luong-style general attention mechanism and beam search in pytorch. I employ label smoothing and mixture of softmaxes with several other tweaks for better results. My forked version of *[torchtext](https://github.com/Mrpatekful/text)* is used as dataloader during training.

## Usage

The model can be trained with the following command.

```bash
./run.sh "train" "<data_dir>" "<model_dir>"
```

An interactive evaluation mode is available on the trained model by
setting the `eval` flag.

```bash
./run.sh "eval" "<data_dir>" "<model_dir>"
```

## Results

```text
A man in jeans at the beach playing with a red ball.
ein mann in jeans am am strand mit einem roten ball


I am playing tennis.
ich spiele tennis .


Do you want to play basketball?
willst sie basketball spielen spielen


How old are you?
wie alt sind du ?


A man cooking food on the stove.
ein mann kocht essen dem dem herd . <eos>
```

## References

```text
@article{SutskeverVL14,
    author    = {Ilya Sutskever and Oriol Vinyals and Quoc V. Le},
    title     = {Sequence to Sequence Learning with Neural Networks},
    url       = {http://arxiv.org/abs/1409.3215},
}
```

```text
@article{LuongPM15,
    author    = {Minh{-}Thang Luong and Hieu Pham and Christopher D. Manning},
    title     = {Effective Approaches to Attention-based Neural Machine Translation},
    url       = {http://arxiv.org/abs/1508.04025},
}
```

```text
@article{abs-1711-03953,
  author    = {Zhilin Yang and Zihang Dai and Ruslan Salakhutdinov and William W. Cohen},
  title     = {Breaking the Softmax Bottleneck: {A} High-Rank {RNN} Language Model},
  url       = {http://arxiv.org/abs/1711.03953},
}
```
