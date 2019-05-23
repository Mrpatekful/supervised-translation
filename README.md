# Neural machine translation

Supervised neural machine translation on english-german multi30k dataset with seq2seq architecture, luong-style general attention mechanism and beam search in pytorch. I employ label smoothing and mixture of softmaxes with several other tweaks for better results. My forked [version](https://github.com/Mrpatekful/text) of torchtext is used as dataloader during training.

## Results

The model reaches 20 BLEU score out of the box with the default parameters in 10 epochs. The randomly picked examples below are from the test split of multi30k dataset and they give an overall idea about the translation performance.

```text
I am playing tennis.
ich spiele tennis . <eos>


How old are you?
wie alt sind du ? <eos>


Do you want to play basketball?
willst sie basketball spielen spielen <eos>


A man in jeans at the beach playing with a red ball.
ein mann in jeans am am strand mit einem roten ball <eos>


A man cooking food on the stove.
ein mann kocht essen dem dem herd . <eos>
```

## References

- **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)**

- **[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)**

- **[Breaking the Softmax Bottleneck: A High-Rank RNN Language Model](https://arxiv.org/pdf/1711.03953.pdf)**
