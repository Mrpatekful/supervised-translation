# Neural machine translation

Minimalist implementation of supervised neural machine translation on multi30k dataset with seq2seq architecture, attention meachanism and beam search in pytorch. My forked [version](https://github.com/Mrpatekful/text) of torchtext is used as dataloader.

## References

- **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)**

- **[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)**

- **[Beam Search Strategies for Neural Machine Translation](https://arxiv.org/pdf/1702.01806.pdf)**

## Results

```
i am playing tennis .
ich spiele tennis tennis <eos>


how old are you ?
wie alt sind du ? <eos>


do you want to play basketball ?
willst sie basketball spielen spielen <eos>


what is your favourite hobby ?
was ist dein hobby freizeitbeschäftigung ? <eos>


my name is something .
mein heiße ist etwas . <eos>
```