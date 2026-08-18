# Demo audio

Same sentences, same reference voice, same call, two models:

- `NN_stock.wav` — stock Chatterbox Multilingual v3, given the identical
  normalized text, no language tag (the base has no Amharic mode and no
  Fidel tokenizer coverage; this is what "no support" sounds like).
- `NN_finetuned.wav` — v3 + this adapter through the released loader, no
  language tag, `temperature=0.6`, `cfg_weight=0.5`.

Texts are in `sentences.txt` (tab-separated: index, text). `reference.wav`
is the voice both models cloned (one of us). All generated clips carry the
PerTh watermark; `before_after.mp4` is all eight pairs in one video for
GitHub.
