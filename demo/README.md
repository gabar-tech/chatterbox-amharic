# Demo audio

Same sentences, same reference voice, two models:

- `NN_stock.wav` — stock Chatterbox Multilingual v3, given the identical
  Amharic text with `language_id="en"` (the base model has no Amharic mode
  and zero Fidel tokenizer coverage; this is what "no support" sounds like).
- `NN_finetuned.wav` — v3 + this adapter, `language_id="am"`,
  `temperature=0.6`, `cfg_weight=0.5`.

Texts are in `sentences.txt` (tab-separated: index, text). The reference
voice is a consenting in-house speaker from the `ih_` corpus. All clips carry
the PerTh watermark.
