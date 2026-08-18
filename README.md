---
language:
  - am
license: cc-by-sa-4.0
base_model: ResembleAI/chatterbox
base_model_relation: adapter
library_name: peft
pipeline_tag: text-to-speech
datasets:
  - google/WaxalNLP
  - mozilla-foundation/common_voice_25_0
tags:
  - text-to-speech
  - amharic
  - ethiopia
  - lora
  - voice-cloning
  - chatterbox
---

# Chatterbox Amharic

A LoRA adapter and an extended Fidel tokenizer that teach
[Chatterbox Multilingual v3](https://huggingface.co/ResembleAI/chatterbox)
(Resemble AI, MIT) to speak Amharic, with voice cloning from about ten
seconds of reference audio. Trained only on speech we own or that is licensed
for it.

Stock Chatterbox cannot read Amharic at all: its tokenizer maps every Ge'ez
character to `[UNK]`. So the "before" clips below aren't a weaker version of
the same thing; they're the model guessing at unknown tokens. We add 244
tokens for the script and teach the model what they sound like.

The repo also has [`amharic_text.py`](amharic_text.py), the text normalizer
the model was trained through. No dependencies, works on its own
([below](#amharic-text-front-end)).

## Hear it

Same sentence, same reference voice, same call, both models. No language tag
on either (the adapter was trained without one); the stock tokenizer has no
Ge'ez characters, and a language tag doesn't change that.

| # | Text covers | Stock Chatterbox v3 | + Gabar adapter |
|:-:|---|:-:|:-:|
| 1 | Ordinary prose | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/01_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/01_finetuned.wav"></audio> |
| 2 | Prose, ፥ punctuation | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/02_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/02_finetuned.wav"></audio> |
| 3 | Prose (ejectives ቡ/ጅ) | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/03_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/03_finetuned.wav"></audio> |
| 4 | Numbers + ዓ.ም. date abbreviation | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/04_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/04_finetuned.wav"></audio> |
| 5 | ዶ/ር title abbreviation | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/05_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/05_finetuned.wav"></audio> |
| 6 | Question intonation | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/06_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/06_finetuned.wav"></audio> |
| 7 | Technical prose | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/07_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/07_finetuned.wav"></audio> |
| 8 | Mixed punctuation + question | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/08_stock.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/08_finetuned.wav"></audio> |

Reading this on GitHub? The players only render on Hugging Face. Click a
clip in [`demo/`](demo/) to play it, or watch
[`demo/before_after.mp4`](demo/before_after.mp4) (all eight pairs, 1:47).
The weights (`new_lang_adapter/`, 194 MB) are only on
[Hugging Face](https://huggingface.co/gabar-tech/chatterbox-amharic);
everything else is mirrored here.

Texts: [`demo/sentences.txt`](demo/sentences.txt). Reference voice:
[`demo/reference.wav`](demo/reference.wav), one of us
(<audio controls preload="none" src="https://huggingface.co/gabar-tech/chatterbox-amharic/resolve/main/demo/reference.wav"></audio>). Both models got the text after
`amharic_text.normalize`, `temperature=0.6`, `cfg_weight=0.5`, seed 1234.
One take per sentence per model, no picking. Known weaknesses are under
Limitations.

## What this is, and what it isn't

An adapter: LoRA weights on the T3 text-to-speech-token transformer,
full-rank embeddings for the new tokens, and the extended tokenizer. You
apply it on top of Chatterbox Multilingual v3, which you download from
Resemble. We ship our delta, not a copy of their model.

Not merged weights, not a standalone model, not a production service. Amharic
only: Tigrinya and Ge'ez use the same script and the model will "read" them,
but it has never heard them.

## Amharic text front-end

[`amharic_text.py`](amharic_text.py) built the training labels, and the
loader runs it on every input, so training and inference see the same text.
One file, standard library only, same licence as the adapter, usable
without the model. (One fix since training: a dotted abbreviation's trailing
dot mid-sentence, as in `… ዓ.ም. የአገሪቱ …`, is no longer read as a full stop.
Labels were built with the version whose SHA-256 is in
`training_config.json`; the only effect on the model is one fewer spurious
pause.)

It converts Ge'ez numerals, digits, decimals, percentages and clock times to
words (`1500` → አንድ ሺህ አምስት መቶ, `75%` → ሰባ አምስት በመቶ, `3:30` → ሶስት ሰዓት
ተኩል); expands about a hundred common abbreviations, keeping the inflected
suffix (`ዶ/ር` → ዶክተር, `ዓ.ም.` → ዓመተ ምሕረት, `መ/ቤቱ` → መሥሪያ ቤቱ); collapses the
four consonant families that are spelled several ways but pronounced the same
(ሐ ኀ ኅ → ሀ, ሠ → ሰ, ዐ → አ, ፀ → ጸ); reduces punctuation to the five marks that
change how you say something (`።` `፣` `፤` `?` `!`); strips URLs, emoji and
control characters. It's a subset of what we run in production.

```python
from amharic_text import normalize, split_sentences
normalize("ዶ/ር አበበ በ2018 ዓ.ም በተደረገው ምርጫ 75% ድምፅ አገኙ።")
# 'ዶክተር አበበ በሁለት ሺህ አስራ ስምንት ኣመተ ምህረት በተደረገው ምርጫ ሰባ አምስት በመቶ ድምጽ አገኙ።'
```

## Training data

Three sources. Every clip's filename starts with its corpus prefix; the
assembled training directory was audited before training and the output is
committed as is ([`audit/corpus_audit.txt`](audit/corpus_audit.txt)). The
adapter was trained from scratch on exactly that directory, starting from
Resemble's stock v3 T3.

| prefix | source | licence | clips | hours |
|---|---|---|---|---|
| `ih_` | Our own studio recordings | ours | 573 | 1.25 |
| `wxl_` | [WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) Amharic (Digital Umuganda / Google) | CC-BY-SA-4.0 | 40921 | 190.93 |
| `cv_` | [Common Voice](https://commonvoice.mozilla.org/) Amharic | CC0-1.0 | 1055 | 1.45 |
| | **total** | | 42549 | 193.63 |

WaxalNLP comes as 48 kHz, Common Voice as 32/48 kHz MP3; both were resampled
down to 24 kHz, nothing was upsampled. We used all of the Waxal Amharic that
fit the trainer's 3 to 25 second window (190.93 h of roughly 191): it's
the largest licensed Amharic speech set and has hundreds of speakers, which
is what voice cloning needs. Everyone in the training set recorded under one
of these three licences. We don't redistribute the corpus; the public parts
are at their sources, the studio recordings stay with us.

## Licence: why CC-BY-SA-4.0

Waxal is CC-BY-SA. Whether share-alike carries from a dataset into weights
trained on it is an open legal question, and we didn't want to build a
release on the answer we'd prefer. So the adapter is CC-BY-SA-4.0 too, with
credit to Digital Umuganda, the WaxalNLP contributors, and the Common Voice
contributors. The base model stays MIT; this licence covers what we add.

## Architecture

Base: Chatterbox Multilingual v3, `ResembleAI/chatterbox` at revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`, T3 file `t3_mtl23ls_v3.safetensors`. Pinned on purpose:
the adapter only makes sense on that exact T3. v3 has Resemble's
hallucination and speaker-similarity fixes over v2; S3Gen, the voice encoder
and the tokenizer are the same as v2 and untouched.

| component | treatment |
|---|---|
| T3 (text→speech-token transformer, 0.5 B) | LoRA r=64, α=128, dropout 0.05 on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` + `spkr_enc`; base weights frozen |
| `text_emb` / `text_head` | trained full-rank and shipped whole (PEFT `modules_to_save`). New vocabulary rows can't be learned through a low-rank delta. |
| Tokenizer | base multilingual tokenizer + 244 added tokens: every Ge'ez character seen in the corpus, plus ። ፣ ፤ ? ! and U+135F. `"ሰላም"` in the stock tokenizer is `[UNK] [UNK] [UNK]`. |
| S3Gen (speech tokens → waveform, includes the PerTh watermark) | frozen, not shipped |
| Voice encoder | frozen, not shipped |

A few things worth knowing if you build on this:

*No language token.* The base tokenizer has `[fr]`, `[de]` and so on; there
is no `[am]` and we didn't add one. The adapter was trained on plain
normalized text, and the loader tokenizes without a language prefix, without
lower-casing or NFKD. `language_id="am"` on stock
`ChatterboxMultilingualTTS.generate` raises `ValueError`; use the loader.

*Alignment guard is off.* Upstream enables its attention-alignment
hallucination guard only when `text_tokens_dict_size == 2454`. With the
extended vocabulary it's off, in training and at inference. The loader chunks
by sentence instead, which handles the common failure (T3 stopping at the
first sentence-final mark).

*Front-end parity.* We ran `amharic_text.py` side by side with our internal
label pipeline over every training transcript as a sanity check:
FRONTEND PARITY OK: released amharic_text.normalize agrees with the internal label pipeline on 42535/42557 texts (99.948%); 22 inspected differences. The differences are in
[`audit/frontend_parity.txt`](audit/frontend_parity.txt): number ranges,
where the released file says "እስከ" and the internal one didn't. No
gemination marks were used, see Limitations.

Training config (full resolved config with data manifest hashes in
[`training_config.json`](training_config.json)): AdamW, lr 2e-05, cosine,
5 % warm-up, weight decay 0.01, grad clip 1.0, bf16, gradient checkpointing,
effective batch 16 (4 × 4 accumulation), 8 epochs, seed 42, clips 3
to 25 s at 24 kHz, with the
[chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)
toolkit behind our own wrapper. It started from the stock v3 T3, not from any
earlier checkpoint of ours, and we checked that: LoRA never touches
layernorm weights, so the merged model's layernorms identify its base. They
match the v3 multilingual T3 exactly and not the English T3
([`audit/base_lineage.txt`](audit/base_lineage.txt)). Inference:
`temperature=0.6`, `cfg_weight=0.5`, from a sweep.

## Evaluation

Held-out set: 100 clips from the same corpus, split before training
by a seeded speaker-disjoint rule, so whole speakers are held out and none of
their sentences appear in training. Checked independently of the trainer's
own assertion: HELD OUT: eval ∩ train = ∅ at clip, speaker and sentence level; all eval stems are ih_/wxl_/cv_.
([`audit/holdout_verify.txt`](audit/holdout_verify.txt)). Both models ran on
the same clips with the same per-clip reference audio (the held-out
speaker's own recording), through the same code: the released loader for the
adapter, stock v3 with the same normalized text and no language tag.

| metric | stock Chatterbox v3 | + Gabar adapter |
|---|---|---|
| Amharic CER ↓ (Meta omniASR-CTC-3B) | 0.932 | 0.095 |
| UTMOS ↑ (naturalness MOS predictor) | 2.359 | 2.711 |
| ECAPA cosine ↑ (speaker similarity to reference) | 0.610 | 0.860 |
| generation failures (empty / <0.5 s / error) | 0.0% | 1.0% |

Per-clip numbers: [`audit/eval/`](audit/eval/).

How CER is measured. We transcribe the generated audio with Meta's stock
[omniASR-CTC-3B](https://huggingface.co/facebook/omniASR-CTC-3B), which we
didn't train, and compare to the reference text after `normalize_for_metric`
(collapses the homophone families so ሀ/ሐ/ኀ spellings don't count as errors,
strips punctuation). Same ASR, same normalization, both models. The stock
column is a floor: the base model can't read Fidel, so most of what it
produces isn't Amharic. UTMOS and ECAPA involve no ASR. UTMOS was trained on
English MOS ratings and both outputs are Amharic, so its near-tie says more
about the metric than the models. Means are over clips that produced audio;
failures are on their own row so they can't hide in an average.

Our own listening verdict: intelligible Amharic, not yet fully natural
(read-aloud cadence, occasional flat prosody); the stock output is garbled and
not Amharic. The demo pairs are above; judge for yourself.

## Usage

```bash
pip install "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2" peft safetensors huggingface_hub torchaudio
```

```python
from huggingface_hub import hf_hub_download
import importlib.util, torchaudio

# the loader + text front-end ship in this repo
spec = importlib.util.spec_from_file_location(
    "amharic_tts", hf_hub_download("gabar-tech/chatterbox-amharic", "amharic_tts.py"))
amharic_tts = importlib.util.module_from_spec(spec); spec.loader.exec_module(amharic_tts)

tts = amharic_tts.load_amharic_tts(device="cuda")   # downloads base v3 (pinned) + adapter
wav = tts.generate(
    "ሰላም! ይህ ከጽሑፍ በቀጥታ የተፈጠረ የአማርኛ ድምፅ ነው። ዛሬ ነሐሴ 11 ቀን 2018 ዓ.ም. ነው።",
    audio_prompt_path="reference.wav",     # ~10 s of the voice to clone, with consent
    temperature=0.6, cfg_weight=0.5)
torchaudio.save("out.wav", wav, tts.sr)      # 24 kHz, PerTh-watermarked
print(tts.normalize("ዛሬ ነሐሴ 11 ቀን 2018 ዓ.ም. ነው።"))   # what the model actually read
```

Or from a checkout: `python amharic_tts.py "ሰላም ዓለም።" --ref reference.wav --out out.wav`.

`generate()` normalizes the text, splits at sentence-final marks (T3 tends to
stop at the first ። / ? / !), synthesizes each sentence against the
reference and joins them. `normalize=False` / `split_sentences=False` turn
those off. We ran the snippet above as written in a fresh virtualenv with
only the packages listed, on NVIDIA RTX A6000, Ubuntu 22.04.5 LTS, python 3.11.10, torch 2.6.0+cu124, chatterbox-tts@5de7a54, peft 0.20.0, before publishing.

## Watermarking

Chatterbox puts Resemble's PerTh watermark in every waveform it generates.
We left that alone; nothing in this adapter touches S3Gen or the vocoder,
where it happens. We ran the public `resemble-perth` detector over every demo
clip from both models and the held-out eval outputs, with the natural
reference recording as a negative control: present on all 115 checked files (detector confidence ≥ 0.5 on every generated file; the natural reference recording scores 0.0, so the detector is discriminating)
([`audit/watermark_verify.txt`](audit/watermark_verify.txt)). If you build on
this, leave it in. It's about the only provenance signal that exists for
synthetic Amharic right now.

## Intended use

- Amharic speech interfaces, audiobooks, education, accessibility, media
  production, with the consent of whoever's voice you clone.
- Research on low-resource TTS, script extension, Ethiopian language tech.

## Out of scope

- Cloning someone's voice without their informed consent.
- Political persuasion, impersonating public figures, fraud (voice
  authentication included), harassment.
- Other languages. The adapter makes the base worse at its other languages.
- Anything safety-critical or broadcast without a human listening first.

## Limitations

- Gemination (consonant length; contrastive in Amharic, unwritten in normal
  spelling) isn't marked. The model reads minimal pairs like አለ "said" / አለ
  "there is" from context and will get some wrong. Our internal models use a
  lexicon-based marker; we left it out so that what you type is what the
  model was trained on, with nothing private in between.
- Numbers, abbreviations and Ge'ez numerals are expanded by
  `amharic_text.py`. Skip it, or feed it things it doesn't handle (currency
  symbols, odd date formats, Latin words), and the model gets them raw.
- The first word or two of an utterance are sometimes slurred before the
  model settles; mid-sentence text is steadier. Sentences that open with a
  number phrase show it most.
- Ejectives (ቀ, ጠ, ጨ, ጰ, ጸ) are much better than nothing but not uniformly
  native.
- Very short inputs (a single word, "እሺ") can come out unstable. Put them in
  a sentence.
- Long passages: chunk at sentence boundaries. The loader does this for you.
- 193.63 hours, most of it read speech (Waxal and Common Voice are
  ASR corpora). Expect a neutral, read-aloud delivery; expressive and
  spontaneous speech is thin.
- Amharic only.

## Risks and misuse

This model clones a voice from about ten seconds of audio, in a language with
well over 100 million speakers, in a region where synthetic political speech
is a real problem. Until now Amharic speakers had an accidental protection:
mainstream voice cloning couldn't read Fidel. This release takes that away,
and we're the ones doing it. We think it's the right call: Amharic speakers
should have the same accessibility, education and media tools everyone else
has, and this capability was coming whether or not we built it. But there is
another side, and we won't pretend otherwise. The misuse we expect first is
audio impersonation of politicians, clergy and journalists, and outside the
PerTh watermark these outputs carry there is essentially no detection
infrastructure for Amharic.

Publishing weights means giving up control; a licence clause doesn't change
that. What we could do, we did: outputs are watermarked and we kept that path
intact; the acceptable-use terms above are conditions, not suggestions; every
voice in the training data was recorded under a licence that allows this. If
you work on detection or verification for Amharic synthetic speech, that's
the most useful thing anyone could add here, and we'll help: contact@gabar.io.

## Attribution

- Resemble AI, for Chatterbox Multilingual (MIT), the base model and the
  PerTh watermarker.
- Digital Umuganda and the WaxalNLP contributors, Amharic speech data
  (CC-BY-SA-4.0).
- Mozilla Common Voice contributors, Amharic speech data (CC0-1.0).

## Citation

```bibtex
@misc{gabar2026chatterboxamharic,
  title  = {Chatterbox Amharic: a Fidel extension of Chatterbox Multilingual},
  author = {{Gabar Technologies}},
  year   = {2026},
  url    = {https://huggingface.co/gabar-tech/chatterbox-amharic}
}
```
