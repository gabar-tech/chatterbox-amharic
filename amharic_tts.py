"""Load Chatterbox Multilingual v3 + the gabar-tech Amharic adapter.

This repo ships a *delta*, not a merged model:

    new_lang_adapter/          PEFT LoRA adapter for T3 (q/k/v/o, gate/up/down,
                               spkr_enc) PLUS the full-rank ``text_emb`` /
                               ``text_head`` tables saved as ``modules_to_save``
                               (they carry the added Fidel rows)
    tokenizer/tokenizer.json   base Chatterbox multilingual tokenizer + Fidel
    tokenizer/extension_report.json   {"new_vocab_size": ..., ...}
    amharic_text.py            the text front-end the model was trained on

Loading = build T3 with the extended vocab, copy the stock v3 T3 weights into
it (existing rows), attach the PEFT adapter (LoRA + the trained embedding
tables), swap in the extended tokenizer. S3Gen, the voice encoder and the
PerTh watermarker are stock v3, untouched.

    from amharic_tts import load_amharic_tts
    tts = load_amharic_tts(device="cuda")           # downloads base + adapter
    wav = tts.generate("ሰላም! ይህ የአማርኛ ድምፅ ነው።",
                       audio_prompt_path="reference.wav")   # ~10 s of speech
    import torchaudio; torchaudio.save("out.wav", wav, tts.sr)

Requirements: chatterbox-tts (a build that knows ``t3_model="v3"``, i.e. the
GitHub main branch — PyPI 0.1.7 predates v3), peft, torch, torchaudio,
safetensors, huggingface_hub.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ID = "gabar-tech/chatterbox-amharic"
BASE_REPO = "ResembleAI/chatterbox"
# Chatterbox Multilingual v3 — pinned commit of ResembleAI/chatterbox that
# introduced t3_mtl23ls_v3.safetensors (2026-06-10). The adapter is only
# valid against THIS T3; a different base revision = undefined output.
BASE_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_FILE = "t3_mtl23ls_v3.safetensors"
S3GEN_FILE = "s3gen.pt"
VE_FILE = "ve.pt"
CONDS_FILE = "conds.pt"
BASE_FILES = (T3_FILE, S3GEN_FILE, VE_FILE, CONDS_FILE)

DEFAULT_TEMPERATURE = 0.6
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_EXAGGERATION = 0.5
_SENTENCE_GAP_S = 0.15


def _download(repo_id: str, revision: str | None = None,
              allow_patterns=None) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, revision=revision,
                                  allow_patterns=allow_patterns))


def _load_frontend(adapter_dir: Path):
    """``amharic_text`` from the adapter checkout (single source of truth)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "amharic_text", adapter_dir / "amharic_text.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AmharicTTS:
    """Thin wrapper: text front-end + sentence chunking around
    ``ChatterboxMultilingualTTS``."""

    def __init__(self, model, frontend):
        self.model = model
        self.frontend = frontend
        self.sr = model.sr
        self.device = model.device

    def normalize(self, text: str) -> str:
        return self.frontend.normalize(text)

    def generate(self, text: str, audio_prompt_path: str | None = None, *,
                 temperature: float = DEFAULT_TEMPERATURE,
                 cfg_weight: float = DEFAULT_CFG_WEIGHT,
                 exaggeration: float = DEFAULT_EXAGGERATION,
                 normalize: bool = True, split_sentences: bool = True,
                 **kwargs):
        """Return a ``[1, N]`` float tensor at ``self.sr`` (24 kHz).

        Text goes through ``amharic_text.normalize`` (the training-label
        pipeline) unless ``normalize=False``. Long inputs are synthesised one
        sentence at a time — T3 tends to stop at the first sentence-final
        mark — and joined with a short gap. Every chunk goes through the
        stock ``ChatterboxMultilingualTTS.generate`` (PerTh watermark
        included). ``language_id`` is intentionally NOT passed: the adapter
        was trained without a language token (there is no ``[am]`` token in
        the base vocabulary), and the tokenizer below ignores it anyway.
        """
        import torch
        text = self.normalize(text) if normalize else text.strip()
        if not text:
            raise ValueError("empty text after normalization")
        chunks = self.frontend.split_sentences(text) if split_sentences else [text]
        wavs = []
        gap = torch.zeros(1, int(_SENTENCE_GAP_S * self.sr))
        for i, chunk in enumerate(chunks):
            wav = self.model.generate(
                chunk, language_id=None,
                audio_prompt_path=audio_prompt_path if i == 0 else None,
                temperature=temperature, cfg_weight=cfg_weight,
                exaggeration=exaggeration, **kwargs)
            wavs.append(wav.detach().cpu())
            if i < len(chunks) - 1:
                wavs.append(gap)
        return torch.cat(wavs, dim=-1)


def load_amharic_tts(device: str = "cuda",
                     adapter_dir: str | Path | None = None,
                     base_dir: str | Path | None = None,
                     merge_adapter: bool = True) -> AmharicTTS:
    """Return a ready :class:`AmharicTTS`.

    ``adapter_dir`` / ``base_dir`` point at local checkouts (offline use, CI);
    otherwise both are fetched from the Hub (base pinned to ``BASE_REVISION``,
    only the four files the multilingual model needs).
    """
    import torch
    from safetensors.torch import load_file
    from peft import PeftModel
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.tokenizers import MTLTokenizer

    adapter = Path(adapter_dir) if adapter_dir else _download(REPO_ID)
    base = Path(base_dir) if base_dir else _download(
        BASE_REPO, BASE_REVISION, allow_patterns=list(BASE_FILES))
    for f in BASE_FILES[:3]:
        if not (base / f).exists():
            raise FileNotFoundError(f"base dir {base} is missing {f}")
    map_location = torch.device("cpu") if device in ("cpu", "mps") else None

    report = json.loads(
        (adapter / "tokenizer" / "extension_report.json").read_text("utf-8"))
    new_vocab_size = int(report["new_vocab_size"])

    # T3 with the extended text vocab; stock v3 weights copied in. text_emb /
    # text_head are larger than the base tables — copy the overlapping rows,
    # the Fidel rows come from the adapter's modules_to_save.
    cfg = T3Config.multilingual()
    cfg.text_tokens_dict_size = new_vocab_size
    t3 = T3(cfg)
    base_state = load_file(str(base / T3_FILE))
    if "model" in base_state:
        base_state = base_state["model"][0]
    own = t3.state_dict()
    missing = [k for k in base_state if k not in own]
    if missing or len(base_state) < 0.99 * len(own):
        raise RuntimeError(
            f"base T3 checkpoint does not match this chatterbox build: "
            f"{len(missing)} unknown keys (e.g. {missing[:3]}), "
            f"{len(base_state)} vs {len(own)} tensors")
    for k, v in base_state.items():
        if own[k].shape != v.shape:
            if k not in ("text_emb.weight", "text_head.weight", "text_head.bias"):
                raise RuntimeError(f"unexpected shape mismatch on {k}: "
                                   f"{tuple(v.shape)} vs {tuple(own[k].shape)}")
            n = min(own[k].shape[0], v.shape[0])
            own[k][:n].copy_(v[:n])
        else:
            own[k].copy_(v)
    t3.load_state_dict(own)
    t3 = PeftModel.from_pretrained(t3, str(adapter / "new_lang_adapter"),
                                   is_trainable=False)
    if merge_adapter:
        t3 = t3.merge_and_unload()
    t3.to(device).eval()

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(base / VE_FILE, map_location=map_location,
                                  weights_only=True))
    ve.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(base / S3GEN_FILE,
                                     map_location=map_location,
                                     weights_only=True))
    s3gen.to(device).eval()

    class _FidelTokenizer(MTLTokenizer):
        """Tokenize exactly as the training toolkit did: no language token,
        no lower-casing, no NFKD — just space → [SPACE] and encode. Written
        against the underlying `tokenizers` object so it behaves the same on
        every chatterbox-tts build (0.1.4's text_to_tokens has no
        lowercase/nfkd arguments; newer ones do)."""

        def encode(self, txt, *args, **kwargs):
            return self.tokenizer.encode(txt.replace(" ", "[SPACE]")).ids

        def text_to_tokens(self, text, *args, **kwargs):
            return torch.IntTensor(self.encode(text)).unsqueeze(0)

    tokenizer = _FidelTokenizer(str(adapter / "tokenizer" / "tokenizer.json"))

    conds = None
    if (base / CONDS_FILE).exists():
        conds = Conditionals.load(base / CONDS_FILE,
                                  map_location=map_location).to(device)

    model = ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device,
                                      conds=conds)
    return AmharicTTS(model, _load_frontend(adapter))


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import torchaudio

    ap = argparse.ArgumentParser(description="Synthesize Amharic speech.")
    ap.add_argument("text")
    ap.add_argument("--ref", required=True, help="reference voice wav (~10 s)")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adapter-dir")
    ap.add_argument("--base-dir")
    a = ap.parse_args()
    tts = load_amharic_tts(a.device, a.adapter_dir, a.base_dir)
    wav = tts.generate(a.text, audio_prompt_path=a.ref)
    torchaudio.save(a.out, wav, tts.sr)
    print(f"wrote {a.out} ({wav.shape[-1] / tts.sr:.1f}s)", file=sys.stderr)
