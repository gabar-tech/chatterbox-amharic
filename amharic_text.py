"""Amharic text front-end for gabar-tech/chatterbox-amharic.

Part of https://huggingface.co/gabar-tech/chatterbox-amharic (CC-BY-SA-4.0).
Standard library only — usable on its own for any Amharic TTS/NLP pipeline.

This is the EXACT text pipeline the adapter was trained on: the training
labels were produced by :func:`normalize` from this file, and the loader in
``amharic_tts.py`` applies :func:`normalize` to every input before it reaches
the model. Train/inference parity is by construction — if you bypass this
function and feed the model raw text, expect degraded output on numbers,
abbreviations, homophone spellings and punctuation.

What it does (in order):

1. NFC-compose; drop URLs, e-mail addresses, emoji, control characters and
   any U+135F gemination marks (this model does not use them).
2. Ge'ez numerals (፩ … ፼) → Arabic digits (``፬፻፳፪`` → ``422``).
3. Common abbreviations → spoken form (``ዶ/ር`` → ``ዶክተር``, ``ዓ.ም`` →
   ``ዓመተ ምሕረት``, …). Inflection-tolerant: ``መ/ቤቱ`` → ``መሥሪያ ቤቱ``.
4. ``75%`` → ``ሰባ አምስት በመቶ``; integers, decimals and comma-grouped numbers
   → Amharic words (``1500`` → ``አንድ ሺህ አምስት መቶ``); simple times
   (``3:30`` → ``ሦስት ሰዓት ተኩል``).
5. Homophone collapse: the four Amharic consonant families that are spelled
   differently but pronounced identically are mapped to one canonical
   series (ሐ ኀ ኅ → ሀ, ሠ → ሰ, ዐ → አ, ፀ → ጸ, and their vowel orders). This is
   acoustically lossless and removes spelling variance the model would
   otherwise have to memorise.
6. Punctuation canonicalisation: keep the five marks that carry prosody
   (``።`` sentence end, ``፣`` / ``፤`` short pause, ``?`` / ``!``), map ASCII
   and Ethiopic variants onto them (``.``→``።``, ``,``→``፣``, ``፥``/``:``→``፤``,
   ``፧``→``?``), drop every other punctuation mark, collapse runs.
7. Whitespace squeeze.

Nothing here requires a dependency beyond the standard library.
"""
from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize", "split_sentences", "collapse_homophones",
           "canonicalize_punctuation", "expand_numbers", "integer_to_words"]

# ── 1. cleanup ──────────────────────────────────────────────────────────

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_CTRL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\ufeff]")
_WS_RE = re.compile(r"\s+")


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF
            or 0x1F900 <= cp <= 0x1F9FF or cp == 0xFE0F)


_REPEATED_PUNCT_RE = re.compile(r"([.!?,;])\1+")
_HASHTAG_MENTION_RE = re.compile(r"[#@](\w+)")
_BAD_CHARS_RE = re.compile(r"[/\\|_*&~^+=<>{}\[\]]")
_GEEZ_PUNCT_RUN_RE = re.compile(r"[፠-፨](?:\s*[፠-፨])+")
_TYPO_COLON_DASH_RE = re.compile(r":[-–—]")
# `::` is the ordinary way people type the Ethiopic full stop ። on an ASCII
# keyboard. Mapping each `:` separately turned a sentence end into a short
# pause, so consume the pair first.
_ASCII_FULL_STOP_RE = re.compile(r"::")   # "ስራ:- ..." list/heading artefact, not punctuation
_GEEZ_PUNCT_MAP = {0x1360: "።", 0x1361: " ", 0x1365: "፣",
                   0x1366: "፣", 0x1367: "?", 0x1368: "።"}


_GEMINATION_MARK = "\u135f"   # ETHIOPIC COMBINING GEMINATION MARK — not used by this model


def _cleanup(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace(_GEMINATION_MARK, "")
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _CTRL_RE.sub("", text)
    text = "".join(ch for ch in text if not _is_emoji(ch))
    return _REPEATED_PUNCT_RE.sub(r"\1", text)


def _post_expansion_cleanup(text: str) -> str:
    """Symbols the model has no use for; rare Ethiopic marks → the common
    ones (``፡፡`` and ``፠``/``፨`` → ``።``, ``፡`` → space, ``፤``/``፥``/``፦`` →
    ``፣``, ``፧`` → ``?``)."""
    text = _TYPO_COLON_DASH_RE.sub(" ", text)
    text = _ASCII_FULL_STOP_RE.sub("።", text)
    text = _BAD_CHARS_RE.sub(" ", text)
    text = _HASHTAG_MENTION_RE.sub(r"\1", text)
    text = _GEEZ_PUNCT_RUN_RE.sub("። ", text)
    return text.translate(_GEEZ_PUNCT_MAP)


# ── 2. Ge'ez numerals ───────────────────────────────────────────────────

_GEEZ_VALUES = {
    "፩": 1, "፪": 2, "፫": 3, "፬": 4, "፭": 5, "፮": 6, "፯": 7, "፰": 8, "፱": 9,
    "፲": 10, "፳": 20, "፴": 30, "፵": 40, "፶": 50, "፷": 60, "፸": 70, "፹": 80,
    "፺": 90, "፻": 100, "፼": 10000,
}
_GEEZ_RUN_RE = re.compile(r"[፩-፼]+")


def _geez_run_to_int(run: str) -> int:
    total = current = 0
    for ch in run:
        v = _GEEZ_VALUES[ch]
        if v >= 10000:
            total += max(current, 1) * v
            current = 0
        elif v == 100:
            current = max(current, 1) * v
        else:
            current += v
    return total + current


def convert_geez_numerals(text: str) -> str:
    return _GEEZ_RUN_RE.sub(lambda m: str(_geez_run_to_int(m.group(0))), text)


# ── 3. abbreviations ────────────────────────────────────────────────────

ABBREVIATIONS: dict[str, str] = {
    # titles / honorifics
    "ዶ/ር": "ዶክተር", "ዶ.ር": "ዶክተር",
    "ፕ/ር": "ፕሮፌሰር", "ፕ.ር": "ፕሮፌሰር",
    "ረ/ፕ/ር": "ረዳት ፕሮፌሰር",
    "ወ/ሮ": "ወይዘሮ", "ወ.ሮ": "ወይዘሮ",
    "ወ/ሪት": "ወይዘሪት", "ወ/ሪ": "ወይዘሪት",
    "መ/ር": "መምህር",
    "ኢ/ር": "ኢንጂነር",
    "ጋ/ኛ": "ጋዜጠኛ",
    "አምባ/ር": "አምባሳደር",
    "ሊ/መንበር": "ሊቀ መንበር",
    "ኮ/ል": "ኮሎኔል", "ሌ/ኮ/ል": "ሌተና ኮሎኔል",
    "ጄ/ል": "ጄኔራል", "ብ/ጄ/ል": "ብርጋዴር ጄኔራል", "ሜ/ጄ/ል": "ሜጀር ጄኔራል",
    "ሻ/ል": "ሻለቃ", "ሻ/በል": "ሻምበል",
    "መቶ/አለቃ": "መቶ አለቃ", "መ/አለቃ": "መቶ አለቃ",
    "ጠ/ሚ/ር": "ጠቅላይ ሚኒስትር", "ጠ/ሚኒስትር": "ጠቅላይ ሚኒስትር", "ጠ/ሚ": "ጠቅላይ ሚኒስትር",
    "ም/ጠ/ሚ/ር": "ምክትል ጠቅላይ ሚኒስትር", "ም/ጠ/ሚ": "ምክትል ጠቅላይ ሚኒስትር",
    "ሚ/ር": "ሚኒስትር",
    "ፕ/ት": "ፕሬዚዳንት", "ም/ፕ/ት": "ምክትል ፕሬዚዳንት",
    "ሥ/አስኪያጅ": "ሥራ አስኪያጅ",
    # compound names
    "ወ/ጊዮርጊስ": "ወልደ ጊዮርጊስ", "ወ/ማርያም": "ወልደ ማርያም", "ወ/ሥላሴ": "ወልደ ሥላሴ",
    "ገ/መድህን": "ገብረ መድህን", "ገ/ሥላሴ": "ገብረ ሥላሴ", "ገ/ስላሴ": "ገብረ ሥላሴ",
    "ገ/ሚካኤል": "ገብረ ሚካኤል",
    "ኃ/ማርያም": "ኃይለ ማርያም", "ሃ/ማርያም": "ኃይለ ማርያም",
    "ኃ/ሥላሴ": "ኃይለ ሥላሴ", "ኃ/ስላሴ": "ኃይለ ሥላሴ",
    "ቅ/ጊዮርጊስ": "ቅዱስ ጊዮርጊስ", "ቅ/ሚካኤል": "ቅዱስ ሚካኤል", "ቅ/ማርያም": "ቅድስት ማርያም",
    "ተ/ሃይማኖት": "ተክለ ሃይማኖት",
    # places / institutions
    "አ.አ": "አዲስ አበባ", "አ/አ": "አዲስ አበባ",
    "ድ/ዳዋ": "ድሬ ዳዋ", "ባ/ዳር": "ባሕር ዳር",
    "ት/ቤት": "ትምህርት ቤት", "ት.ቤት": "ትምህርት ቤት",
    "መ/ቤት": "መሥሪያ ቤት", "መ.ቤት": "መሥሪያ ቤት",
    "ጽ/ቤት": "ጽሕፈት ቤት", "ፅ/ቤት": "ጽሕፈት ቤት",
    "ፍ/ቤት": "ፍርድ ቤት", "ፍ.ቤት": "ፍርድ ቤት",
    "ጠ/ፍ/ቤት": "ጠቅላይ ፍርድ ቤት", "ከ/ፍ/ቤት": "ከፍተኛ ፍርድ ቤት",
    "ፌ/ፍ/ቤት": "ፌደራል ፍርድ ቤት",
    "ም/ቤት": "ምክር ቤት",
    "ቤ/ክ": "ቤተ ክርስቲያን", "ቤ/ክርስቲያን": "ቤተ ክርስቲያን",
    "ቤ/መ": "ቤተ መንግሥት",
    "ክ/ከተማ": "ክፍለ ከተማ", "ክ/ሀገር": "ክፍለ ሀገር", "ክ/ዘመን": "ክፍለ ዘመን",
    "ዩ/ቲ": "ዩኒቨርሲቲ",
    "ሆ/ል": "ሆስፒታል",
    "ተ/መ/ድ": "ተባበሩት መንግሥታት ድርጅት",
    "ኢ/ኦ/ተ/ቤ/ክ": "ኢትዮጵያ ኦርቶዶክስ ተዋሕዶ ቤተ ክርስቲያን",
    "ብ/ብ/ሕ": "ብሔር ብሔረሰቦችና ሕዝቦች",
    "ፌ/ዴ/ሪ": "ፌደራላዊ ዴሞክራሲያዊ ሪፐብሊክ",
    # calendar / general / units
    "እ.ኤ.አ": "እንደ አውሮፓውያን አቆጣጠር", "እ/ኤ/አ": "እንደ አውሮፓውያን አቆጣጠር",
    "ዓ.ም": "ዓመተ ምሕረት", "ዓ/ም": "ዓመተ ምሕረት",
    "ዓ.ዓ": "ዓመተ ዓለም", "ዓ/ዓ": "ዓመተ ዓለም",
    "ወዘተ": "ወዘተረፈ",
    "ቁ.": "ቁጥር",
    "ኪ.ሜ": "ኪሎ ሜትር", "ኪ/ሜ": "ኪሎ ሜትር",
    "ኪ.ግ": "ኪሎ ግራም", "ኪ/ግ": "ኪሎ ግራም",
    "ሊ.ትር": "ሊትር",
    "መ/ቅ": "መጽሐፍ ቅዱስ",
    "ኃ/የተ/የግ/ማህበር": "ኃላፊነቱ የተወሰነ የግል ማህበር",
    "ኃ/የተ/የግ/ማ": "ኃላፊነቱ የተወሰነ የግል ማህበር",
    "አ/ማህበር": "አክሲዮን ማህበር",
}

_GEEZ_LO, _GEEZ_HI = 0x1200, 0x137F


def _family_chars(ch: str) -> str | None:
    """The 7 vowel-order glyphs of ``ch``'s consonant row (ት → ተቱቲታቴትቶ)."""
    cp = ord(ch)
    if not (_GEEZ_LO <= cp <= _GEEZ_HI):
        return None
    base = (cp // 8) * 8
    return "".join(chr(base + i) for i in range(7))


def _compile_abbreviations():
    rules = []
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        exp = ABBREVIATIONS[abbr]
        fam = _family_chars(abbr[-1])
        if fam and exp[-1] in fam:
            # inflection-tolerant: carry the surface suffix onto the expansion
            pat = re.compile(re.escape(abbr[:-1]) + f"[{fam}][ሀ-፿]*")
            rules.append((abbr, exp, pat, exp[:-1]))
        else:
            rules.append((abbr, exp, None, ""))
    return rules


_ABBR_RULES = _compile_abbreviations()


# Dotted abbreviations are conventionally written with a trailing dot too
# (ዓ.ም. እ.ኤ.አ. ዶ.ር.). Mid-sentence that dot belongs to the abbreviation and
# must not become a sentence end; at the end of a sentence it still does.
def expand_abbreviations(text: str) -> str:
    for abbr, exp, pat, stem in _ABBR_RULES:
        if pat is not None:
            n = len(abbr) - 1
            text = pat.sub(lambda m, s=stem, n=n: f" {s}{m.group(0)[n:]} ", text)
        elif abbr in text:
            if "." in abbr:
                # consume the trailing dot when more text follows
                text = re.sub(re.escape(abbr) + r"\.(?=\s*[^\s.።?!])", f" {exp} ", text)
            text = text.replace(abbr, f" {exp} ")
    return text


# ── 4. numbers ──────────────────────────────────────────────────────────

_ONES = {0: "ዜሮ", 1: "አንድ", 2: "ሁለት", 3: "ሦስት", 4: "አራት", 5: "አምስት",
         6: "ስድስት", 7: "ሰባት", 8: "ስምንት", 9: "ዘጠኝ"}
_TENS = {10: "አሥር", 20: "ሃያ", 30: "ሠላሳ", 40: "አርባ", 50: "ሃምሳ",
         60: "ስልሳ", 70: "ሰባ", 80: "ሰማንያ", 90: "ዘጠና"}
_TEEN_PREFIX = "አሥራ"
_SCALES = [(10**12, "ትሪሊዮን"), (10**9, "ቢሊዮን"), (10**6, "ሚሊዮን"),
           (10**3, "ሺህ"), (10**2, "መቶ")]
_MAX_SPELLED = 10**15


def integer_to_words(n: int) -> str:
    if n < 0:
        return f"ኔጌቲቭ {integer_to_words(-n)}"
    if n >= _MAX_SPELLED:
        return _digits_to_words(str(n))
    return _positive(n)


def _positive(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n == 10:
        return _TENS[10]
    if n < 20:
        return f"{_TEEN_PREFIX} {_ONES[n % 10]}"
    if n < 100:
        t, o = _TENS[(n // 10) * 10], n % 10
        return t if o == 0 else f"{t} {_ONES[o]}"
    for scale, label in _SCALES:
        if n >= scale:
            lead, rem = divmod(n, scale)
            head = label if (lead == 1 and scale == 100) else f"{_positive(lead)} {label}"
            return head if rem == 0 else f"{head} {_positive(rem)}"
    return _ONES[n]


def _digits_to_words(s: str) -> str:
    return " ".join(_ONES[int(d)] for d in s if d.isdigit())


_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_DAYPART = r"(am|pm|AM|PM|a\.m\.|p\.m\.|ጠዋት|ጧት|ከሰዓት|ሌሊት|ለሊት|ምሽት|ማታ)"
# H:MM[:SS] [daypart] [ሰዓት]  — seconds are dropped, a trailing bare "ሰዓት" is
# absorbed (the spoken form already says ሰዓት), a following daypart word is
# spoken first (ጠዋት 3:30 → ጠዋት ሶስት ሰዓት ተኩል).
_TIME_RE = re.compile(
    r"(?<!\d)(\d{1,2})[:፡፥](\d{2})(?:[:፡፥](\d{2}))?"
    r"(?:\s*" + _DAYPART + r")?"
    r"(?:\s*ሰዓት(?![\u1200-\u137f]))?"
    r"(?!\d)")
_TIME_RANGE_RE = re.compile(
    r"(\d{1,2}[:፡፥]\d{2}(?:[:፡፥]\d{2})?(?:\s*" + _DAYPART + r")?(?:\s*ሰዓት(?![ሀ-፿]))?)"
    r"\s*[-–—]\s*(?=\d{1,2}[:፡፥]\d{2})")
_PERIOD_DAYPART = {"am": "ሌሊት", "pm": "ከሰዓት", "ጠዋት": "ጠዋት", "ጧት": "ጠዋት", "ከሰዓት": "ከሰዓት",
                   "ሌሊት": "ሌሊት", "ለሊት": "ሌሊት", "ምሽት": "ምሽት", "ማታ": "ምሽት"}
_NUMBER_RANGE_RE = re.compile(r"\d+(?:\s*[-–—]\s*\d+)+")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![A-Za-z])"
    r"|(?<![A-Za-z0-9])-?\d+(?:\.\d+)?(?![A-Za-z])")


def _looks_like_phone(digits: str) -> bool:
    n = len(digits)
    return n >= 9 and (digits.startswith(("09", "07"))
                       or (digits.startswith("251") and n >= 12))


def _number_body_to_words(body: str) -> str:
    if _looks_like_phone(body):
        return _digits_to_words(body)
    if "." in body:
        i, f = body.split(".", 1)
        return f"{integer_to_words(int(i or '0'))} ነጥብ {_digits_to_words(f)}"
    return integer_to_words(int(body))


def _sub_number(m: re.Match) -> str:
    body = m.group(0).replace(",", "")
    neg = body.startswith("-")
    if neg:
        body = body[1:]
    spoken = _number_body_to_words(body)
    return f"ኔጌቲቭ {spoken}" if neg else spoken


def _sub_percent(m: re.Match) -> str:
    body = m.group(1)
    neg = body.startswith("-")
    if neg:
        body = body[1:]
    spoken = f"{_number_body_to_words(body)} በመቶ"
    return f"ኔጌቲቭ {spoken}" if neg else spoken


def _sub_time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    period = (m.group(4) or "").lower().replace(".", "")
    if h == 24 and mi == 0:
        h = 0
    if h > 23 or mi > 59:
        return m.group(0)
    daypart = _PERIOD_DAYPART.get(period)
    prefix = f"{daypart} " if daypart else ""
    hw = _positive(h) if h > 0 else _ONES[0]
    if mi == 0:
        return f"{prefix}{hw} ሰዓት"
    if mi == 15:
        return f"{prefix}{hw} ሰዓት ከሩብ"
    if mi == 30:
        return f"{prefix}{hw} ሰዓት ተኩል"
    if mi == 45:
        return f"{prefix}ለ{_positive(1 if h >= 23 else h + 1)} ሰዓት ሩብ ጉዳይ"
    return f"{prefix}{hw} ሰዓት ከ{_positive(mi)}"


def _sub_range(m: re.Match) -> str:
    parts = re.split(r"\s*[-–—]\s*", m.group(0))
    if len(parts) != 2 or _looks_like_phone("".join(parts)):
        return m.group(0)
    return f"{parts[0]} እስከ {parts[1]}"


def expand_numbers(text: str) -> str:
    text = convert_geez_numerals(text)
    text = expand_abbreviations(text)
    text = _PERCENT_RE.sub(_sub_percent, text)
    text = _TIME_RANGE_RE.sub(r"\1 እስከ ", text)
    text = _TIME_RE.sub(_sub_time, text)
    text = _NUMBER_RANGE_RE.sub(_sub_range, text)
    text = _NUMBER_RE.sub(_sub_number, text)
    return text


# ── 5. homophone collapse ───────────────────────────────────────────────

_HOMOPHONES: dict[str, str] = {}
for _group in ("ሀሐኀኅ", "ሰሠ", "አዐ", "ጸፀ"):
    _canon = _group[0]
    for _ch in _group[1:]:
        _HOMOPHONES[_ch] = _canon
        for _k in range(1, 7):
            _HOMOPHONES[chr(ord(_ch) + _k)] = chr(ord(_canon) + _k)


def collapse_homophones(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return "".join(_HOMOPHONES.get(ch, ch) for ch in text)


# ── 6. punctuation ──────────────────────────────────────────────────────

_PUNCT_MAP = {".": "።", ",": "፣", ";": "፤", ":": "፤",
              "፥": "፤", "፦": "፤", "፧": "?",
              "።": "።", "፣": "፣", "፤": "፤", "?": "?", "!": "!"}
_KEEP = "።፣፤?!"
_MARK_RUN_RE = re.compile("([" + _KEEP + r"])(?:\s*[" + _KEEP + "])+")
# A sentence mark typed straight against the next word ("እዩ።እቲ") leaves the
# model no boundary to breathe at; give it the space the writer omitted.
_MARK_GLUED_RE = re.compile("([" + _KEEP + r"])(?=[\w\u1200-\u137F])")


def canonicalize_punctuation(text: str) -> str:
    out = []
    for ch in text:
        mapped = _PUNCT_MAP.get(ch)
        if mapped is not None:
            out.append(mapped)
        elif unicodedata.category(ch).startswith("P"):
            continue
        else:
            out.append(ch)
    return _MARK_GLUED_RE.sub(r"\1 ", _MARK_RUN_RE.sub(r"\1", "".join(out)))


# ── public entry points ─────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Model-facing text. Apply to training labels AND to inference input."""
    if not text:
        return ""
    text = _cleanup(text)
    text = expand_numbers(text)
    text = _post_expansion_cleanup(text)
    text = collapse_homophones(text)
    text = canonicalize_punctuation(text)
    return _WS_RE.sub(" ", text).strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[።?!])\s+")


def split_sentences(text: str) -> list[str]:
    """Split normalized text at sentence-final marks. The T3 model tends to
    stop generating at the first sentence-final mark, so long inputs should
    be synthesised sentence by sentence and concatenated."""
    return [s for s in _SENT_SPLIT_RE.split(text.strip()) if s]


if __name__ == "__main__":  # pragma: no cover
    import sys
    for line in sys.stdin:
        print(normalize(line.rstrip("\n")))
