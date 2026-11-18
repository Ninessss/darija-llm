"""Normalisation des textes en darija (alphabet arabe et arabizi)."""

import re

# Diacritiques arabes (harakat) et tatweel
_DIACRITICS = re.compile(r"[ً-ْٰ]")
_TATWEEL = "ـ"

# Variantes d'alef -> alef simple
_ALEF_VARIANTS = re.compile(r"[آأإٱ]")

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_PHONE = re.compile(r"\+?\d[\d\s.-]{7,}\d")
_SPACES = re.compile(r"\s+")

# Une lettre répétée 3 fois ou plus ("bzzzzaf" -> "bzzaf")
_STRETCH = re.compile(r"(.)\1{2,}")


def clean(text: str) -> str:
    """Retire URLs, mentions, numéros de téléphone et espaces superflus."""
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _PHONE.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def normalize_arabic(text: str) -> str:
    """Normalise l'alphabet arabe : alef, ta marbouta, alef maqsura, tatweel, diacritiques."""
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _ALEF_VARIANTS.sub("ا", text)   # أ إ آ ٱ -> ا
    text = text.replace("ة", "ه")     # ة -> ه
    text = text.replace("ى", "ي")     # ى -> ي
    return text


def reduce_stretching(text: str, max_repeat: int = 2) -> str:
    """Réduit les lettres étirées à `max_repeat` occurrences."""
    return _STRETCH.sub(lambda m: m.group(1) * max_repeat, text)


def detect_script(text: str, threshold: float = 0.5) -> str:
    """Renvoie 'arabizi' si la majorité des lettres sont latines, sinon 'arabic'."""
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    letters = arabic + latin
    if letters == 0:
        return "unknown"
    return "arabizi" if latin / letters >= threshold else "arabic"


def normalize(text: str) -> str:
    """Pipeline complet appliqué à toutes les sources."""
    text = clean(text)
    text = reduce_stretching(text)
    if detect_script(text) == "arabic":
        text = normalize_arabic(text)
    else:
        text = text.lower()
    return text
