"""Métriques partagées par toutes les expériences."""

import math
from typing import Iterable, Sequence

import numpy as np


# ---------- Expériences 1 et 2 : tokenisation ----------

def n_tokens(tokenizer, text: str) -> int:
    """Nombre de tokens produits par un tokenizer Hugging Face, sans tokens spéciaux."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def fertility(tokenizer, texts: Iterable[str]) -> float:
    """Nombre moyen de tokens par mot (mots séparés par des espaces)."""
    total_tokens, total_words = 0, 0
    for t in texts:
        total_tokens += n_tokens(tokenizer, t)
        total_words += len(t.split())
    return total_tokens / max(total_words, 1)


def parity(tokenizer, texts: Sequence[str], reference: Sequence[str]) -> float:
    """Ratio de parité moyen (Petrov et al., 2023).

    `texts[i]` et `reference[i]` sont la même phrase dans deux langues
    (par ex. darija et français). Un ratio de 2 = deux fois plus de tokens.
    """
    assert len(texts) == len(reference), "Les deux listes doivent être parallèles"
    ratios = [
        n_tokens(tokenizer, a) / max(n_tokens(tokenizer, b), 1)
        for a, b in zip(texts, reference)
    ]
    return float(np.mean(ratios))


def is_broken_token(tokenizer, tok_id: int) -> bool:
    """Vrai si le token est inconnu ([UNK]) ou ne forme pas un caractère complet.

    Les tokenizers « byte-level » (GPT-2, Llama 3, Qwen) et ceux avec « byte fallback »
    (Gemma) découpent les lettres rares en octets : décodé seul, un tel token donne
    le caractère de remplacement U+FFFD. C'est le signe qu'une lettre arabe a été
    coupée en morceaux.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and tok_id == unk_id:
        return True
    return "\ufffd" in tokenizer.decode([tok_id])


def unk_or_byte_rate(tokenizer, texts: Iterable[str]) -> float:
    """Part des tokens inconnus ou réduits à des octets bruts (voir `is_broken_token`)."""
    total, bad = 0, 0
    for t in texts:
        for tok_id in tokenizer.encode(t, add_special_tokens=False):
            total += 1
            bad += is_broken_token(tokenizer, tok_id)
    return bad / max(total, 1)


# ---------- Expérience 3 : modèle de langue ----------

def bits_per_char(total_nll_nats: float, n_chars: int) -> float:
    """Bits par caractère : comparable entre tokenizers différents.

    `total_nll_nats` = somme des -log p(token) (en nats, comme la CrossEntropyLoss
    de PyTorch avec reduction='sum') sur tout le texte de test.
    `n_chars` = nombre de caractères de ce même texte.
    """
    return total_nll_nats / (n_chars * math.log(2))


# ---------- Expérience 4 : classification ----------

def classification_report(y_true, y_pred, labels=None) -> dict:
    """Accuracy, F1 macro et matrice de confusion."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }
