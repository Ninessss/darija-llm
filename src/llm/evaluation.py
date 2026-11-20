
"""Validation et évaluation des prédictions des modèles."""

import re

from src.metrics import classification_report
from src.llm.prompts import LABELS


def parse_prediction(response: str) -> str | None:
    """
    Extrait le label final de la réponse du modèle.

    Accepte :
    - une réponse classique contenant uniquement un label ;
    - une réponse structurée avec Label: ;
    - une réponse CoT avec Final label: .

    Retourne None si la réponse est invalide.
    """
    if not isinstance(response, str) or not response.strip():
        return None

    response = response.strip().lower()

    # Priorité au label final pour les réponses CoT.
    final_match = re.search(
        r"(?im)^\s*final\s+label\s*:\s*([a-z_]+)\s*$",
        response,
    )

    if final_match:
        label = final_match.group(1).strip()
        return label if label in LABELS else None

    # Format classique : Label: algerian
    label_match = re.search(
        r"(?im)^\s*label\s*:\s*([a-z_]+)\s*$",
        response,
    )

    if label_match:
        label = label_match.group(1).strip()
        return label if label in LABELS else None

    # Réponse brute : algerian
    if response in LABELS:
        return response

    return None


def evaluate_predictions(
        y_true: list[str],
        y_pred: list[str | None],
) -> dict:
    """
    Calcule les métriques de classification.

    Les prédictions invalides sont comptabilisées comme incorrectes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Les listes y_true et y_pred doivent avoir la même taille."
        )

    if not y_true:
        raise ValueError("La liste des vérités terrain est vide.")

    if not set(y_true).issubset(set(LABELS)):
        raise ValueError("Les vérités terrain contiennent un label inconnu.")

    invalid_label = "__invalid__"

    valid_predictions = [
        prediction if prediction in LABELS else invalid_label
        for prediction in y_pred
    ]

    report = classification_report(
        y_true,
        valid_predictions,
        labels=LABELS,
    )

    report["invalid_count"] = sum(
        prediction == invalid_label
        for prediction in valid_predictions
    )

    report["invalid_rate"] = (
            report["invalid_count"] / len(y_pred)
    )

    return report
