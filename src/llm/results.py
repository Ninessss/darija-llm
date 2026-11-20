
"""Sauvegarde des résultats des expériences LLM."""

from pathlib import Path

import pandas as pd


def save_predictions(
        results_df: pd.DataFrame,
        model_name: str,
        mode: str,
        n_shot: int,
        output_dir: str = "results/llm",
) -> Path:
    """Sauvegarde les IDs et prédictions, sans publier les phrases."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Exclure explicitement le texte et la réponse brute.
    safe_columns = [
        "id",
        "true_label",
        "prediction",
        "is_valid",
        "model",
        "mode",
        "n_shot",
    ]

    missing = set(safe_columns) - set(results_df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes pour la sauvegarde : {sorted(missing)}"
        )

    results = results_df[safe_columns].copy()

    filename = f"{model_name}_{mode}_{n_shot}shot_predictions.csv"
    file_path = output_path / filename

    results.to_csv(file_path, index=False, encoding="utf-8-sig")

    return file_path


def save_metrics(
        metrics: dict,
        model_name: str,
        mode: str,
        n_shot: int,
        output_dir: str = "results/llm",
) -> Path:
    """Sauvegarde ou met à jour les métriques d'une expérience."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_row = {
        "model": model_name,
        "mode": mode,
        "n_shot": n_shot,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "invalid_count": metrics["invalid_count"],
        "invalid_rate": metrics["invalid_rate"],
    }

    file_path = output_path / "metrics.csv"

    if file_path.exists():
        existing = pd.read_csv(file_path)

        existing = existing[
            ~(
                    (existing["model"] == model_name)
                    & (existing["mode"] == mode)
                    & (existing["n_shot"] == n_shot)
            )
        ]

        updated = pd.concat(
            [existing, pd.DataFrame([metrics_row])],
            ignore_index=True,
        )
    else:
        updated = pd.DataFrame([metrics_row])

    updated.to_csv(file_path, index=False, encoding="utf-8-sig")

    return file_path


def save_confusion_matrix(
        metrics: dict,
        model_name: str,
        mode: str,
        n_shot: int,
        labels: list[str],
        output_dir: str = "results/llm/confusion_matrices",
) -> Path:
    """Sauvegarde la matrice de confusion au format CSV."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    confusion_df = pd.DataFrame(
        metrics["confusion"],
        index=labels,
        columns=labels,
    )

    confusion_df.index.name = "true_label"

    filename = f"{model_name}_{mode}_{n_shot}shot_confusion.csv"
    file_path = output_path / filename

    confusion_df.to_csv(file_path, encoding="utf-8-sig")

    return file_path
