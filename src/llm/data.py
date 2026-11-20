
"""Chargement et préparation des données pour l'expérience LLM."""

from pathlib import Path
import json
import random

import pandas as pd


REQUIRED_COLUMNS = ["id", "text", "dialect", "split"]

DIALECT_MAPPING = {
    "DZ": "algerian",
    "MA": "moroccan",
    "TN": "tunisian",
    "MSA": "standard_arabic",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Charge et vérifie le fichier Parquet DID."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_parquet(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    df = df.copy()

    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Les IDs doivent être présents et uniques.")

    if df["text"].isna().any():
        raise ValueError("Certaines phrases sont manquantes.")

    if df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Certaines phrases sont vides.")

    # Normalisation des codes dialectaux
    df["dialect"] = df["dialect"].astype(str).str.strip().str.upper()

    if not df["dialect"].isin(DIALECT_MAPPING).all():
        unknown = sorted(set(df["dialect"]) - set(DIALECT_MAPPING))
        raise ValueError(f"Dialectes inconnus : {unknown}")

    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Certains splits ne sont pas reconnus.")

    df["id"] = df["id"].astype(str)
    df["target"] = df["dialect"].map(DIALECT_MAPPING)

    return df


def prepare_evaluation_data(
        df: pd.DataFrame,
        split: str = "test",
) -> pd.DataFrame:
    """Sélectionne les données d'un split précis."""

    if split not in ["train", "val", "test"]:
        raise ValueError("Split invalide.")

    selected = df[df["split"] == split].copy()

    return selected.reset_index(drop=True)


def select_few_shot_examples(
        df: pd.DataFrame,
        n_examples: int = 5,
        seed: int = 42,
) -> list[dict]:
    """
    Sélectionne 1, 3 ou 5 exemples PAR dialecte depuis train.

    Les exemples sont mélangés globalement pour éviter de regrouper
    les exemples d'un même dialecte.
    """

    if n_examples not in [1, 3, 5]:
        raise ValueError("n_examples doit être 1, 3 ou 5.")

    required = {"id", "text", "target", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes nécessaires : {sorted(required)}")

    train_df = (
        df[df["split"] == "train"]
        .copy()
        .drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )

    if train_df.empty:
        raise ValueError("Aucune donnée disponible dans train.")

    rng = random.Random(seed)
    labels = list(DIALECT_MAPPING.values())

    samples_by_label = {}

    for label in labels:
        candidates = train_df[train_df["target"] == label]

        if len(candidates) < 5:
            raise ValueError(
                f"Moins de 5 exemples disponibles pour {label}."
            )

        indices = rng.sample(list(candidates.index), 5)

        samples_by_label[label] = [
            train_df.loc[index].to_dict()
            for index in indices
        ]

    selected = []

    for round_index in range(5):
        round_labels = labels.copy()
        rng.shuffle(round_labels)

        for label in round_labels:
            row = samples_by_label[label][round_index]

            selected.append({
                "id": str(row["id"]),
                "text": str(row["text"]),
                "label": row["target"],
            })

    return selected[:4 * n_examples]


def save_few_shot_ids(
        examples: list[dict],
        output_path: str | Path = "results/llm/few_shot_ids.json",
) -> Path:
    """Sauvegarde uniquement les IDs des exemples few-shot."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ids = [example["id"] for example in examples]

    with path.open("w", encoding="utf-8") as file:
        json.dump(ids, file, indent=2, ensure_ascii=False)

    return path


def save_test_ids(
        test_df: pd.DataFrame,
        output_path: str | Path = "results/llm/test_ids.json",
) -> Path:
    """Sauvegarde uniquement les IDs du jeu de test."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ids = test_df["id"].astype(str).tolist()

    with path.open("w", encoding="utf-8") as file:
        json.dump(ids, file, indent=2, ensure_ascii=False)

    return path


def select_test_subset(
        df: pd.DataFrame,
        n_per_class: int = 125,
        seed: int = 42,
) -> pd.DataFrame:
    """Sélectionne un sous-échantillon équilibré du test."""

    test_df = df[df["split"] == "test"].copy()
    rng = random.Random(seed)
    selected_indices = []

    for label in DIALECT_MAPPING.values():
        candidates = test_df[test_df["target"] == label]

        if len(candidates) < n_per_class:
            raise ValueError(
                f"Pas assez de phrases de test pour {label}."
            )

        selected_indices.extend(
            rng.sample(list(candidates.index), n_per_class)
        )

    selected = test_df.loc[selected_indices].copy()

    return selected.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)
