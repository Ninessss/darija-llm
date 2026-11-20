
"""Orchestration complète des expériences de classification LLM."""

from pathlib import Path

import pandas as pd

from src.llm.data import (
    select_few_shot_examples,
    save_few_shot_ids,
)
from src.llm.runner import run_evaluation
from src.llm.results import (
    save_predictions,
    save_metrics,
    save_confusion_matrix,
)
from src.llm.prompts import LABELS


def run_all_experiments(
        models: dict,
        dataset: pd.DataFrame,
        output_dir: str = "results/llm",
        seed: int = 42,
) -> pd.DataFrame:
    """
    Exécute toutes les configurations pour les modèles fournis.

    models : dictionnaire {nom: (model, tokenizer)}
    dataset : DataFrame contenant id, text, target et split.

    Le nombre de shots est compté par dialecte :
    0-shot = 0 exemple
    1-shot = 4 exemples
    3-shot = 12 exemples
    5-shot = 20 exemples
    """

    if not models:
        raise ValueError("Aucun modèle n'a été fourni.")

    required_columns = {"id", "text", "target", "split"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            f"Colonnes nécessaires : {sorted(required_columns)}"
        )

    test_df = dataset[dataset["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("Le jeu de test est vide.")

    # Vérifier que les quatre dialectes sont présents dans le test.
    if not set(test_df["target"].unique()).issubset(set(LABELS)):
        raise ValueError("Le jeu de test contient un dialecte inconnu.")

    # Une seule sélection de 20 exemples, commune à tous les modèles.
    fixed_examples = select_few_shot_examples(
        dataset,
        n_examples=5,
        seed=seed,
    )

    save_few_shot_ids(
        fixed_examples,
        output_path=f"{output_dir}/few_shot_ids.json",
    )

    configurations = [
        (0, False),
        (1, False),
        (3, False),
        (5, False),
        (0, True),
        (1, True),
        (3, True),
        (5, True),
    ]

    all_metrics = []

    for model_name, (model, tokenizer) in models.items():
        print(f"\n========== Modèle : {model_name} ==========")

        for n_shot, cot in configurations:
            print(
                f"Configuration : {n_shot}-shot par dialecte | "
                f"{'CoT' if cot else 'Classique'}"
            )

            examples = (
                fixed_examples[:4 * n_shot]
                if n_shot > 0
                else None
            )

            results_df, metrics = run_evaluation(
                model=model,
                tokenizer=tokenizer,
                test_df=test_df,
                examples=examples,
                cot=cot,
                model_name=model_name,
                n_shot_per_class=n_shot,
            )

            mode = "cot" if cot else "standard"

            save_predictions(
                results_df=results_df,
                model_name=model_name,
                mode=mode,
                n_shot=n_shot,
                output_dir=output_dir,
            )

            save_metrics(
                metrics=metrics,
                model_name=model_name,
                mode=mode,
                n_shot=n_shot,
                output_dir=output_dir,
            )

            save_confusion_matrix(
                metrics=metrics,
                model_name=model_name,
                mode=mode,
                n_shot=n_shot,
                labels=LABELS,
                output_dir=f"{output_dir}/confusion_matrices",
            )

            all_metrics.append({
                "model": model_name,
                "mode": mode,
                "n_shot": n_shot,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "invalid_count": metrics["invalid_count"],
                "invalid_rate": metrics["invalid_rate"],
            })

    summary = pd.DataFrame(all_metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_path / "experiment_summary.csv",
        index=False,
        encoding="utf-8-sig",
        )

    return summary
