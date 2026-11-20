
"""Pipeline d'évaluation des modèles LLM."""

import pandas as pd

from src.llm.prompts import build_prompt
from src.llm.inference import generate_response
from src.llm.evaluation import parse_prediction, evaluate_predictions


def run_evaluation(
        model,
        tokenizer,
        test_df: pd.DataFrame,
        examples: list[dict] | None = None,
        cot: bool = False,
        model_name: str = "unknown",
        n_shot_per_class: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Évalue un modèle sur un jeu de test."""

    if test_df.empty:
        raise ValueError("Le jeu de test est vide.")

    required_columns = {"id", "text", "target"}
    if not required_columns.issubset(test_df.columns):
        raise ValueError(
            f"Colonnes nécessaires : {sorted(required_columns)}"
        )

    examples = examples or []

    if n_shot_per_class is None:
        n_shot_per_class = len(examples) // 4

    results = []

    for _, row in test_df.iterrows():
        prompt = build_prompt(
            text=row["text"],
            examples=examples,
            cot=cot,
        )

        raw_response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=100 if cot else 10,
        )

        prediction = parse_prediction(raw_response)

        results.append({
            "id": str(row["id"]),
            "text": row["text"],
            "true_label": row["target"],
            "raw_response": raw_response,
            "prediction": prediction,
            "is_valid": prediction is not None,
            "model": model_name,
            "mode": "cot" if cot else "standard",
            "n_shot": n_shot_per_class,
        })

    results_df = pd.DataFrame(results)

    metrics = evaluate_predictions(
        y_true=results_df["true_label"].tolist(),
        y_pred=results_df["prediction"].tolist(),
    )

    return results_df, metrics
