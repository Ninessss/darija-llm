
"""Visualisations des résultats des expériences LLM."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.llm.prompts import LABELS


def plot_f1_by_shots(
        metrics_path: str = "results/llm/metrics.csv",
        output_dir: str = "results/figures",
) -> list[Path]:
    """
    Crée un graphique macro-F1 par nombre d'exemples,
    avec une courbe par modèle et un graphique par mode.
    """
    metrics_file = Path(metrics_path)

    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Fichier de métriques introuvable : {metrics_file}"
        )

    df = pd.read_csv(metrics_file)

    required = {"model", "mode", "n_shot", "f1_macro"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Colonnes manquantes : {sorted(required - set(df.columns))}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for mode in ["standard", "cot"]:
        mode_df = df[df["mode"] == mode]

        if mode_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        for model_name, group in mode_df.groupby("model"):
            group = group.sort_values("n_shot")
            ax.plot(
                group["n_shot"],
                group["f1_macro"],
                marker="o",
                label=model_name,
            )

        ax.set_title(f"Macro-F1 selon le nombre d'exemples ({mode})")
        ax.set_xlabel("Nombre d'exemples")
        ax.set_ylabel("Macro-F1")
        ax.set_xticks([0, 1, 3, 5])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Modèle")
        fig.tight_layout()

        file_path = output_path / f"macro_f1_{mode}.png"
        fig.savefig(file_path, dpi=300)
        plt.close(fig)

        generated_files.append(file_path)

    return generated_files


def plot_confusion_matrix(
        predictions_path: str,
        output_path: str,
) -> Path:
    """
    Génère une matrice de confusion visuelle à partir
    d'un fichier CSV de prédictions.
    """
    predictions_file = Path(predictions_path)

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {predictions_file}"
        )

    df = pd.read_csv(predictions_file)

    required = {"true_label", "prediction"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Colonnes manquantes : {sorted(required - set(df.columns))}"
        )

    if df.empty:
        raise ValueError("Le fichier de prédictions est vide.")

    if not df["true_label"].isin(LABELS).all():
        raise ValueError("Le fichier contient des labels réels inconnus.")

    invalid_count = (
            df["prediction"].isna()
            | ~df["prediction"].isin(LABELS)
    ).sum()

    # La matrice classique contient uniquement les quatre classes.
    # Les prédictions invalides sont indiquées séparément dans le titre.
    valid_predictions = df["prediction"].where(
        df["prediction"].isin(LABELS),
        "__invalid__",
    )

    matrix = confusion_matrix(
        df["true_label"],
        valid_predictions,
        labels=LABELS,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(matrix, interpolation="nearest")

    ax.set_title(
        f"Matrice de confusion\n"
        f"Prédictions invalides : {invalid_count}/{len(df)}"
    )
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Dialecte prédit",
        ylabel="Dialecte réel",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(destination, dpi=300)
    plt.close(fig)

    return destination
