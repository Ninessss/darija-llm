"""Rassemble les fine-tunings (runs/F_*) : tableau par run, moyenne ± écart-type par mode, figure.

    python scripts/plot_finetune.py

Sorties : results/exp3_finetune_runs.csv, results/exp3_finetune.csv, results/figures/exp3_finetune.png
(uniquement des chiffres : aucune phrase de MADAR).
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BASELINE = (0.890, 0.002)        # Exp. 4 : TF-IDF + régression logistique, même test (journal du 20/11)
MODES = {"scratch": "sans pré-entraînement", "frozen": "pré-entraîné, gelé", "full": "pré-entraîné, fine-tuning complet"}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "F_*", "summary.json"))):
        s = json.load(open(path, encoding="utf-8"))
        rows.append({"run": os.path.basename(os.path.dirname(path)), "mode": s["args"]["mode"],
                     "seed": s["args"]["seed"], "lr": s["lr"], "params_entraines": s["params_entraines"],
                     "meilleure_epoch": s["meilleure_epoch"], "val_accuracy": s["val"]["accuracy"],
                     "test_accuracy": s["test"]["accuracy"], "test_f1_macro": s["test"]["f1_macro"],
                     **{f"f1_{k}": v for k, v in s["test"]["f1_par_classe"].items()},
                     "minutes": round(s["temps_s"] / 60, 1)})
    if not rows:
        print("Aucun run F_* terminé.")
        return
    runs = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    runs.to_csv(os.path.join(ROOT, "results", "exp3_finetune_runs.csv"), index=False)

    agg = (runs.groupby("mode")
               .agg(n_seeds=("seed", "count"),
                    accuracy=("test_accuracy", "mean"), accuracy_std=("test_accuracy", "std"),
                    f1_macro=("test_f1_macro", "mean"), f1_macro_std=("test_f1_macro", "std"),
                    **{f"f1_{d}": (f"f1_{d}", "mean") for d in ["DZ", "MA", "TN", "MSA"]})
               .reindex([m for m in MODES if m in set(runs["mode"])])
               .round(4))
    agg.to_csv(os.path.join(ROOT, "results", "exp3_finetune.csv"))
    print(agg.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg))
    ax.bar(x, agg["accuracy"], yerr=agg["accuracy_std"].fillna(0), capsize=6, color="#4c78a8")
    for i, v in enumerate(agg["accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.axhline(BASELINE[0], color="#e45756", ls="--", label=f"baseline TF-IDF, Exp. 4 ({BASELINE[0]:.3f})")
    ax.axhline(0.25, color="grey", ls=":", label="hasard (4 classes)")
    ax.set_xticks(x, [MODES[m] for m in agg.index], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy sur le test (moyenne sur les seeds)")
    ax.set_title("Mini-GPT : identification du dialecte (MADAR, 4 classes)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "exp3_finetune.png")
    fig.savefig(out, dpi=150)
    print("Figure :", out)


if __name__ == "__main__":
    main()
