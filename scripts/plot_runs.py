"""Rassemble les résultats de runs/*/ et produit le tableau et les figures de l'Expérience 3.

    python scripts/plot_runs.py

Sorties : results/exp3_runs.csv et results/figures/exp3_*.png (uniquement des chiffres et des courbes).
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")                     # pas d'écran sur le serveur
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

os.makedirs("results/figures", exist_ok=True)

rows, logs = [], {}
for summ in sorted(glob.glob("runs/*/summary.json")):
    run = os.path.basename(os.path.dirname(summ))
    s = json.load(open(summ, encoding="utf-8"))
    a = s["args"]
    rows.append({
        "run": run, "groupe": a.get("group", ""), "tokenizer": os.path.basename(os.path.normpath(a["tokenizer"])),
        "optimiseur": a["optim"], "lr": a["lr"], "weight_decay": a["wd"], "planning": a["schedule"],
        "batch": a["batch"], "steps": a["steps"], "norm": a["norm"], "seed": a["seed"],
        "params_M": round(s["params_total"] / 1e6, 2), "params_hors_emb_M": round(s["params_hors_embeddings"] / 1e6, 2),
        "val_bpc": round(s["val"]["bpc"], 4), "test_bpc": round(s["test"]["bpc"], 4),
        "caracteres_par_token": round(s["test"]["caracteres_par_token"], 2),
        "val_loss_finale": round(s["val_loss_finale"], 4), "minutes": round(s["temps_entrainement_s"] / 60, 1),
    })
    logs[run] = pd.read_csv(os.path.join(os.path.dirname(summ), "log.csv"))

if not rows:
    raise SystemExit("Aucun entraînement terminé dans runs/.")
table = pd.DataFrame(rows)
table.to_csv("results/exp3_runs.csv", index=False)
print(table.to_string(index=False))

# Palier B : comparaison des tokenizers, en bits par caractère (la seule mesure comparable)
b = table[table.groupe == "B"]
if len(b):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in b.run:
        lg = logs[run]
        axes[0].plot(lg.step, lg.val_loss, label=run.replace("B_", ""))
    axes[0].set_title("Loss de validation par token\n(pas comparable entre tokenizers)")
    axes[0].set_xlabel("step"); axes[0].legend()
    axes[1].bar(b.run.str.replace("B_", ""), b.test_bpc, color="#4C78A8")
    for i, v in enumerate(b.test_bpc):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    axes[1].set_title("Bits par caractère sur le test\n(plus bas = meilleur, comparable)")
    fig.tight_layout(); fig.savefig("results/figures/exp3_tokenizers.png", dpi=200); plt.close(fig)

# Palier C : une figure par ablation, avec la référence en pointillés
ref = table[table.run == "C_reference"]
for groupe in ["optimiseur", "weight_decay", "planning_lr", "batch", "normalisation"]:
    sub = table[table.groupe == groupe]
    if not len(sub):
        continue
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_col = "tokens_vus" if groupe == "batch" else "step"
    if len(ref):
        lg = logs["C_reference"]
        ax.plot(lg[x_col], lg.val_loss, "k--", label=f"référence ({ref.test_bpc.iloc[0]:.3f} bpc)")
    for _, r in sub.iterrows():
        lg = logs[r.run]
        ax.plot(lg[x_col], lg.val_loss, label=f"{r.run.replace('C_', '')} ({r.test_bpc:.3f} bpc)")
    ax.set_xlabel("tokens vus" if groupe == "batch" else "step")
    ax.set_ylabel("loss de validation")
    ax.set_title(f"Ablation : {groupe.replace('_', ' ')}")
    ax.legend(fontsize=8)
    lo = min(lg.val_loss.min() for lg in [logs[r] for r in sub.run]) if len(sub) else None
    if lo is not None:
        ax.set_ylim(lo * 0.95, lo * 1.8)
    fig.tight_layout(); fig.savefig(f"results/figures/exp3_ablation_{groupe}.png", dpi=200); plt.close(fig)

print("Tableau : results/exp3_runs.csv | figures : results/figures/exp3_*.png")
