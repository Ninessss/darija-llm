"""Évaluation humaine du mini-GPT (Expérience 3) : textes générés mélangés à de vrais textes, notés à l'aveugle.

Deux étapes :

1) Préparer les fiches (sur le serveur, là où est le modèle) :
    python scripts/human_eval.py prepare --ckpt runs/B_bpe16k/model.pt \\
        --real ~/darija-data/human_eval_real.parquet --dialects MA --n 30 --out runs/human_eval

   -> runs/human_eval/fiche_evaluateur.csv : textes mélangés et numérotés, colonnes de notes vides
      (à copier une fois par évaluateur, à ouvrir dans Excel ou Google Sheets) ;
   -> runs/human_eval/cle.csv : pour chaque numéro, « genere » ou « reel » (à NE PAS donner aux évaluateurs).

   Les fiches contiennent des textes des jeux de données : elles restent hors de Git (dossier runs/).

2) Calculer les résultats quand les fiches remplies sont revenues :
    python scripts/human_eval.py score --key runs/human_eval/cle.csv --sheets runs/human_eval/rempli_*.csv

   -> results/exp3_human_eval.csv (uniquement des chiffres : notes moyennes, taux de détection).

Notes demandées pour chaque texte :
    naturel (1 à 5)       : est-ce que ça sonne comme de la vraie darija ?
    comprehensible (1 à 5) : est-ce que la phrase a un sens ?
    humain_ou_machine     : H (écrit par un humain) ou M (généré par une machine)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.minigpt import generate, load_model  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NOTE_COLS = ["naturel_1_a_5", "comprehensible_1_a_5", "humain_ou_machine_H_M"]


def prepare(a):
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    device = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    model, tok, eos = load_model(a.ckpt, device=device)
    dialects = a.dialects.split(",")

    real = pd.read_parquet(a.real)
    col = "dialect" if "dialect" in real.columns else "label"
    items = []
    for d in dialects:
        r = real[real[col] == d]
        if len(r) < a.n:
            print(f"Attention : seulement {len(r)} vrais textes {d} (au lieu de {a.n}).")
        for t in r.sample(min(a.n, len(r)), random_state=a.seed)["text"]:
            items.append({"source": "reel", "dialect": d, "text": str(t)})

        # Textes générés, de longueur comparable aux vrais (entre min_words et max_words mots).
        # Le modèle actuel ne connaît que le marocain : pour un autre dialecte, il n'y a pas de condition.
        kept, tries = 0, 0
        while kept < a.n and tries < a.n * 50:
            tries += 1
            t = generate(model, tok, "", max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                         top_k=a.top_k, device=device, eos_id=eos).strip()
            n_words = len(t.split())
            if n_words < a.min_words:
                continue
            if n_words > a.max_words:
                t = " ".join(t.split()[:a.max_words])
            items.append({"source": "genere", "dialect": d, "text": t})
            kept += 1
        print(f"{d} : {kept} textes générés gardés sur {tries} essais")

    df = pd.DataFrame(items).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    df.insert(0, "numero", np.arange(1, len(df) + 1))
    os.makedirs(a.out, exist_ok=True)
    sheet = df[["numero", "text"]].rename(columns={"text": "texte"})
    for c in NOTE_COLS:
        sheet[c] = ""
    # utf-8-sig : Excel affiche correctement l'arabe
    sheet.to_csv(os.path.join(a.out, "fiche_evaluateur.csv"), index=False, encoding="utf-8-sig")
    df[["numero", "source", "dialect"]].to_csv(os.path.join(a.out, "cle.csv"), index=False)
    print(f"{len(df)} textes ({(df.source == 'genere').sum()} générés, {(df.source == 'reel').sum()} réels)")
    print(f"Fiche : {os.path.join(a.out, 'fiche_evaluateur.csv')}  |  Clé (à garder pour soi) : {os.path.join(a.out, 'cle.csv')}")


def score(a):
    key = pd.read_csv(a.key)
    files = sorted(f for pattern in a.sheets for f in glob.glob(os.path.expanduser(pattern)))
    if not files:
        sys.exit("Aucune fiche remplie trouvée.")
    notes = []
    for i, f in enumerate(files, 1):
        s = pd.read_csv(f, encoding="utf-8-sig")
        s["evaluateur"] = i
        notes.append(s[["numero", "evaluateur"] + NOTE_COLS])
    n = pd.concat(notes).merge(key, on="numero")
    for c in NOTE_COLS[:2]:
        n[c] = pd.to_numeric(n[c], errors="coerce")
    guess = n[NOTE_COLS[2]].astype(str).str.strip().str.upper().str[:1]
    n["devine_machine"] = guess.map({"M": True, "H": False})
    n["correct"] = n["devine_machine"] == n["source"].eq("genere")

    res = (n.groupby(["dialect", "source"])
             .agg(notes=("numero", "size"),
                  naturel=("naturel_1_a_5", "mean"), naturel_std=("naturel_1_a_5", "std"),
                  comprehensible=("comprehensible_1_a_5", "mean"),
                  comprehensible_std=("comprehensible_1_a_5", "std"),
                  juge_machine=("devine_machine", "mean"))
             .round(3).reset_index())
    print(res.to_string(index=False))
    acc = n.dropna(subset=["devine_machine"])["correct"].mean()
    print(f"\n{len(files)} évaluateurs. Détection humain / machine correcte : {acc:.1%} "
          "(50 % = les évaluateurs ne font pas la différence).")
    out = os.path.join(ROOT, "results", "exp3_human_eval.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.assign(n_evaluateurs=len(files), detection_correcte=round(acc, 3)).to_csv(out, index=False)
    print("Résultats :", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="générer les textes et créer les fiches")
    pp.add_argument("--ckpt", required=True)
    pp.add_argument("--real", required=True, help="human_eval_real.parquet")
    pp.add_argument("--dialects", default="MA", help="MA par défaut : le modèle n'a appris que le marocain")
    pp.add_argument("--n", type=int, default=30, help="textes par dialecte et par source")
    pp.add_argument("--min_words", type=int, default=8)
    pp.add_argument("--max_words", type=int, default=40)
    pp.add_argument("--max_new_tokens", type=int, default=80)
    pp.add_argument("--temperature", type=float, default=0.8)
    pp.add_argument("--top_k", type=int, default=50)
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--gpu", type=int, default=0)
    pp.add_argument("--out", default="runs/human_eval")
    ps = sub.add_parser("score", help="calculer les résultats à partir des fiches remplies")
    ps.add_argument("--key", required=True)
    ps.add_argument("--sheets", nargs="+", required=True)
    a = p.parse_args()
    prepare(a) if a.cmd == "prepare" else score(a)


if __name__ == "__main__":
    main()
