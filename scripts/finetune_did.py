"""Fine-tuning du mini-GPT pour l'identification du dialecte (DZ / MA / TN / MSA) sur did.parquet (MADAR).

Question : le pré-entraînement sur la darija aide-t-il à reconnaître le dialecte ?
On compare trois façons d'utiliser le même modèle (même architecture, même tokenizer) :

    --mode scratch : poids tirés au hasard, tout est entraîné sur did.parquet (aucun pré-entraînement) ;
    --mode frozen  : poids pré-entraînés GELÉS, seule la tête de classification apprend (« linear probe ») ;
    --mode full    : poids pré-entraînés, tout le modèle est ré-entraîné (fine-tuning complet).

Le test est celui de l'Exp. 4 (même did.parquet) : les résultats se comparent directement
à la baseline TF-IDF (accuracy 0,890 ± 0,002) et aux LLM de l'Exp. 5.

Exemple (depuis la racine du dépôt) :
    python scripts/finetune_did.py --data ~/darija-data/did.parquet --ckpt runs/B_bpe16k/model.pt \\
        --mode full --seed 42 --out runs/F_full_s42 --gpu 0

Sorties dans --out : log.csv (courbes), summary.json (configuration + résultats), predictions.csv
(id, vrai label, prédiction : **jamais le texte**, MADAR n'est pas redistribuable).
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.minigpt import GPT, GPTClassifier, GPTConfig, load_model, load_tokenizer, make_optimizer  # noqa: E402

LABELS = ["DZ", "MA", "TN", "MSA"]
DEFAULT_LR = {"scratch": 3e-4, "frozen": 1e-3, "full": 1e-4}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="did.parquet (colonnes id, text, label, split)")
    p.add_argument("--ckpt", required=True,
                   help="model.pt pré-entraîné ; en mode scratch, sert seulement à reprendre la même architecture et le même tokenizer")
    p.add_argument("--mode", required=True, choices=["scratch", "frozen", "full"])
    p.add_argument("--max_len", type=int, default=64, help="longueur maximale d'une phrase, en tokens")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3, help="arrêt anticipé : epochs sans progrès sur la validation")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="par défaut : 3e-4 (scratch), 1e-3 (frozen), 1e-4 (full)")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def encode(tok, texts, max_len, pad_id):
    """Textes -> tenseurs (ids, masque), remplissage à droite. Renvoie aussi la part de phrases tronquées."""
    ids = tok(list(texts), add_special_tokens=False)["input_ids"]
    truncated = float(np.mean([len(x) > max_len for x in ids]))
    ids = [x[:max_len] or [pad_id] for x in ids]
    T = max(len(x) for x in ids)
    X = torch.full((len(ids), T), pad_id, dtype=torch.long)
    M = torch.zeros((len(ids), T), dtype=torch.bool)
    for i, x in enumerate(ids):
        X[i, :len(x)] = torch.tensor(x)
        M[i, :len(x)] = True
    return X, M, truncated


def report(y_true, y_pred, n_classes):
    """Accuracy, F1 macro et matrice de confusion (lignes = vraie classe, colonnes = prédiction)."""
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    tp = np.diag(conf).astype(float)
    prec = tp / np.maximum(conf.sum(0), 1)
    rec = tp / np.maximum(conf.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"accuracy": float(tp.sum() / max(conf.sum(), 1)), "f1_macro": float(f1.mean()),
            "f1_par_classe": {LABELS[i]: round(float(f1[i]), 4) for i in range(n_classes)},
            "confusion": conf.tolist()}


@torch.no_grad()
def predict(model, X, M, device, batch=256):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        logits = model(X[i:i + batch].to(device), M[i:i + batch].to(device))
        preds.append(logits.argmax(-1).cpu())
    model.train()
    return torch.cat(preds).numpy()


def main():
    a = parse_args()
    lr = a.lr if a.lr is not None else DEFAULT_LR[a.mode]
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "summary.json")):
        print(f"{a.out} déjà terminé (summary.json présent) : on passe.")
        return
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Appareil : {device} | mode {a.mode} | lr {lr} | seed {a.seed}")

    # ---------------------------------------------------------------- modèle
    if a.mode == "scratch":
        ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=True)
        gpt = GPT(GPTConfig(**ckpt["config"]))                       # même architecture, poids au hasard
        tok, _ = load_tokenizer(ckpt["tokenizer"])
        del ckpt
    else:
        gpt, tok, _ = load_model(a.ckpt, device="cpu")
    gpt.train()
    model = GPTClassifier(gpt, len(LABELS), dropout=a.dropout).to(device)
    if a.mode == "frozen":
        for p in model.gpt.parameters():
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînés : {n_train / 1e6:.3f} M")

    # ---------------------------------------------------------------- données
    df = pd.read_parquet(a.data, columns=["id", "text", "label", "split"])
    df = df[df["label"].isin(LABELS)]
    lab2id = {l: i for i, l in enumerate(LABELS)}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    data, trunc = {}, {}
    for s in ["train", "val", "test"]:
        part = df[df["split"] == s].reset_index(drop=True)
        X, M, trunc[s] = encode(tok, part["text"].astype(str), a.max_len, pad_id)
        y = torch.tensor(part["label"].map(lab2id).values, dtype=torch.long)
        data[s] = (X, M, y, part["id"].tolist())
        print(f"{s:5s} : {len(part):>5} phrases, {trunc[s]:.1%} tronquées à {a.max_len} tokens")

    # ---------------------------------------------------------------- entraînement (+ arrêt anticipé sur la validation)
    opt = make_optimizer(model, "adamw", lr, a.wd)
    X, M, y, _ = data["train"]
    g = torch.Generator().manual_seed(a.seed)
    best_acc, best_state, best_epoch, bad, log = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        perm = torch.randperm(len(X), generator=g)
        losses = []
        for i in range(0, len(X), a.batch):
            b = perm[i:i + a.batch]
            logits = model(X[b].to(device), M[b].to(device))
            loss = F.cross_entropy(logits, y[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val_pred = predict(model, data["val"][0], data["val"][1], device)
        val_acc = float((val_pred == data["val"][2].numpy()).mean())
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc,
                    "temps_s": round(time.time() - t0, 1)})
        pd.DataFrame(log).to_csv(os.path.join(a.out, "log.csv"), index=False)
        print(f"epoch {epoch:>2} | train loss {np.mean(losses):.4f} | val acc {val_acc:.4f} | {time.time() - t0:.0f} s",
              flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch, bad = val_acc, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Arrêt anticipé : pas de progrès depuis {a.patience} epochs.")
                break

    # ---------------------------------------------------------------- évaluation finale (meilleure epoch)
    model.load_state_dict(best_state)
    res = {}
    for s in ["val", "test"]:
        Xs, Ms, ys, ids = data[s]
        pred = predict(model, Xs, Ms, device)
        res[s] = report(ys.numpy(), pred, len(LABELS))
        if s == "test":
            pd.DataFrame({"id": ids, "label": [LABELS[i] for i in ys.numpy()],
                          "prediction": [LABELS[i] for i in pred]}).to_csv(
                os.path.join(a.out, "predictions.csv"), index=False)
    print(f"test : accuracy {res['test']['accuracy']:.4f} | F1 macro {res['test']['f1_macro']:.4f} "
          f"(meilleure epoch : {best_epoch})")

    summary = {"args": vars(a), "lr": lr, "params_entraines": n_train, "meilleure_epoch": best_epoch,
               "epochs_faites": len(log), "temps_s": round(time.time() - t0, 1),
               "part_tronquee": trunc, "labels": LABELS, "val": res["val"], "test": res["test"],
               "device": device}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Résultats dans {a.out}")


if __name__ == "__main__":
    main()
