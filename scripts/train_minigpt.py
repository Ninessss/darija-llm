"""Entraîne un mini-GPT (Expérience 3) et enregistre courbes, bits par caractère et textes générés.

Exemples (depuis la racine du dépôt, environnement activé) :

    # Palier A : test de sanité, le modèle doit apprendre 100 phrases par cœur
    python scripts/train_minigpt.py --data ~/darija-data/lm.parquet --tokenizer tokenizers/darija-bpe-16k \\
        --sanity --out runs/sanity --gpu 0

    # Un entraînement normal
    python scripts/train_minigpt.py --data ~/darija-data/lm.parquet --tokenizer tokenizers/darija-bpe-16k \\
        --steps 4000 --out runs/B_bpe16k --group B --gpu 0

Sorties dans --out : log.csv (courbes), summary.json (configuration + résultats), samples.txt (textes générés),
model.pt (si --save_ckpt).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.minigpt import (GPT, GPTConfig, config_dict, encode_split, estimate_loss, evaluate_bpc,  # noqa: E402
                         generate, get_batch, lr_at, make_optimizer)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # données
    p.add_argument("--data", required=True, help="Parquet avec au moins les colonnes text et split (train/val/test)")
    p.add_argument("--dialects", default=None, help="filtre sur la colonne dialect, ex. MA ou MA,DZ,TN")
    p.add_argument("--max_train_texts", type=int, default=None, help="limite le nombre de textes d'entraînement")
    p.add_argument("--max_eval_tokens", type=int, default=2_000_000, help="limite les tokens évalués en val/test")
    p.add_argument("--tokenizer", required=True, help="dossier d'un tokenizer (tokenizers/darija-bpe-16k) ou nom Hugging Face (gpt2)")
    p.add_argument("--cache_dir", default=os.path.expanduser("~/darija-cache"))
    # modèle
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm", default="layer", choices=["layer", "none"])
    # entraînement
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--optim", default="adamw", choices=["sgd", "sgd_momentum", "rmsprop", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--schedule", default="warmup_cosine", choices=["constant", "step", "exp", "warmup_cosine"])
    p.add_argument("--clip", type=float, default=1.0, help="écrêtage de la norme du gradient (0 = aucun)")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    # divers
    p.add_argument("--sanity", action="store_true", help="palier A : apprendre 100 phrases par cœur")
    p.add_argument("--out", required=True)
    p.add_argument("--group", default="", help="étiquette de l'expérience (B, optim, schedule…), pour les figures")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--n_samples", type=int, default=5)
    return p.parse_args()


def load_tokenizer(name):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if os.path.isdir(name):
        tok = PreTrainedTokenizerFast.from_pretrained(name)
    else:
        tok = AutoTokenizer.from_pretrained(name)
    eos = tok.eos_token_id
    if eos is None:
        eos = tok.convert_tokens_to_ids("<|endoftext|>")
    return tok, eos


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "summary.json")):
        print(f"{a.out} déjà terminé (summary.json présent) : on passe.")
        return
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Appareil : {device}" + (f" ({torch.cuda.get_device_name(a.gpu)})" if device != "cpu" else ""))

    # ---------------------------------------------------------------- données
    df = pd.read_parquet(a.data, columns=None)
    if a.dialects:
        df = df[df["dialect"].isin(a.dialects.split(","))]
    df = df[df["text"].astype(str).str.strip() != ""]
    texts = {s: df.loc[df["split"] == s, "text"].astype(str).tolist() for s in ["train", "val", "test"]}
    if a.max_train_texts:
        texts["train"] = texts["train"][:a.max_train_texts]
    if a.sanity:
        texts["train"] = texts["train"][:100]
        texts["val"] = texts["test"] = texts["train"]     # on teste sur les phrases apprises : la loss doit tomber
        a.dropout, a.schedule, a.wd = 0.0, "constant", 0.0
        a.steps = a.steps if a.steps != 3000 else 500
        a.max_train_texts = 100
    print({s: len(t) for s, t in texts.items()}, "textes")

    tok, eos = load_tokenizer(a.tokenizer)
    tok_name = os.path.basename(os.path.normpath(a.tokenizer))
    key = hashlib.md5(f"{os.path.abspath(a.data)}|{a.dialects}|{a.max_train_texts}|{a.sanity}".encode()).hexdigest()[:8]
    data = {}
    for s in ["train", "val", "test"]:
        path = os.path.join(a.cache_dir, f"{tok_name}__{key}__{s}.npz")
        t0 = time.time()
        data[s] = encode_split(tok, texts[s], path, eos)
        print(f"{s:5s} : {len(data[s][0]):>11,} tokens, {data[s][1]:>12,} caractères  ({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------- modèle
    cfg = GPTConfig(vocab_size=len(tok), block_size=a.block, n_layer=a.n_layer, n_head=a.n_head,
                    n_embd=a.n_embd, dropout=a.dropout, norm=a.norm)
    model = GPT(cfg).to(device)
    n_all, n_core = model.n_params(False), model.n_params(True)
    print(f"Paramètres : {n_all / 1e6:.2f} M au total, dont {n_core / 1e6:.2f} M hors embeddings")
    opt = make_optimizer(model, a.optim, a.lr, a.wd)

    # ---------------------------------------------------------------- entraînement
    log, run_loss, n_run = [], 0.0, 0
    t_start = time.time()
    train_tokens = data["train"][0]
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            val_loss = estimate_loss(model, data["val"][0], a.batch, a.block, device)
            row = {
                "step": step,
                "tokens_vus": step * a.batch * a.block,
                "lr": lr_at(step, a.steps, a.lr, a.schedule),
                "train_loss": run_loss / n_run if n_run else float("nan"),
                "val_loss": val_loss,
                "temps_s": round(time.time() - t_start, 1),
            }
            log.append(row)
            pd.DataFrame(log).to_csv(os.path.join(a.out, "log.csv"), index=False)
            print(f"step {step:>6} | lr {row['lr']:.2e} | train {row['train_loss']:.3f} | val {val_loss:.3f} | {row['temps_s']:.0f} s",
                  flush=True)
            run_loss, n_run = 0.0, 0
            if not np.isfinite(val_loss):
                print("La loss a divergé (NaN ou infini) : arrêt.")
                break
        if step == a.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.schedule)
        x, y = get_batch(train_tokens, a.batch, a.block, device, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        run_loss += loss.item()
        n_run += 1
    train_time = time.time() - t_start

    # ---------------------------------------------------------------- évaluation finale
    res = {}
    for s in ["val", "test"]:
        res[s] = evaluate_bpc(model, data[s][0], data[s][1], a.block, device, max_tokens=a.max_eval_tokens)
        print(f"{s:5s} : {res[s]['bpc']:.4f} bits par caractère  (loss par token {res[s]['loss_par_token']:.3f})")

    samples = [generate(model, tok, "", device=device, eos_id=eos) for _ in range(a.n_samples)]
    with open(os.path.join(a.out, "samples.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(f"--- {i + 1}\n{s}" for i, s in enumerate(samples)))

    summary = {
        "args": vars(a),
        "model": config_dict(cfg),
        "params_total": n_all,
        "params_hors_embeddings": n_core,
        "train_tokens_disponibles": int(len(train_tokens)),
        "tokens_vus": int(a.steps * a.batch * a.block),
        "temps_entrainement_s": round(train_time, 1),
        "tokens_par_s": round(a.steps * a.batch * a.block / max(train_time, 1e-9)),
        "val": res["val"],
        "test": res["test"],
        "val_loss_finale": log[-1]["val_loss"],
        "device": device,
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if a.save_ckpt:
        torch.save({"model": model.state_dict(), "config": config_dict(cfg), "tokenizer": a.tokenizer},
                   os.path.join(a.out, "model.pt"))
    print(f"Terminé en {train_time / 60:.1f} min. Résultats dans {a.out}")


if __name__ == "__main__":
    main()
