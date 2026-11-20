"""Mini-GPT from scratch pour l'Expérience 3 (style nanoGPT, écrit avec torch.nn).

Contenu :
- GPT : le modèle (embeddings, blocs Transformer causaux, tête de prédiction) ;
- make_optimizer / lr_at : les optimiseurs et plannings de learning rate comparés (cours 2) ;
- encode_split : texte -> suite de tokens, mise en cache sur disque ;
- evaluate_bpc : bits par caractère sur un texte complet (comparable entre tokenizers) ;
- generate : génération de texte pour l'évaluation humaine.
"""

import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- modèle

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256      # longueur du contexte, en tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    norm: str = "layer"        # "layer" (LayerNorm) ou "none" (ablation : sans normalisation)


def _norm(cfg):
    return nn.LayerNorm(cfg.n_embd) if cfg.norm == "layer" else nn.Identity()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, C / n_head)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # attention causale : chaque token ne voit que les tokens précédents
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Bloc Transformer « pre-norm » : x + Attention(Norm(x)), puis x + MLP(Norm(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = _norm(cfg), CausalSelfAttention(cfg)
        self.ln2, self.mlp = _norm(cfg), MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = _norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # « weight tying » : même matrice en entrée et en sortie
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    def features(self, idx):
        """États cachés du dernier bloc, après la normalisation finale : (B, T, n_embd).

        Utilisé par le fine-tuning (classification) : ce sont les représentations apprises au pré-entraînement.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, idx, targets=None, reduction="mean"):
        logits = self.head(self.features(idx))           # logits bruts, pas de softmax (CrossEntropyLoss s'en charge)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)
        return logits, loss


# ----------------------------------------------------------------------------- optimisation (cours 2)

def make_optimizer(model, name, lr, weight_decay):
    """Le weight decay ne s'applique qu'aux matrices (pas aux biais ni aux paramètres de LayerNorm)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)       # weight decay « L2 » classique
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))   # weight decay « découplé »
    raise ValueError(f"optimiseur inconnu : {name}")


def lr_at(step, max_steps, base_lr, schedule, warmup_frac=0.05, min_ratio=0.01):
    """Learning rate à l'étape `step` selon le planning choisi."""
    if schedule == "constant":
        return base_lr
    if schedule == "step":            # divisé par 10 à 50 % puis à 75 % de l'entraînement
        return base_lr * (0.1 ** ((step >= 0.5 * max_steps) + (step >= 0.75 * max_steps)))
    if schedule == "exp":             # décroissance exponentielle jusqu'à base_lr * min_ratio
        return base_lr * (min_ratio ** (step / max_steps))
    if schedule == "warmup_cosine":   # montée linéaire puis décroissance en cosinus
        warmup = max(1, int(warmup_frac * max_steps))
        if step < warmup:
            return base_lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    raise ValueError(f"planning inconnu : {schedule}")


# ----------------------------------------------------------------------------- données

def encode_split(tokenizer, texts, cache_path, eos_id, batch=2000):
    """Tokenise une liste de textes, les met bout à bout séparés par un token de fin, et met le résultat en cache.

    Renvoie (tokens, n_caracteres). n_caracteres = nombre de caractères du texte encodé
    (un séparateur compté par texte), utilisé pour les bits par caractère.
    """
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["tokens"], int(data["n_chars"])
    chunks, n_chars = [], 0
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        for ids in tokenizer(part, add_special_tokens=False)["input_ids"]:
            chunks.append(np.asarray(ids + [eos_id], dtype=np.int32))
        n_chars += sum(len(t) + 1 for t in part)
    tokens = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, tokens=tokens, n_chars=n_chars)
    return tokens, n_chars


def get_batch(tokens, batch_size, block_size, device, rng):
    """Tire `batch_size` fenêtres au hasard dans la suite de tokens (x) et leur décalage d'un cran (y)."""
    ix = rng.integers(0, len(tokens) - block_size - 1, size=batch_size)
    x = torch.from_numpy(np.stack([tokens[i:i + block_size] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([tokens[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64))
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------------- évaluation

@torch.no_grad()
def evaluate_bpc(model, tokens, n_chars, block_size, device, batch_size=32, max_tokens=None):
    """Bits par caractère sur toute la suite `tokens`.

    Le texte est découpé en fenêtres successives sans recouvrement : chaque token (sauf le tout
    premier) est prédit exactement une fois. BPC = somme des -log2 p(token) / nombre de caractères.
    Contrairement à la perplexité par token, le BPC ne dépend pas du découpage : il permet de
    comparer deux tokenizers.
    """
    model.eval()
    if max_tokens is not None and len(tokens) > max_tokens:
        n_chars = n_chars * max_tokens / len(tokens)     # proportion de texte évaluée
        tokens = tokens[:max_tokens]
    total_nll, total_tok = 0.0, 0
    starts = list(range(0, len(tokens) - 1, block_size))
    for i in range(0, len(starts), batch_size):
        xs, ys = [], []
        for s in starts[i:i + batch_size]:
            x = tokens[s:s + block_size]
            y = tokens[s + 1:s + 1 + block_size]
            x = x[:len(y)]
            pad = block_size - len(y)
            xs.append(np.pad(x, (0, pad)))
            ys.append(np.pad(y, (0, pad), constant_values=-100))   # -100 = ignoré par cross_entropy
        x = torch.from_numpy(np.stack(xs).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction="sum")
        total_nll += nll.item()
        total_tok += int((y != -100).sum().item())
    model.train()
    return {
        "bpc": total_nll / math.log(2) / n_chars,
        "loss_par_token": total_nll / max(total_tok, 1),
        "tokens": total_tok,
        "caracteres": n_chars,
        "caracteres_par_token": n_chars / max(total_tok, 1),
    }


@torch.no_grad()
def estimate_loss(model, tokens, batch_size, block_size, device, n_batches=20, seed=0):
    """Loss moyenne sur quelques fenêtres tirées au hasard (toujours les mêmes grâce à la graine)."""
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(tokens, batch_size, block_size, device, rng)
        losses.append(model(x, y)[1].item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60, temperature=0.8, top_k=50, device="cpu", eos_id=None):
    """Génère une suite au texte `prompt` (échantillonnage avec température et top-k)."""
    model.eval()
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] or [eos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if eos_id is not None and nxt.item() == eos_id:
            break
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)


def config_dict(cfg):
    return asdict(cfg)


# ----------------------------------------------------------------------------- rechargement

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_tokenizer(name):
    """Charge un tokenizer : dossier du dépôt (tokenizers/darija-bpe-16k) ou nom Hugging Face (gpt2).

    Renvoie (tokenizer, eos_id).
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    path = name if os.path.isabs(name) else os.path.join(REPO_ROOT, name)
    if os.path.isdir(path):
        tok = PreTrainedTokenizerFast.from_pretrained(path)
    elif os.path.isdir(name):
        tok = PreTrainedTokenizerFast.from_pretrained(name)
    else:
        tok = AutoTokenizer.from_pretrained(name)
    eos = tok.eos_token_id
    if eos is None:
        eos = tok.convert_tokens_to_ids("<|endoftext|>")
    return tok, eos


def load_model(ckpt_path, device="cpu", tokenizer=None):
    """Recharge un mini-GPT sauvegardé par train_minigpt.py --save_ckpt.

    Exemple :
        model, tok, eos = load_model("runs/B_bpe16k/model.pt", device="cuda:0")
        print(generate(model, tok, "واحد النهار", device="cuda:0", eos_id=eos))

    `tokenizer` permet de forcer un autre chemin de tokenizer que celui enregistré dans le checkpoint
    (utile si le dépôt a été cloné ailleurs).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok, eos = load_tokenizer(tokenizer or ckpt["tokenizer"])
    assert len(tok) == cfg.vocab_size, (
        f"Le tokenizer ({len(tok)} tokens) ne correspond pas au modèle ({cfg.vocab_size}) : "
        "vérifie le chemin du tokenizer.")
    return model, tok, eos


# ----------------------------------------------------------------------------- classification (fine-tuning)

class GPTClassifier(nn.Module):
    """Mini-GPT + tête de classification, pour l'identification du dialecte.

    La phrase passe dans le GPT ; on fait la moyenne des états cachés de ses vrais tokens
    (le remplissage est ignoré grâce au masque), puis une couche linéaire donne un score par classe.
    Le padding est placé à droite : avec l'attention causale, les vrais tokens ne le voient jamais.
    """

    def __init__(self, gpt: GPT, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.gpt = gpt
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(gpt.cfg.n_embd, n_classes)

    def forward(self, idx, mask):
        h = self.gpt.features(idx)                                   # (B, T, C)
        m = mask.unsqueeze(-1).to(h.dtype)                           # (B, T, 1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)            # moyenne sur les vrais tokens
        return self.classifier(self.drop(pooled))                    # logits bruts (B, n_classes)
