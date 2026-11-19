# Données

Les données ne sont **jamais** versionnées : les licences de MADAR, FLORES+ et Atlaset interdisent de les republier.
Elles sont sur le Drive partagé du groupe (lecture seule) :

```
MyDrive/darija_project/data/
├── raw/        # fichiers téléchargés tels quels (algerian_dialect/, fassila/, tsac/, tunizi/, madar/)
├── interim/    # unified_v0.parquet — produit par notebooks/00_donnees.ipynb
└── processed/  # fichiers prêts pour les expériences — produits par notebooks/00b_nettoyage.ipynb
```

Pour y accéder depuis Colab : dans Drive, « Partagés avec moi » → clic droit sur `darija_project` →
« Organiser » → « Ajouter un raccourci » → **Mon Drive**. Le chemin devient `/content/drive/MyDrive/darija_project/...`.

## Fichiers à utiliser (`data/processed/`)

| Fichier | Lignes | Contenu | Pour |
|---|---:|---|---|
| `did.parquet` | 7 956 | **Identification du dialecte, référence** : MADAR, 4 classes équilibrées DZ / MA / TN / MSA, découpage officiel (≈ 1 590 / 200 / 198 par classe) | Ines (Exp. 4), Salsabil (fine-tuning), Nousseiba (LLM) — **même test pour tous** |
| `did_sources.parquet` | 15 000 | Identification du dialecte sur des textes de réseaux sociaux (5 000 par dialecte, écriture arabe) | Exp. 4, analyse complémentaire |
| `sentiment.parquet` | 138 976 | Sentiment `neg` / `neu` / `pos` : Algerian Dialect, FASSILA (DZ), TSAC, TUNIZI (TN) | Exp. 4 |
| `fakenews.parquet` | 9 541 | FASSILA fake news (sens des labels 0/1 à vérifier dans l'article) | bonus |
| `lm.parquet` | 474 638 | Pré-entraînement (MA) : Atlaset authentique + DarijaBridge, train / val / test 98 / 1 / 1 | Salsabil (Exp. 3) |
| `tokenizer_balanced.parquet` | 106 915 | Corpus du tokenizer, **même nombre de mots par dialecte** (558 411), uniquement des splits `train` | Ammar (Exp. 2) |
| `parallel_eval.parquet` | 20 820 | FLORES+ (MA, TN, MSA, FR, EN — **évaluation uniquement**) + MADAR aligné (DZ, MA, TN, MSA, même `group_id` = même phrase) | Ammar (Exp. 1) |
| `human_eval_real.parquet` | 90 | 30 vrais textes par dialecte (test), témoins pour l'évaluation humaine | Salsabil (Exp. 3) |

## Colonnes

| Colonne | Contenu |
|---|---|
| `id` | identifiant **stable** (empreinte du texte) : identique à chaque exécution, sert à partager les phrases de test |
| `text` | texte nettoyé |
| `dialect` | `DZ`, `MA`, `TN`, `MSA`, `FR`, `EN` |
| `script` | `arab`, `latin`, `mixed`, `other` |
| `lang` | `ar`, `arabizi`, `fr`, `en`, `mixed`, `latin_autre` (détection par règles) |
| `task` | `sentiment`, `fakenews`, `lm`, `parallel`, `did` |
| `label` | étiquette harmonisée (`neg`/`neu`/`pos`, ou le dialecte pour `did`) |
| `label_orig` | étiquette d'origine (ou la ville pour MADAR) |
| `dataset`, `sub_source` | source et sous-source (ville pour MADAR, source d'origine pour Atlaset) |
| `split` | `train`, `val`, `test` |
| `group_id` | corpus parallèles : même valeur = même phrase traduite |
| `usage` | `train_ok` ou `eval_only` (FLORES+ : jamais à l'entraînement) |
| `n_words` | nombre de mots |

## Sources, licences et statut

| Source | Dialecte | Licence / accès | Statut |
|---|---|---|---|
| MADAR Shared Task 2019 (corpus 26 villes) | Alger, Rabat, Fès, Tunis, Sfax, MSA | Recherche interne, **non redistribuable** | ✅ reçu le 19/11 |
| FLORES+ | MA (ary), TN (aeb), MSA, FR, EN | CC BY-SA 4.0, conditions HF : **évaluation uniquement** | ✅ |
| Atlaset | MA | Conditions Hugging Face | ✅ |
| DarijaBridge | MA | Apache-2.0 | ✅ |
| Algerian Dialect (Mendeley) | DZ | CC BY 4.0 | ✅ |
| FASSILA | DZ | Non précisée | ✅ |
| TUNIZI (Zenodo) | TN (arabizi) | CC BY 4.0 | ✅ |
| TSAC | TN | LGPL-3.0 | ✅ |
| NADI 2025 | — | Conditions HF | ❌ non utilisé : c'est de l'audio |

## Choix de nettoyage (à citer dans le rapport)

- **Normalisation** : NFKC, suppression des URL, e-mails, `@pseudos`, numéros, HTML ; lettres répétées 3+ → 2 ;
  diacritiques et tatweel supprimés, alef أ إ آ ٱ → ا ; emojis gardés **seulement** pour le sentiment.
- **Atlaset** : 676 595 textes **synthétiques ou traduits automatiquement** exclus (jeu généré par Gemini, NQ-Open-AYA,
  sharegpt, flan, alpaca, cot, wizardlm, oasst…) → 497 603 textes authentiques gardés.
- **DarijaBridge** : 1 235 091 lignes mais seulement **7 587 phrases uniques** (corpus de traduction).
- **Dédoublonnage** : 13 785 doublons ; 477 textes à labels contradictoires ; 2 003 textes identiques à FLORES+ retirés de l'entraînement ;
  46 fuites train/test retirées (sentiment).
- **Découpage** (seed 42) : test d'origine gardé (FASSILA, TSAC, MADAR), sinon 80 / 10 / 10 stratifié.

## Points d'attention

- **MADAR ne contient pas le français** : l'Exp. 1 compare au français via FLORES+ ; l'algérien (absent de FLORES+) se compare au MSA via MADAR.
- **Déséquilibre** : ≈ 118 M de mots marocains contre ≈ 0,6 M pour DZ et TN → d'où `tokenizer_balanced.parquet`.
- **Écriture** : dans le corpus du tokenizer, le tunisien est surtout en **arabizi** (≈ 435 000 mots sur 626 000, TUNIZI),
  le marocain et l'algérien surtout en arabe. À prendre en compte pour l'Exp. 2.
- **TSAC** : ≈ 17 000 commentaires, dont ≈ 54 % en lettres latines.
- **FASSILA** : majoritairement traduit / paraphrasé depuis l'arabe standard ; très peu de positifs.
- **MADAR** : phrases courtes du domaine du tourisme ; une seule ville algérienne.

## Règle d'or

Données **figées le 24 novembre** (seed 42). Après cette date, plus aucune modification des fichiers de `processed/`.
