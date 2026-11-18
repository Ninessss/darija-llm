# Données

Les données ne sont pas versionnées : les licences de MADAR, FLORES+ et Atlaset interdisent de les republier.
Elles sont stockées sur le Drive partagé :

```
MyDrive/darija_project/data/
├── raw/       # fichiers téléchargés tels quels (algerian_dialect/, fassila/, tsac/, tunizi/, madar/)
├── interim/   # unified_v0.parquet, produit par notebooks/00_donnees.ipynb
└── processed/ # corpus nettoyés, un Parquet par usage
```

## Sources

| Source | Dialectes | Usage | Licence / accès | Statut |
| --- | --- | --- | --- | --- |
| MADAR (parallèle) | Alger, Rabat, Fès, Tunis, Sfax, MSA | Exp. 1 (DZ vs MSA) | Recherche interne, non commercial, **non redistribuable** | Formulaire envoyé, lien par mail |
| MADAR Shared Task 2019 | 25 villes + MSA | Exp. 4 (dialecte) | Licence MADAR + CC BY-NC-ND 4.0 (partie Twitter) | Formulaire envoyé, lien par mail |
| FLORES+ | Marocain (ary), tunisien (aeb), MSA, FR, EN | Exp. 1, **évaluation uniquement** | CC BY-SA 4.0, conditions HF acceptées | Accès accepté (HF) |
| Atlaset | Marocain | Exp. 2, 3 | Conditions Hugging Face acceptées | Accès accepté (HF) |
| DarijaBridge | Marocain | Exp. 2, 3 | Apache-2.0 | Public (HF) |
| Algerian Dialect (Mendeley) | Algérien | Exp. 2, 3, 4 (sentiment) | CC BY 4.0 | Téléchargé |
| FASSILA | Algérien | Exp. 3, 4 (sentiment, fake news) | Non précisée (pas de LICENSE dans le dépôt) | Téléchargé |
| TUNIZI | Tunisien (arabizi) | Exp. 2, 3, 4 (sentiment) | CC BY 4.0 | Téléchargé (Zenodo) |
| TSAC | Tunisien | Exp. 4 (sentiment) | LGPL-3.0 | Téléchargé (GitHub fbougares/TSAC) |
| NADI 2025 | Arabe parlé | Non utilisé (audio) | Conditions HF | Accès accepté, non utilisé |

## Points d'attention

- **MADAR ne contient pas le français** (versions anglaise et française non fournies pour raisons de copyright).
  L'Exp. 1 compare au français via **FLORES+** ; l'algérien, absent de FLORES+, est comparé au MSA via MADAR.
- **NADI 2025 est de l'audio** (identification du dialecte parlé). L'Exp. 4 utilise **MADAR Shared Task 2019** à la place.
- **TSAC = environ 17 000 commentaires** (et non ~7 400), dont **~54 % en arabizi** : la colonne `script` permet de séparer les écritures.
- **FLORES+ : jamais à l'entraînement** (`usage = eval_only`) et jamais republié en ligne.
- **FASSILA** : majoritairement traduit ou paraphrasé depuis de l'arabe standard ; seule une partie vient de YouTube (`sub_source`).
- **Atlaset** : contient des données synthétiques ou traduites automatiquement (à filtrer via `sub_source` à l'étape de nettoyage).
- **MADAR** : découper train/test **par phrase** (`group_id`), sinon fuite entre dialectes.

## Format commun (`data/interim/unified_v0.parquet`)

| Colonne | Contenu |
| --- | --- |
| `id` | identifiant unique `dataset:split:numéro` |
| `text` | texte brut (nettoyé à l'étape suivante) |
| `dialect` | `DZ`, `MA`, `TN`, `MSA`, `FR`, `EN` |
| `script` | `arab`, `latin`, `mixed`, `other` (détecté automatiquement) |
| `task` | `sentiment`, `fakenews`, `lm`, `parallel`, `did` |
| `label` | étiquette harmonisée (`neg` / `neu` / `pos` pour le sentiment) |
| `label_orig` | étiquette d'origine |
| `dataset` | nom du dataset |
| `sub_source` | sous-source (ex. youtube / translated pour FASSILA, ville pour MADAR) |
| `orig_split` | split d'origine (`train`, `dev`, `test`, `all`) |
| `group_id` | corpus parallèles : même valeur = même phrase traduite |
| `usage` | `train_ok` ou `eval_only` |

## Règle d'or

Les splits sont figés à la fin de la semaine 2 (seed 42) et ne sont plus jamais modifiés.
