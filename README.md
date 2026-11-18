# Les LLM parlent-ils maghrébin ?

Tokenisation, compréhension et modélisation des darijas algérienne, marocaine et tunisienne.

Projet de groupe — matière Machine Learning, IMT Nord Europe.

## Questions de recherche

1. **Coût de tokenisation** : combien de tokens en plus coûte la darija par rapport au français et à l'arabe standard, selon le tokenizer et l'alphabet (arabe / arabizi) ?
2. **Tokenizer dédié** : un tokenizer BPE entraîné sur la darija réduit-il cet écart ?
3. **Mini-GPT from scratch** : un petit GPT modélise-t-il mieux la darija avec notre tokenizer qu'avec celui de GPT-2 ?
4. **Compréhension** : un petit modèle spécialisé bat-il un grand LLM en zero/few-shot sur l'identification du dialecte et le sentiment ?

## Structure du dépôt

```
darija-llm/
├── data/          # scripts de téléchargement et de nettoyage (les données ne sont PAS versionnées)
├── src/           # code partagé : normalisation, métriques, chargement
├── notebooks/     # un notebook par expérience
├── results/       # CSV de résultats, figures finales, journal des résultats
├── demo/          # application Gradio
└── report/        # rapport et slides
```

## Notebooks

| Notebook | Expérience | Responsable |
| --- | --- | --- |
| `00_donnees.ipynb` | Préparation du corpus | À compléter |
| `01_cout_tokens.ipynb` | Exp. 1 — coût de la darija en tokens | Ammar |
| `02_tokenizer_bpe.ipynb` | Exp. 2 — tokenizer BPE maghrébin | Ammar |
| `03_mini_gpt.ipynb` | Exp. 3 — mini-GPT from scratch et ablations | Ammar |
| `04_classification.ipynb` | Exp. 4 — dialecte et sentiment | À compléter |

## Installation

```bash
git clone https://github.com/<organisation>/darija-llm.git
cd darija-llm
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Sur Google Colab :

```python
!git clone https://github.com/<organisation>/darija-llm.git
%cd darija-llm
!pip install -q -r requirements.txt
```

## Données

Les données brutes et nettoyées vivent sur le Drive partagé, jamais dans Git.
Voir `data/README.md` pour la liste des sources, leurs licences et la façon de les récupérer.

## Reproduire les résultats

1. Préparer les données : `notebooks/00_donnees.ipynb`
2. Lancer les notebooks 01 à 04 dans l'ordre.
3. Les figures et tableaux finaux sont écrits dans `results/`.

## Règles de travail

Voir `CONTRIBUTING.md`.
