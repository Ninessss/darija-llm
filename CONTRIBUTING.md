# Règles de travail

## Branches

- `main` contient uniquement du code relu qui tourne.
- Une branche par expérience ou par tâche :
  - `data/nettoyage`
  - `exp1/cout-tokens`
  - `exp2/tokenizer-bpe`
  - `exp3/mini-gpt`
  - `exp4/classification`
  - `demo/gradio`
- On ne pousse jamais directement sur `main` : on ouvre une Pull Request, relue par son binôme.

## Commits

Messages courts, au présent, préfixés par la zone touchée :

```
exp1: ajoute le calcul du ratio de parité
src: normalisation des alef et du tatweel
fix(exp3): checkpoint sauvegardé sur le Drive
```

## Notebooks

- Un notebook par expérience, qui tourne de bout en bout.
- Avant de committer : `Kernel > Restart & Run All`, puis vider les sorties trop lourdes.
- Le code réutilisé par plusieurs notebooks va dans `src/`, pas en copier-coller.

## Résultats

- Chaque résultat chiffré est noté le jour même dans `results/journal.md`.
- Les figures finales vont dans `results/figures/`.

## Ce qui ne va jamais dans Git

- Les données (brutes ou nettoyées)
- Les checkpoints de modèles
- Les clés d'API et tokens Hugging Face (`.env`)
