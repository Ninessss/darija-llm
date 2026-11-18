# Données

Les données ne sont pas versionnées. Elles sont stockées sur le Drive partagé :

```
Drive/darija-llm/data/
├── raw/         # fichiers téléchargés tels quels
└── processed/   # corpus nettoyés, un Parquet par usage
```

## Sources

| Source | Dialectes | Usage | Licence / accès | Statut |
| --- | --- | --- | --- | --- |
| MADAR | Alger, Rabat, Fès, Tunis, Sfax (+ autres) | Exp. 1, Exp. 4 (dialecte) | Non commercial, formulaire | À demander |
| FLORES+ | Marocain (ary), tunisien (aeb) | Exp. 1 | CC BY-SA 4.0 | À télécharger |
| Atlaset | Marocain | Exp. 2, 3 | Conditions à accepter | À demander |
| DarijaBridge | Marocain | Exp. 2, 3 | À vérifier | À télécharger |
| Algerian Dialect | Algérien | Exp. 2, 3, 4 (sentiment) | À vérifier | À vérifier |
| FASSILA | Algérien | Exp. 3, 4 | À vérifier | À vérifier |
| TUNIZI | Tunisien (arabizi) | Exp. 2, 3, 4 (sentiment) | À vérifier | À télécharger |
| TSAC | Tunisien (arabe) | Exp. 4 | À vérifier | À télécharger |
| NADI | 21 pays arabes | Exp. 4 (dialecte) | Inscription | À demander |

## Format commun

Un fichier Parquet par usage, avec les colonnes :

| Colonne | Contenu |
| --- | --- |
| `text` | texte nettoyé |
| `dialect` | `dz`, `ma`, `tn` ou `msa` |
| `script` | `arabic` ou `arabizi` |
| `source` | nom du dataset |
| `split` | `train`, `val` ou `test` |
| `label` | label de tâche (sentiment), vide sinon |

## Règle d'or

Les splits sont figés à la fin de la semaine 2 (seed 42) et ne sont plus jamais modifiés.
