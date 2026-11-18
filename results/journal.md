# Journal des résultats

Une ligne par résultat, notée le jour même.

| Date | Exp. | Configuration | Métrique | Valeur | Seed | Commit | Auteur |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | 4 | FLORES+ 3 classes, TF-IDF char 2-5 (20k), nn.Linear, Adam lr=1e-3, batch 64, 20 epochs | Accuracy / F1 macro (test) | 0.891 / 0.891 | 42 | dbd3cb6 | Ines |
| 2026-11-18 | 4 | Même baseline, 60 epochs | Accuracy validation | 0.905 (vs 0.904 à 20 epochs) | 42 | dbd3cb6 | Ines |