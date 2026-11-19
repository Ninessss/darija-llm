# Journal des résultats

Une ligne par résultat, notée le jour même.

| Date | Exp. | Configuration | Métrique | Valeur | Seed | Commit | Auteur |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18 | 4 | FLORES+ 3 classes, TF-IDF char 2-5 (20k), nn.Linear, Adam lr=1e-3, batch 64, 20 epochs | Accuracy / F1 macro (test) | 0.891 / 0.891 | 42 | dbd3cb6 | Ines |
| 2026-11-18 | 4 | Même baseline, 60 epochs | Accuracy validation | 0.905 (vs 0.904 à 20 epochs) | 42 | dbd3cb6 | Ines |
| 2026-11-18 | 4 | MLP 128 ReLU, AdamW, weight decay 0 / 0.01 / 0.1, arrêt anticipé | Accuracy validation | 0.914 dans les 3 cas (meilleure epoch 5, 5, 3) | 42 | dbd3cb6 | Ines |
| 2026-11-18 | 4 | MLP 128 ReLU, AdamW, weight decay 0, arrêt anticipé (epoch 5) | Accuracy / F1 macro (test) | 0.896 / 0.896 | 42 | dbd3cb6 | Ines |
| 2026-11-19 | 4 | MADAR did.parquet 4 classes (DZ/MA/TN/MSA), TF-IDF char 2-5 (20k), nn.Linear, Adam lr=1e-3, batch 64, 20 epochs | Accuracy / F1 macro (test, 792 phrases) | 0.894 / 0.895 | 42 | 9dcf21a | Ines |
| 2026-11-19 | 4 | MADAR, MLP 128 ReLU, AdamW, weight decay 0 / 0.01 / 0.1, arrêt anticipé | Accuracy validation | 0.902 dans les 3 cas (meilleure epoch 7) | 42 | 9dcf21a | Ines |
| 2026-11-19 | 4 | MADAR, MLP 128 ReLU, weight decay 0, arrêt anticipé (epoch 7) | Accuracy / F1 macro (test) | 0.886 / 0.887 | 42 | 9dcf21a | Ines |