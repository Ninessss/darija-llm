#!/usr/bin/env bash
# Lance les entraînements de l'Expérience 3 sur les 2 GPU libres (0 et 1), en deux files parallèles.
#
#   bash scripts/run_exp3.sh B ~/darija-data/lm.parquet     # palier B : comparaison des tokenizers
#   bash scripts/run_exp3.sh C ~/darija-data/lm.parquet     # palier C : ablations (cours 2)
#
# À lancer dans tmux : les entraînements continuent après la déconnexion.
# Un entraînement déjà terminé (summary.json présent) est sauté : on peut relancer sans tout refaire.
# GPUS="0 1" par défaut ; pour n'utiliser que le GPU 1 : GPUS="1" bash scripts/run_exp3.sh C ...
# Test rapide (quelques secondes par entraînement) : STEPS=20 EXTRA="--n_embd 64 --block 64" bash scripts/run_exp3.sh C ...

set -u
PALIER=${1:?"palier B ou C"}
DATA=${2:?"chemin du fichier lm.parquet"}
GPUS=(${GPUS:-0 1})
EXTRA=${EXTRA:-}                      # options ajoutées à tous les entraînements
GPT2=${GPT2:-gpt2}
BPE16=tokenizers/darija-bpe-16k
BPE32=tokenizers/darija-bpe-32k

# ------------------------------------------------------------------ liste des entraînements : "nom|groupe|options"
RUNS=()
if [ "$PALIER" = "B" ]; then
  STEPS=${STEPS:-4000}
  COMMON="--steps $STEPS --save_ckpt $EXTRA"
  RUNS+=("B_bpe16k|B|--tokenizer $BPE16 $COMMON")
  RUNS+=("B_gpt2|B|--tokenizer $GPT2 $COMMON --grad_accum 4")
  RUNS+=("B_bpe32k|B|--tokenizer $BPE32 $COMMON")
elif [ "$PALIER" = "C" ]; then
  STEPS=${STEPS:-1500}                # budget réduit et identique pour toutes les ablations
  BASE="--tokenizer $BPE16 --steps $STEPS $EXTRA"
  # Référence : AdamW, lr 1e-3, weight decay 0,1, warmup + cosinus, batch 32, LayerNorm
  RUNS+=("C_reference|reference|$BASE")
  RUNS+=("C_reference_seed43|reference|$BASE --seed 43")
  RUNS+=("C_reference_seed44|reference|$BASE --seed 44")
  # Optimiseurs, tous sans weight decay pour ne comparer que l'optimiseur.
  # Learning rate usuel de chaque optimiseur (SGD a besoin d'un lr bien plus grand qu'Adam).
  RUNS+=("C_opt_sgd|optimiseur|$BASE --optim sgd --lr 0.1 --wd 0")
  RUNS+=("C_opt_sgd_momentum|optimiseur|$BASE --optim sgd_momentum --lr 0.05 --wd 0")
  RUNS+=("C_opt_rmsprop|optimiseur|$BASE --optim rmsprop --lr 3e-4 --wd 0")
  RUNS+=("C_opt_adam|optimiseur|$BASE --optim adam --lr 1e-3 --wd 0")
  RUNS+=("C_opt_adamw|optimiseur|$BASE --optim adamw --lr 1e-3 --wd 0")
  # Weight decay : 0 (= C_opt_adamw) contre 0,1 (= référence) ; on ajoute une valeur forte
  RUNS+=("C_wd_1.0|weight_decay|$BASE --wd 1.0")
  # Plannings de learning rate
  RUNS+=("C_lr_constant|planning_lr|$BASE --schedule constant")
  RUNS+=("C_lr_step|planning_lr|$BASE --schedule step")
  RUNS+=("C_lr_exp|planning_lr|$BASE --schedule exp")
  # Taille de batch, à nombre de tokens vus constant (steps × batch identique)
  RUNS+=("C_batch16|batch|$BASE --batch 16 --steps $((STEPS * 2))")
  RUNS+=("C_batch64|batch|$BASE --batch 64 --steps $((STEPS / 2))")
  # Normalisation : sans LayerNorm
  RUNS+=("C_sans_norm|normalisation|$BASE --norm none")
else
  echo "Palier inconnu : $PALIER (B ou C)"; exit 1
fi

mkdir -p runs
echo "${#RUNS[@]} entraînements, répartis sur les GPU ${GPUS[*]}"

# ------------------------------------------------------------------ une file par GPU, exécutée en arrière-plan
run_queue () {
  local gpu=$1; shift
  for spec in "$@"; do
    IFS='|' read -r name group opts <<< "$spec"
    echo "[GPU $gpu] $(date +%H:%M) début  $name"
    python scripts/train_minigpt.py --data "$DATA" --out "runs/$name" --group "$group" --gpu "$gpu" $opts \
      > "runs/$name.log" 2>&1
    echo "[GPU $gpu] $(date +%H:%M) fin    $name ($(tail -n 1 "runs/$name.log"))"
  done
}

for k in "${!GPUS[@]}"; do
  queue=()
  for i in "${!RUNS[@]}"; do
    if [ $((i % ${#GPUS[@]})) -eq "$k" ]; then queue+=("${RUNS[$i]}"); fi
  done
  run_queue "${GPUS[$k]}" "${queue[@]}" &
done
wait
echo "Tous les entraînements du palier $PALIER sont terminés."
python scripts/plot_runs.py
