#!/usr/bin/env bash
# Fine-tuning du mini-GPT sur l'identification du dialecte (did.parquet, MADAR) :
# 3 modes (scratch / frozen / full) × 3 seeds, répartis sur les 2 GPU libres, puis tableau + figure.
#
#   bash scripts/run_finetune.sh ~/darija-data/did.parquet
#
# CKPT=runs/B_bpe16k/model.pt par défaut (le modèle pré-entraîné de référence).
# GPUS="0 1" par défaut. Un run déjà terminé (summary.json présent) est sauté.
# Test rapide : EXTRA="--epochs 1" bash scripts/run_finetune.sh ...

set -u
DATA=${1:?"chemin du fichier did.parquet"}
CKPT=${CKPT:-runs/B_bpe16k/model.pt}
GPUS=(${GPUS:-0 1})
EXTRA=${EXTRA:-}

RUNS=()
for seed in 42 43 44; do
  for mode in full frozen scratch; do
    RUNS+=("F_${mode}_s${seed}|--mode $mode --seed $seed $EXTRA")
  done
done

mkdir -p runs
echo "${#RUNS[@]} fine-tunings, répartis sur les GPU ${GPUS[*]}"

run_queue () {
  local gpu=$1; shift
  for spec in "$@"; do
    IFS='|' read -r name opts <<< "$spec"
    echo "[GPU $gpu] $(date +%H:%M) début  $name"
    python scripts/finetune_did.py --data "$DATA" --ckpt "$CKPT" --out "runs/$name" --gpu "$gpu" $opts \
      > "runs/$name.log" 2>&1
    echo "[GPU $gpu] $(date +%H:%M) fin    $name ($(grep '^test' "runs/$name.log" | tail -n 1))"
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
echo "Tous les fine-tunings sont terminés."
python scripts/plot_finetune.py
