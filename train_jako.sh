#!/usr/bin/env bash
# Reversible-integer Japanese -> Korean translation: the exact commands used.
#
# Single process throughout, threads pinned. Oversubscription (more worker
# threads than cores) has previously cost more than it bought on this machine,
# so nothing here forks a pool.
set -eu

PY="${PY:-/c/Users/kdp75/AppData/Local/Programs/Python/Python313/python.exe}"
THREADS="${THREADS:-4}"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

echo "### Gate 1: corpus download, cleaning, tokenizer round-trip"
"$PY" reversible_jako_data.py --corpus tatoeba --vocab-size 2000 --max-len 32 --report

echo
echo "### Gates 2-5: inverse exactness, cache-free == cached, per-layer == batched,"
echo "###            attention vs matched-forward float autograd"
"$PY" reversible_jako_verify.py --all --threads "$THREADS"

echo
echo "### Depth-independent activation memory (K = 2, 4, 8)"
"$PY" reversible_jako_mt.py --mode memory --corpus tatoeba --vocab-size 2000 \
    --dim 64 --ffn-hidden 128 --batch 16 --depths 2 4 8 --threads "$THREADS"

echo
echo "### Gate 6: intentional overfit on 512 real Tatoeba JA-KO pairs"
"$PY" reversible_jako_mt.py --mode train --corpus tatoeba --max-pairs 512 \
    --epochs 70 --batch 16 --dim 64 --ffn-hidden 128 --k-enc 2 --k-dec 3 \
    --vocab-size 2000 --hidden-bits 16 --lr-bits 0 --show-samples 10 \
    --eval-every 10 --threads "$THREADS" --out jako_gate6_overfit.json

echo
echo "### Gate 7: real small training on TED2020"
"$PY" reversible_jako_mt.py --mode train --corpus ted2020 --max-pairs 6000 \
    --epochs 12 --batch 16 --dim 64 --ffn-hidden 128 --k-enc 2 --k-dec 3 \
    --vocab-size 3000 --hidden-bits 16 --lr-bits 0 --show-samples 10 \
    --eval-every 3 --threads "$THREADS" --out jako_results.json
