set -u
PY="$LOCALAPPDATA/Programs/Python/Python313/python.exe"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONIOENCODING=utf-8
r() { echo; echo "########## $1 ##########"; shift; "$PY" -X utf8 -u "$@" 2>&1 | tail -n "${TAIL:-6}"; echo "exit=$?"; }
TAIL=3  r "digits golden fingerprint" reversible_integer_golden_capture.py
TAIL=25 r "digits gate matrix"        reversible_integer_gate_matrix.py
TAIL=20 r "structural gates"          reversible_integer_structural.py
TAIL=8  r "conv structural (verify)"  reversible_arch.py --arch conv --mode verify
TAIL=8  r "attn structural (verify)"  reversible_arch.py --arch attn --mode verify
TAIL=8  r "jako gates 2-5"            reversible_jako_verify.py --threads 6 --out jako_gates_causalfix.json
TAIL=4  r "jako head gate"            reversible_jako_headgate.py --threads 6 --out jako_headgate_causalfix.json
TAIL=12 r "jako causality A/B/C"      reversible_jako_causalgate.py --threads 6 --out jako_causalgate_fixed.json
echo; echo "########## DONE ##########"
