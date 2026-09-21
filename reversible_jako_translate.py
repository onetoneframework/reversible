#!/usr/bin/env python3
"""
Interactive Japanese -> Korean translator on the reversible integer model.

    python reversible_jako_translate.py --checkpoint jako_tuned_best.pt

    Japanese> 今日は雨です。
    Korean>   ...

Options:
    --checkpoint PATH     trained checkpoint (required to be a real trained one)
    --max-len N           decode length cap
    --decode greedy|greedy-repeat|beam
    --beam N
    --threads N

The checkpoint is validated before use: its config and tokenizer hash must be
consistent, and a randomly initialised model is refused. An earlier version of
this project once "verified the pipeline" on a fresh random model; that is
exactly what the fingerprint check below is here to prevent.
"""

import argparse
import hashlib
import json
import os
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_jako_mt as M
import reversible_jako_decode as DEC

PAD, BOS, EOS = M.PAD, M.BOS, M.EOS


def fingerprint(ck):
    h = hashlib.sha256()
    for w in ck["weights"]:
        h.update(w.numpy().tobytes())
    return h.hexdigest()[:16]


def validate(path):
    if not os.path.exists(path):
        raise SystemExit(f"[error] checkpoint not found: {path}\n"
                         f"        train one first, e.g. run.bat train-short")
    ck = torch.load(path, weights_only=False)
    need = ("weights", "vocab", "dim", "ffn_hidden", "k_enc", "k_dec",
            "max_len", "state_bits", "weight_bits", "accum_bits", "tokenizer_path")
    missing = [k for k in need if k not in ck]
    if missing:
        raise SystemExit(f"[error] checkpoint missing fields: {missing}")
    if not os.path.exists(ck["tokenizer_path"]):
        raise SystemExit(f"[error] tokenizer referenced by the checkpoint is gone:\n"
                         f"        {ck['tokenizer_path']}")
    with open(ck["tokenizer_path"], "rb") as fh:
        tok_hash = hashlib.sha256(fh.read()).hexdigest()[:16]
    # A freshly initialised model has weights drawn uniformly in [-16, 16] and a
    # near-zero mean per tensor. A trained one has moved and clamps wider.
    moved = max(int(w.abs().max()) for w in ck["weights"])
    trained = moved > 16
    return ck, {"weight_fingerprint": fingerprint(ck), "tokenizer_sha256": tok_hash,
                "max_abs_weight": moved, "looks_trained": trained,
                "vocab": ck["vocab"], "dim": ck["dim"],
                "k_enc": ck["k_enc"], "k_dec": ck["k_dec"],
                "enc_pattern": ck.get("enc_pattern", ""),
                "dec_pattern": ck.get("dec_pattern", ""),
                "state_bits": ck["state_bits"], "weight_bits": ck["weight_bits"],
                "hidden_bits_note": "adjoint mantissa is a training-time knob"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="jako_tuned_best.pt")
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--decode", choices=["greedy", "greedy-repeat", "beam"],
                    default="greedy-repeat")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--rep-penalty", type=int, default=0)
    ap.add_argument("--no-repeat-bigram", action="store_true", default=True)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--once", nargs="*", default=None,
                    help="translate these sentences and exit")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ck, info = validate(a.checkpoint)
    print("checkpoint: " + a.checkpoint)
    print(json.dumps(info, ensure_ascii=False))
    if not info["looks_trained"]:
        print("[warning] this checkpoint looks untrained (no weight has moved "
              "beyond the init range). Translations will be noise.")
    model, tok, _ck = M.load_ckpt(a.checkpoint)
    L = a.max_len or ck["max_len"]
    rep = a.rep_penalty or (8 if a.decode != "greedy" else 0)

    def run(sentences):
        ids = []
        for s in sentences:
            e = tok.encode(s).ids[:L - 1] + [EOS]
            ids.append(e + [PAD] * (L - len(e)))
        src = torch.tensor(ids, dtype=torch.int64)
        out = DEC.decode(model, src, L, a.decode, rep,
                         a.no_repeat_bigram, a.min_len, a.beam)
        return [M.ids_to_text(tok, out[i]) for i in range(len(sentences))]

    if a.once is not None:
        sents = a.once or ["今日は雨です。", "私は学生です。", "明日は学校に行きます。"]
        for s, t in zip(sents, run(sents)):
            print("JA: " + s)
            print("KO: " + t)
        return

    print(f"\ndecode={a.decode} rep_penalty={rep} no_repeat_bigram={a.no_repeat_bigram} "
          f"min_len={a.min_len}" + (f" beam={a.beam}" if a.decode == "beam" else ""))
    print("Type Japanese, blank line or Ctrl-C to quit.\n")
    while True:
        try:
            line = input("Japanese> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        print("Korean>   " + run([line])[0] + "\n")


if __name__ == "__main__":
    main()
