#!/usr/bin/env python3
"""
Translate a fixed probe set every time a training run saves a new checkpoint.

A training run only writes its checkpoint on an improving validation, so this
shows the translations for exactly the epochs that got better -- an epoch with
no new best leaves no weights to decode from, and this says so rather than
repeating the previous epoch's output as if it were new.

Runs outside the trainer, so it works on a job that is already in flight.

  python reversible_jako_probe.py --ckpt jako_50k_best.pt --watch
  python reversible_jako_probe.py --ckpt jako_50k_best.pt          # once
"""

import argparse
import io
import os
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_jako_mt as M
import reversible_jako_decode as DEC

# Short, everyday, and held out: the acceptance sentences are blocked from the
# training corpus, so these are never memorised.
PROBE = [
    ("こんにちは。", "안녕하세요."),
    ("私は学生です。", "저는 학생입니다."),
    ("三つください。", "세 개 주세요."),
    ("今日は暑いです。", "오늘은 덥습니다."),
    ("駅はどこですか。", "역은 어디입니까?"),
    ("水を飲みます。", "물을 마십니다."),
    ("わかりません。", "모르겠습니다."),
    ("何時に始まりますか。", "몇 시에 시작합니까?"),
]


def settled(path, quiet=6.0, tries=40):
    """Wait until the file has stopped growing; torch.save is not atomic."""
    last = (-1, -1.0)
    for _ in range(tries):
        try:
            st = os.stat(path)
        except OSError:
            time.sleep(2.0)
            continue
        now = (st.st_size, st.st_mtime)
        if now == last and time.time() - st.st_mtime > quiet:
            return True
        last = now
        time.sleep(2.0)
    return False


OUT_FILE = None


def emit(line=""):
    """Print, and append to the durable transcript if one was asked for."""
    print(line, flush=True)
    if OUT_FILE:
        with io.open(OUT_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def probe(path, threads, rep_penalty, min_len, tag=""):
    torch.set_num_threads(threads)
    model, tok, ck = M.load_ckpt(path)
    L = ck["max_len"]
    ids = []
    for ja, _ref in PROBE:
        e = tok.encode(ja).ids[:L - 1] + [M.EOS]
        ids.append(e + [M.PAD] * (L - len(e)))
    src = torch.tensor(ids, dtype=torch.int64)
    t0 = time.time()
    g = DEC.decode(model, src, L, "greedy", 0, False, 1, 1)
    r = DEC.decode(model, src, L, "greedy-repeat", rep_penalty, True, min_len, 1)
    gs, gh = DEC.stats(tok, g)
    rs, rh = DEC.stats(tok, r)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    emit("=" * 76)
    emit("[%s] %s%s  epoch %s  (decoded in %.0fs)"
         % (stamp, tag, os.path.basename(path), ck.get("epoch"), time.time() - t0))
    emit("=" * 76)
    for i, (ja, ref) in enumerate(PROBE):
        emit("JA      %s" % ja)
        emit("REF     %s" % ref)
        emit("GREEDY  %s" % gh[i])
        emit("REPEAT  %s" % rh[i])
        emit()
    for name, st in (("greedy", gs), ("repeat", rs)):
        emit("  %-7s korean%% %-6s repl%% %-5s rep_uni%% %-6s rep_bi%% %-6s distinct %s"
             % (name, st["korean_char_pct"], st["replacement_char_pct"],
                st["repeat_unigram_pct"], st["repeat_bigram_pct"],
                st["distinct_outputs"]))
    emit()
    return ck.get("epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jako_50k_best.pt")
    ap.add_argument("--watch", action="store_true",
                    help="keep running, decoding each time the checkpoint changes")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=2,
                    help="kept small on purpose: the trainer owns the cores")
    ap.add_argument("--rep-penalty", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--out", default="",
                    help="append every probe to this file as well as stdout")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global OUT_FILE
    OUT_FILE = a.out or None

    if not a.watch:
        probe(a.ckpt, a.threads, a.rep_penalty, a.min_len, a.tag)
        return

    seen = None
    deadline = time.time() + a.max_hours * 3600
    while time.time() < deadline:
        try:
            mt = os.stat(a.ckpt).st_mtime
        except OSError:
            time.sleep(a.poll)
            continue
        if mt != seen and settled(a.ckpt):
            seen = os.stat(a.ckpt).st_mtime
            try:
                probe(a.ckpt, a.threads, a.rep_penalty, a.min_len, a.tag)
            except Exception as e:          # a half-written file, most likely
                emit("[%s] probe failed: %s: %s"
                     % (time.strftime("%H:%M:%S"), type(e).__name__, e))
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
