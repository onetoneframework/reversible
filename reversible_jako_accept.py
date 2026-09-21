#!/usr/bin/env python3
"""
30-sentence acceptance test for the reversible-integer JA->KO translator.

Held-out everyday Japanese covering greetings, self-introduction, weather, time
and dates, movement, school and work, food, affirmation and negation, numbers,
and simple questions. References are written out so the test is self-contained
and does not depend on a corpus split.

Reports, for every configuration given, the full 30 outputs (no cherry-picking)
plus token accuracy, chrF, BLEU, repetition, Korean-character ratio and
sequence exact match.

  python reversible_jako_accept.py --ckpt jako_tuned_best.pt --compare jako_model.pt
"""

import argparse
import json
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import reversible_jako_mt as M
import reversible_jako_decode as DEC

# (Japanese, Korean reference). Everyday, short, and deliberately varied.
SENTENCES = [
    ("こんにちは。", "안녕하세요."),
    ("おはようございます。", "안녕히 주무셨어요."),
    ("ありがとうございます。", "감사합니다."),
    ("すみません。", "죄송합니다."),
    ("さようなら。", "안녕히 가세요."),
    ("私は学生です。", "저는 학생입니다."),
    ("私の名前は田中です。", "제 이름은 다나카입니다."),
    ("日本から来ました。", "일본에서 왔습니다."),
    ("今日は暑いです。", "오늘은 덥습니다."),
    ("今日は雨です。", "오늘은 비가 옵니다."),
    ("明日は寒いでしょう。", "내일은 추울 것입니다."),
    ("今何時ですか。", "지금 몇 시입니까?"),
    ("明日は日曜日です。", "내일은 일요일입니다."),
    ("明日は学校に行きます。", "내일 학교에 갑니다."),
    ("駅はどこですか。", "역은 어디입니까?"),
    ("電車で行きます。", "전철로 갑니다."),
    ("彼は東京に住んでいます。", "그는 도쿄에 살고 있습니다."),
    ("私は会社員です。", "저는 회사원입니다."),
    ("仕事は忙しいです。", "일이 바쁩니다."),
    ("この本は面白いです。", "이 책은 재미있습니다."),
    ("水を飲みます。", "물을 마십니다."),
    ("私はコーヒーを飲みません。", "저는 커피를 마시지 않습니다."),
    ("ご飯を食べました。", "밥을 먹었습니다."),
    ("それはとてもおいしいです。", "그것은 매우 맛있습니다."),
    ("はい、そうです。", "네, 그렇습니다."),
    ("いいえ、違います。", "아니요, 다릅니다."),
    ("わかりません。", "모르겠습니다."),
    ("これはいくらですか。", "이것은 얼마입니까?"),
    ("三つください。", "세 개 주세요."),
    ("何時に始まりますか。", "몇 시에 시작합니까?"),
]


def evaluate_ckpt(path, mode, rep_penalty, no_rep_bigram, min_len, beam, threads):
    torch.set_num_threads(threads)
    model, tok, ck = M.load_ckpt(path)
    L = ck["max_len"]
    ja = [s for s, _r in SENTENCES]
    ref = [r for _s, r in SENTENCES]
    ids = []
    for s in ja:
        e = tok.encode(s).ids[:L - 1] + [M.EOS]
        ids.append(e + [M.PAD] * (L - len(e)))
    src = torch.tensor(ids, dtype=torch.int64)
    out = DEC.decode(model, src, L, mode, rep_penalty, no_rep_bigram, min_len, beam)
    st, hyp = DEC.stats(tok, out)

    # Teacher-forced token accuracy against the written references.
    tin, tout = [], []
    for r in ref:
        t = tok.encode(r).ids[:L - 1]
        ti = [M.BOS] + t
        to = t + [M.EOS]
        tin.append(ti + [M.PAD] * (L - len(ti)))
        tout.append(to + [M.PAD] * (L - len(to)))
    tin = torch.tensor(tin, dtype=torch.int64)
    tout = torch.tensor(tout, dtype=torch.int64)
    lg, _s = model.logits(model.decode(tin, model.encode(src)[0], src)[0])
    _g, stt = M.hinge_grad(lg, tout, 16)
    st["token_acc"] = round(100.0 * stt["correct"] / max(1, stt["tokens"]), 2)
    st["chrf"] = round(M.chrf(hyp, ref), 3)
    st["bleu"] = round(M.bleu(hyp, ref), 3)
    st["seq_exact_match"] = round(100.0 * sum(
        1 for h, r in zip(hyp, ref) if h.strip() == r.strip()) / len(ref), 2)
    return hyp, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="jako_tuned_best.pt")
    ap.add_argument("--compare", default="", help="baseline checkpoint (original config)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rep-penalty", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--out", default="jako_acceptance.json")
    a = ap.parse_args()

    cols = {}
    if a.compare:
        cols["ORIGINAL"] = evaluate_ckpt(a.compare, "greedy", 0, False, 1, 1, a.threads)
    cols["TUNED GREEDY"] = evaluate_ckpt(a.ckpt, "greedy", 0, False, 1, 1, a.threads)
    cols["TUNED PRACTICAL"] = evaluate_ckpt(
        a.ckpt, "greedy-repeat", a.rep_penalty, True, a.min_len, 1, a.threads)

    print("=" * 78)
    for i, (ja, ref) in enumerate(SENTENCES):
        print(f"[{i+1:>2}] JA:  {ja}")
        print(f"     REF: {ref}")
        for name, (hyp, _st) in cols.items():
            print(f"     {name:<16}: {hyp[i]}")
        print()
    print("=" * 78)
    hdr = f"{'config':<18}" + "".join(f"{k:>10}" for k in
                                      ("token_acc", "chrF", "BLEU", "seq_EM",
                                       "rep_uni%", "rep_bi%", "korean%", "repl%",
                                       "distinct"))
    print(hdr)
    for name, (_h, st) in cols.items():
        print(f"{name:<18}" + "".join(f"{st[k]:>10}" for k in
                                      ("token_acc", "chrf", "bleu", "seq_exact_match",
                                       "repeat_unigram_pct", "repeat_bigram_pct",
                                       "korean_char_pct", "replacement_char_pct",
                                       "distinct_outputs")))
    json.dump({name: {"outputs": h, "metrics": s} for name, (h, s) in cols.items()},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
