#!/usr/bin/env python3
"""
Japanese-Korean parallel corpus download, cleaning and subword tokenisation.

Corpora (real JA-KO pairs, not substituted from any other language pair):

  Tatoeba   https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/ja-ko.txt.zip
            CC-BY 2.0 FR. Small, clean, single sentences. Used for the overfit gate.
  TED2020   https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-ko.txt.zip
            TED talk subtitles, CC BY-NC-ND 4.0. Used for the real training gate.

Both are distributed by OPUS (https://opus.nlpl.eu/) in Moses format: one plain
text file per language, line-aligned.

  python reversible_jako_data.py --corpus tatoeba --report
  python reversible_jako_data.py --corpus ted2020 --max-pairs 30000 --build-tokenizer
"""

import argparse
import hashlib
import sys

# Windows consoles here default to CP949, which cannot encode kana/kanji.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import io
import json
import os
import re
import unicodedata
import urllib.request
import zipfile

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_jako_data")

CORPORA = {
    "both": {"url": "(tatoeba + ted2020)", "license": "see components",
             "cite": "OPUS Tatoeba + OPUS TED2020"},
    "tatoeba": {
        "url": "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/ja-ko.txt.zip",
        "license": "CC-BY 2.0 FR",
        "cite": "OPUS Tatoeba v2023-04-12",
    },
    "ted2020": {
        "url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-ko.txt.zip",
        "license": "CC BY-NC-ND 4.0 (TED)",
        "cite": "OPUS TED2020 v1",
    },
    "opensubtitles": {
        "url": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/ja-ko.txt.zip",
        "license": "OpenSubtitles.org terms; no explicit open licence",
        "cite": "OPUS OpenSubtitles v2024",
    },
    "mixed": {"url": "(tatoeba + ted2020 + filtered opensubtitles)",
              "license": "see components",
              "cite": "OPUS Tatoeba + TED2020 + OpenSubtitles v2024"},
}

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Rough script checks, used only to reject pairs that are obviously not JA/KO.
HANGUL = re.compile(r"[가-힣]")
KANA = re.compile(r"[぀-ヿ]")
CJK = re.compile(r"[一-鿿]")


def fetch(corpus):
    spec = CORPORA[corpus]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{corpus}_ja-ko.zip")
    if not (os.path.exists(path) and os.path.getsize(path) > 1000):
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            blob = r.read()
        with open(path, "wb") as fh:
            fh.write(blob)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return path, digest


def read_pairs(corpus):
    """Returns ([(ja, ko, tag)], meta). The tag names the source corpus so that
    validation can be scored per source after the corpora are mixed."""
    if corpus == "both":
        a, ma = read_pairs("tatoeba")
        b, mb = read_pairs("ted2020")
        return a + b, {"zip": "tatoeba+ted2020", "sha256": ma["sha256"] + "/" + mb["sha256"],
                       "ja_file": "combined", "ko_file": "combined",
                       "raw_lines": ma["raw_lines"] + mb["raw_lines"],
                       "url": ma["url"] + " ; " + mb["url"],
                       "license": ma["license"] + " ; " + mb["license"],
                       "cite": ma["cite"] + " ; " + mb["cite"]}
    path, digest = fetch(corpus)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        ja_name = next(n for n in names if n.endswith(".ja"))
        ko_name = next(n for n in names if n.endswith(".ko"))
        ja = z.read(ja_name).decode("utf-8", "replace").split("\n")
        ko = z.read(ko_name).decode("utf-8", "replace").split("\n")
    n = min(len(ja), len(ko))
    tagged = [(ja[i], ko[i], corpus) for i in range(n)]
    return tagged, {"zip": os.path.basename(path), "sha256": digest,
                                       "ja_file": ja_name, "ko_file": ko_name,
                                       "raw_lines": n, **CORPORA[corpus]}


def clean(pairs, min_chars=2, max_chars=80, block_acceptance=False):
    """Normalise, drop empties, length outliers, obvious script mismatches, dupes.

    block_acceptance also drops any pair matching one of the 30 acceptance
    sentences on a normalised key, so that test set stays genuinely held out.
    Six of them ("ありがとうございます。" and friends) do occur in Tatoeba/TED."""
    seen = set()
    out = []
    stats = {"raw": len(pairs), "empty": 0, "length": 0, "script": 0, "dupe": 0,
             "acceptance": 0}
    blocked = set()
    if block_acceptance:
        for ja, ko in acceptance_sentences():
            blocked.add(norm_key(ja))
            blocked.add(norm_key(ko))
        blocked.discard("")
    for rec in pairs:
        ja, ko, tag = rec[0], rec[1], (rec[2] if len(rec) > 2 else "?")
        ja = unicodedata.normalize("NFKC", ja).strip()
        ko = unicodedata.normalize("NFKC", ko).strip()
        if not ja or not ko:
            stats["empty"] += 1
            continue
        if not (min_chars <= len(ja) <= max_chars and min_chars <= len(ko) <= max_chars):
            stats["length"] += 1
            continue
        # Korean side must contain Hangul; Japanese side must contain kana or CJK.
        if not HANGUL.search(ko) or not (KANA.search(ja) or CJK.search(ja)):
            stats["script"] += 1
            continue
        if blocked and (norm_key(ja) in blocked or norm_key(ko) in blocked):
            stats["acceptance"] += 1
            continue
        key = (ja, ko)
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        out.append((ja, ko, tag))
    stats["kept"] = len(out)
    return out, stats


# -----------------------------------------------------------------------------
# OpenSubtitles: strict filtering
#
# OpenSubtitles is built from alternative subtitle uploads for the same films,
# so the same line is aligned many times over and OPUS says so itself. Random
# sampling from it would mostly buy duplicates and misalignments, so the filter
# below is deliberately harsh: it throws away far more than it keeps, and the
# corpus is large enough (1.19M raw ja-ko lines) that this still leaves plenty.
# -----------------------------------------------------------------------------

HTML_TAG = re.compile(r"</?[a-zA-Z][^>]{0,60}>")
ASS_TAG = re.compile(r"\{[^{}]{0,60}\}")
TIMECODE = re.compile(r"\d{1,2}\s*:\s*\d{2}(?:\s*:\s*\d{2})?(?:[,.]\d{1,3})?")
BRACKETED = re.compile(r"[(\[\uff08\u3010][^)\]\uff09\u3011]{0,60}[)\]\uff09\u3011]")
MUSIC = re.compile(r"[\u266a\u266b\u266c\u2669\u25c6\u2605\u2606\u203b]")
LEAD_DASH = re.compile(r"^[-\u2010-\u2015]\s*")
TURN_DASH = re.compile(r"\S\s+[-\u2010-\u2015]\s+\S")
SPEAKER = re.compile(r"^[^\s:\uff1a]{1,14}[:\uff1a]\s")
URLISH = re.compile(r"(?i)(https?://|www\.|opensubtitles|\.com\b|\.net\b|\.org\b)")
LATIN = re.compile(r"[A-Za-z]")
WORDISH = re.compile(r"[0-9A-Za-z\uac00-\ud7a3\u3041-\u309f\u30a0-\u30ff\u4e00-\u9fff]")
NONWORD = re.compile(r"[^0-9A-Za-z\uac00-\ud7a3\u3041-\u309f\u30a0-\u30ff\u4e00-\u9fff]")
JAPANESE = re.compile(r"[\u3041-\u309f\u30a0-\u30ff\u4e00-\u9fff]")


def norm_key(s):
    """Near-duplicate key: letters and digits only, case folded."""
    return NONWORD.sub("", s).lower()


def acceptance_sentences():
    """The 30 held-out acceptance sentences, read out of the acceptance script
    without importing it (that would pull in torch)."""
    import ast
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "reversible_jako_accept.py")
    if not os.path.exists(path):
        return []
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SENTENCES":
            return ast.literal_eval(node.value)
    return []


def _strip_markup(t, st):
    """Remove subtitle markup. Returns None when the line should be rejected."""
    if MUSIC.search(t):
        st["music"] += 1
        return None
    if URLISH.search(t):
        st["url"] += 1
        return None
    if TIMECODE.search(t):
        st["timecode"] += 1
        return None
    t = ASS_TAG.sub(" ", HTML_TAG.sub(" ", t))
    # Bracketed spans are stage directions and sound effects: (\uc6c3\uc74c), [\uc74c\uc545].
    stripped = BRACKETED.sub(" ", t)
    if len(t) and (len(t) - len(stripped)) / len(t) > 0.20:
        st["bracket"] += 1
        return None
    t = stripped
    if TURN_DASH.search(t):
        st["two_speaker"] += 1
        return None
    t = LEAD_DASH.sub("", t)
    if SPEAKER.search(t):
        st["speaker"] += 1
        return None
    return re.sub(r"\s+", " ", t).strip()


def clean_subtitles(pairs, block_pairs=(), take=0, seed=0,
                    min_chars=4, max_chars=60, ratio_lo=0.70, ratio_hi=2.20,
                    script_frac=0.70, latin_frac=0.30, wordish_frac=0.60):
    """
    Hard filter for OpenSubtitles. `block_pairs` are the pairs already in the
    corpus (and the acceptance sentences); anything matching them on a
    normalised key is dropped so the new data cannot leak the test set in.

    The ratio bounds are the 5th/95th percentile of the Tatoeba+TED2020 corpus
    (0.787 / 2.027), widened slightly.
    """
    st = dict(raw=len(pairs), empty=0, music=0, url=0, timecode=0, bracket=0,
              two_speaker=0, speaker=0, length=0, ratio=0, symbols=0, latin=0,
              script=0, mixed_script=0, dupe=0, src_dupe=0, tgt_dupe=0, blocked=0)

    blocked = set()
    for rec in block_pairs:
        blocked.add(norm_key(rec[0]))
        blocked.add(norm_key(rec[1]))
    for ja, ko in acceptance_sentences():
        blocked.add(norm_key(ja))
        blocked.add(norm_key(ko))
    blocked.discard("")

    seen_pair, seen_ja, seen_ko = set(), set(), set()
    out = []
    for rec in pairs:
        ja = unicodedata.normalize("NFKC", rec[0]).strip()
        ko = unicodedata.normalize("NFKC", rec[1]).strip()
        if not ja or not ko:
            st["empty"] += 1
            continue
        ja = _strip_markup(ja, st)
        if ja is None:
            continue
        ko = _strip_markup(ko, st)
        if ko is None:
            continue
        if not (min_chars <= len(ja) <= max_chars and min_chars <= len(ko) <= max_chars):
            st["length"] += 1
            continue
        r = len(ko) / len(ja)
        if not (ratio_lo <= r <= ratio_hi):
            st["ratio"] += 1
            continue
        # Numbers-and-symbols-only lines carry no translation signal.
        wj = WORDISH.findall(ja)
        wk = WORDISH.findall(ko)
        nsj = len(ja.replace(" ", ""))
        nsk = len(ko.replace(" ", ""))
        if (not wj or not wk or len(wj) < wordish_frac * nsj
                or len(wk) < wordish_frac * nsk):
            st["symbols"] += 1
            continue
        if (len(LATIN.findall(ja)) > latin_frac * len(wj)
                or len(LATIN.findall(ko)) > latin_frac * len(wk)):
            st["latin"] += 1
            continue
        # Script purity: the Japanese side must be mostly Japanese and carry no
        # Hangul; the Korean side must be mostly Hangul and carry no kana.
        if len(JAPANESE.findall(ja)) < script_frac * len(wj):
            st["script"] += 1
            continue
        if len(HANGUL.findall(ko)) < script_frac * len(wk):
            st["script"] += 1
            continue
        if HANGUL.search(ja) or KANA.search(ko):
            st["mixed_script"] += 1
            continue
        kj, kk = norm_key(ja), norm_key(ko)
        if not kj or not kk:
            st["symbols"] += 1
            continue
        if kj in blocked or kk in blocked:
            st["blocked"] += 1
            continue
        if (kj, kk) in seen_pair:
            st["dupe"] += 1
            continue
        # Alternative uploads align the same source line to many targets (and
        # vice versa). Keep one alignment per side, not one per pair.
        if kj in seen_ja:
            st["src_dupe"] += 1
            continue
        if kk in seen_ko:
            st["tgt_dupe"] += 1
            continue
        seen_pair.add((kj, kk))
        seen_ja.add(kj)
        seen_ko.add(kk)
        out.append((ja, ko, "opensubtitles"))

    st["kept"] = len(out)
    if take and len(out) > take:
        import random
        random.Random(seed).shuffle(out)
        out = out[:take]
    st["taken"] = len(out)
    return out, st


def build_tokenizer(pairs, vocab_size, path, mode="char", min_char_count=1):
    """
    Shared subword tokenizer over both languages.

    mode="char"  BPE over Unicode characters. Every token decodes to valid text,
                 so generation can never emit a broken UTF-8 sequence. This
                 matters: with byte-level BPE a third of the vocabulary decodes
                 to U+FFFD on its own, and an under-trained model emits those.
    mode="byte"  the original byte-level BPE, kept for comparison.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    if mode == "byte":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        init_alpha = pre_tokenizers.ByteLevel.alphabet()
    else:
        # Split on whitespace only, keep the space as a marker, and let BPE work
        # on whole characters. Every character seen in the corpus is in the
        # alphabet, so nothing falls back to a byte fragment.
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁",
                                                     prepend_scheme="always")
        tok.decoder = decoders.Metaspace(replacement="▁",
                                         prepend_scheme="always")
        # Every character in the corpus is forced into the vocabulary, which on a
        # mixed corpus means ~4.6k single characters and almost no room left for
        # merges. min_char_count drops the long tail (rare kanji seen once or
        # twice in a subtitle) to <unk> and spends the slots on merges instead.
        counts = {}
        for pr in pairs:
            for sent in pr[:2]:
                for ch in sent:
                    counts[ch] = counts.get(ch, 0) + 1
        # The word-boundary marker is inserted by the pre-tokenizer, so it never
        # shows up in `counts`; without it here the alphabet cap evicts it and
        # every space in the corpus tokenises to <unk>.
        init_alpha = ["▁"] + sorted(c for c, n in counts.items()
                                         if n >= min_char_count and c != "▁")
        # initial_alphabet only *adds* characters; the trainer still takes every
        # character it sees as a base token. limit_alphabet is the cap that
        # actually keeps the long tail out.
        limit = len(init_alpha) if min_char_count > 1 else None
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2,
        initial_alphabet=init_alpha, show_progress=False,
        **({"limit_alphabet": limit} if mode != "byte" and limit else {}))
    tok.train_from_iterator((s for p in pairs for s in p[:2]), trainer)
    tok.save(path)
    return tok


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def encode_pairs(tok, pairs, max_len):
    """Returns (src_ids, tgt_in_ids, tgt_out_ids, kept, kept_index).

    kept_index maps each encoded row back to its position in `pairs`, so a
    parallel array (the corpus tag) stays aligned when rows are dropped."""
    src, tin, tout, idx = [], [], [], []
    for i, rec in enumerate(pairs):
        s = tok.encode(rec[0]).ids
        t = tok.encode(rec[1]).ids
        # src: tokens + EOS ; tgt_in: BOS + tokens ; tgt_out: tokens + EOS
        if len(s) + 1 > max_len or len(t) + 1 > max_len:
            continue
        s = s + [EOS]
        ti = [BOS] + t
        to = t + [EOS]
        src.append(s + [PAD] * (max_len - len(s)))
        tin.append(ti + [PAD] * (max_len - len(ti)))
        tout.append(to + [PAD] * (max_len - len(to)))
        idx.append(i)
    return src, tin, tout, len(idx), idx


def prepare(corpus="tatoeba", max_pairs=0, vocab_size=4000, max_len=32, seed=0,
            valid_frac=0.05, rebuild=False, max_tok=0, tok_mode="char",
            subs_pairs=25000, subs_pool=0, base_pairs=25000, base_pool=0,
            min_char_count=1):
    """
    corpus="mixed" is Tatoeba+TED2020 plus a hard-filtered slice of
    OpenSubtitles. The subtitle slice is trimmed to `subs_pairs` *after* the
    max_tok filter, so the requested count is what actually reaches training.
    Every pair keeps a corpus tag, and the tags are returned alongside the
    encoded arrays so validation can be scored per source.
    """
    os.makedirs(CACHE, exist_ok=True)
    import random
    subs_stats = None
    if corpus == "mixed":
        base, meta = read_pairs("both")
        base, cstats = clean(base, block_acceptance=True)
        raw_subs, meta_s = read_pairs("opensubtitles")
        pool = subs_pool or max(4 * subs_pairs, 100000)
        bpool = base_pool or pool
        subs, subs_stats = clean_subtitles(raw_subs, base, take=pool, seed=seed)
        del raw_subs
        # Sample the base down to the same pool size, so the tokenizer is
        # trained on the same 1:1 mixture the model will actually see.
        random.Random(seed).shuffle(base)
        base = base[:bpool]
        meta = {"zip": meta["zip"] + "+" + meta_s["zip"],
                "sha256": meta["sha256"] + "/" + meta_s["sha256"],
                "ja_file": "combined", "ko_file": "combined",
                "raw_lines": meta["raw_lines"] + meta_s["raw_lines"],
                "url": meta["url"] + " ; " + meta_s["url"],
                "license": meta["license"] + " ; " + meta_s["license"],
                "cite": meta["cite"] + " ; " + meta_s["cite"]}
        pairs = base + subs
        tok_tag = "mixed%dk_%dk_a%d" % (bpool // 1000, pool // 1000, min_char_count)
    else:
        pairs, meta = read_pairs(corpus)
        # The acceptance sentences are blocked from every corpus, not just the
        # mixed one: six of them occur verbatim in Tatoeba/TED.
        pairs, cstats = clean(pairs, block_acceptance=True)
        tok_tag = corpus + ("_a%d" % min_char_count if min_char_count > 1 else "")
    random.Random(seed).shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]

    tok_path = os.path.join(CACHE, "tok_%s_%d_%s.json" % (tok_tag, vocab_size, tok_mode))
    if rebuild or not os.path.exists(tok_path):
        build_tokenizer(pairs, vocab_size, tok_path, tok_mode, min_char_count)
    tok = load_tokenizer(tok_path)

    if max_tok:
        # Short-sentence curriculum: keep pairs whose BOTH sides fit in
        # max_tok subword tokens, so the model sees learnable mappings first.
        enc = tok.encode_batch([x for p in pairs for x in (p[0], p[1])])
        pairs = [p for i, p in enumerate(pairs)
                 if len(enc[2 * i].ids) <= max_tok and len(enc[2 * i + 1].ids) <= max_tok]
        del enc
    tag_counts_pool = {}
    for p in pairs:
        tag_counts_pool[p[2]] = tag_counts_pool.get(p[2], 0) + 1
    if corpus != "mixed" and base_pairs:
        # Same knob as the mixed path: trim after max_tok so the requested pair
        # count is what actually reaches training.
        pairs = pairs[:base_pairs]
    if corpus == "mixed" and (subs_pairs or base_pairs):
        # Trim each side to its requested count, after max_tok, so the numbers
        # asked for are the numbers that reach training.
        keep, nsub, nbase = [], 0, 0
        for p in pairs:
            if p[2] == "opensubtitles":
                if not subs_pairs or nsub < subs_pairs:
                    keep.append(p)
                    nsub += 1
            elif not base_pairs or nbase < base_pairs:
                keep.append(p)
                nbase += 1
        pairs = keep
        random.Random(seed + 1).shuffle(pairs)

    nval = max(1, int(len(pairs) * valid_frac))
    valid_pairs, train_pairs = pairs[:nval], pairs[nval:]
    tr = encode_pairs(tok, train_pairs, max_len)
    va = encode_pairs(tok, valid_pairs, max_len)
    train_pairs = [train_pairs[i] for i in tr[4]]
    valid_pairs = [valid_pairs[i] for i in va[4]]
    tag_counts = {}
    for p in train_pairs:
        tag_counts[p[2]] = tag_counts.get(p[2], 0) + 1
    vtag_counts = {}
    for p in valid_pairs:
        vtag_counts[p[2]] = vtag_counts.get(p[2], 0) + 1
    info = {
        "corpus": corpus, **meta, "clean_stats": cstats,
        "subtitle_filter": subs_stats,
        "pairs_after_max_tok": tag_counts_pool,
        "pairs_after_limit": len(pairs), "vocab_size": tok.get_vocab_size(),
        "max_len": max_len, "train_encoded": tr[3], "valid_encoded": va[3],
        "train_by_corpus": tag_counts, "valid_by_corpus": vtag_counts,
        "tokenizer": os.path.basename(tok_path),
    }
    return {"train": tr[:3], "valid": va[:3],
            "train_pairs": train_pairs, "valid_pairs": valid_pairs,
            "train_tags": [p[2] for p in train_pairs],
            "valid_tags": [p[2] for p in valid_pairs],
            "tokenizer": tok, "info": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA), default="tatoeba")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-tok", type=int, default=0)
    ap.add_argument("--subs-pairs", type=int, default=25000)
    ap.add_argument("--subs-pool", type=int, default=0)
    ap.add_argument("--base-pairs", type=int, default=25000)
    ap.add_argument("--base-pool", type=int, default=0)
    ap.add_argument("--min-char-count", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    d = prepare(a.corpus, a.max_pairs, a.vocab_size, a.max_len,
                rebuild=a.rebuild, max_tok=a.max_tok,
                subs_pairs=a.subs_pairs, subs_pool=a.subs_pool,
                base_pairs=a.base_pairs, base_pool=a.base_pool,
                min_char_count=a.min_char_count)
    print(json.dumps(d["info"], ensure_ascii=False, indent=2))
    if a.report:
        tok = d["tokenizer"]
        print("\n--- round-trip check (Gate 1) ---")
        for ja, ko, _tag in d["train_pairs"][:5]:
            ja_ids = tok.encode(ja).ids
            ko_ids = tok.encode(ko).ids
            print(f"JA  {ja}")
            print(f"    ids[{len(ja_ids)}] {ja_ids[:12]}{'...' if len(ja_ids) > 12 else ''}")
            print(f"    dec {tok.decode(ja_ids)}")
            print(f"KO  {ko}")
            print(f"    ids[{len(ko_ids)}] {ko_ids[:12]}{'...' if len(ko_ids) > 12 else ''}")
            print(f"    dec {tok.decode(ko_ids)}")
            print()


if __name__ == "__main__":
    main()
