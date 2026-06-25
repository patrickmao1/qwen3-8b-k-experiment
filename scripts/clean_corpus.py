#!/usr/bin/env python3
"""
Clean + dedup + weight the scraped K corpus.

Steps:
  1. Read every harvested file from data/corpus/{k_code,k_docs} via manifest.jsonl
  2. Normalize (strip comments + collapse whitespace) for dedup detection
  3. Exact dedup (sha1 of normalized text)
  4. Near-dedup via MinHash + LSH banding (numpy), Jaccard >= THRESH -> same cluster,
     keep one representative per cluster (the largest file)
  5. Assign keep/weight per policy:
       - legacy (K3/Maude) era            -> exclude (deprecated syntax)
       - exact-dup / near-dup non-rep     -> exclude
       - X86-64 auto-generated            -> keep reps but down-weight (cap its share)
       - non-K markdown (no code blocks)  -> keep, lower weight (prose)
       - everything else (modern K, docs) -> keep, weight 1.0
  6. Write data/corpus/clean_manifest.jsonl + print a stats report.

Token counts here are a char/3.5 ESTIMATE (no tokenizer dep yet); they will be
recomputed with the real Qwen3 tokenizer when the training env is set up.
"""
import json, os, re, hashlib, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
CODE_DIR = os.path.join(CORPUS, "k_code")
DOCS_DIR = os.path.join(CORPUS, "k_docs")
MANIFEST = os.path.join(CORPUS, "manifest.jsonl")
OUT = os.path.join(CORPUS, "clean_manifest.jsonl")

THRESH = 0.80          # Jaccard threshold for "near duplicate"
NUM_PERM = 128         # MinHash permutations
BANDS, ROWS = 32, 4    # 32*4 = 128; permissive candidate generation, verified after
SHINGLE = 8            # token k-gram size
CHARS_PER_TOK = 3.5
X86_REPO = "kframework/X86-64-semantics"
X86_TOKEN_CAP = 0.10   # X86 contributes at most ~10% of kept K tokens (via weight)

_comment_re = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_ws_re = re.compile(r"\s+")
MERSENNE = (1 << 61) - 1

def file_path(row):
    base = CODE_DIR if row["kind"] == "k_code" else DOCS_DIR
    slug = row["repo"].replace("/", "__")
    return os.path.join(base, slug, row["path"])

def normalize(text):
    return _ws_re.sub(" ", _comment_re.sub(" ", text)).strip()

def shingle_hashes(norm_text):
    toks = norm_text.split(" ")
    if len(toks) < SHINGLE:
        grams = [" ".join(toks)] if toks else []
    else:
        grams = [" ".join(toks[i:i+SHINGLE]) for i in range(len(toks)-SHINGLE+1)]
    hs = {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
          for g in grams}
    return np.fromiter(hs, dtype=np.uint64, count=len(hs)) if hs else np.zeros(0, np.uint64)

def minhash(hashes, a, b):
    if hashes.size == 0:
        return np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    # (a*h + b) mod (2^61-1), vectorized over perms x shingles, min over shingles
    h = hashes.astype(object)  # avoid uint64 overflow in mod arithmetic
    sig = np.empty(NUM_PERM, dtype=np.uint64)
    for i in range(NUM_PERM):
        vals = (a[i] * h + b[i]) % MERSENNE
        sig[i] = int(vals.min())
    return sig

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb

def main():
    rng = np.random.default_rng(0)
    a = rng.integers(1, MERSENNE, size=NUM_PERM, dtype=np.int64).astype(object)
    b = rng.integers(0, MERSENNE, size=NUM_PERM, dtype=np.int64).astype(object)

    rows = [json.loads(l) for l in open(MANIFEST)]
    print(f"loaded {len(rows)} files; reading + hashing...", flush=True)

    sigs, shabuf, kept_rows = [], {}, []
    for i, r in enumerate(rows):
        try:
            text = open(file_path(r), encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            continue
        norm = normalize(text)
        r["_sha"] = hashlib.sha1(norm.encode()).hexdigest()
        r["_norm_len"] = len(norm)
        r["est_tokens"] = int(len(text) / CHARS_PER_TOK)
        r["_sig"] = minhash(shingle_hashes(norm), a, b)
        kept_rows.append(r)
        if (i+1) % 1000 == 0:
            print(f"  hashed {i+1}/{len(rows)}", flush=True)

    rows = kept_rows
    n = len(rows)
    print(f"hashed {n} files. exact-dedup + LSH...", flush=True)

    # exact dedup: mark all but the largest of each sha group
    by_sha = collections.defaultdict(list)
    for idx, r in enumerate(rows): by_sha[r["_sha"]].append(idx)
    exact_dup = set()
    for grp in by_sha.values():
        if len(grp) > 1:
            keep = max(grp, key=lambda i: rows[i]["bytes"])
            exact_dup.update(g for g in grp if g != keep)

    # LSH banding -> candidate union-find (near-dup)
    uf = UF(n)
    sig_mat = np.vstack([r["_sig"] for r in rows])
    for band in range(BANDS):
        cols = sig_mat[:, band*ROWS:(band+1)*ROWS]
        buckets = collections.defaultdict(list)
        for idx in range(n):
            if idx in exact_dup:  # exact dups handled already
                continue
            buckets[cols[idx].tobytes()].append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            rep = members[0]
            rs = rows[rep]["_sig"]
            for m in members[1:]:
                # verify estimated Jaccard via signature agreement
                if np.count_nonzero(rs == rows[m]["_sig"]) / NUM_PERM >= THRESH:
                    uf.union(rep, m)

    # cluster -> representative (largest file in cluster)
    clusters = collections.defaultdict(list)
    for idx in range(n):
        if idx in exact_dup:
            continue
        clusters[uf.find(idx)].append(idx)
    near_dup = set()
    cluster_id = {}
    for cid, (root, members) in enumerate(clusters.items()):
        rep = max(members, key=lambda i: rows[i]["bytes"])
        for m in members:
            cluster_id[m] = cid
            if m != rep:
                near_dup.add(m)

    # decisions + weights
    out = []
    x86_rep_tokens = 0
    kept_k_tokens_nonx86 = 0
    for idx, r in enumerate(rows):
        keep, weight, reason = True, 1.0, "keep"
        if r["era"] == "legacy":
            keep, weight, reason = False, 0.0, "legacy-k3"
        elif idx in exact_dup:
            keep, weight, reason = False, 0.0, "exact-dup"
        elif idx in near_dup:
            keep, weight, reason = False, 0.0, "near-dup"
        elif r["kind"] == "k_docs" and not r.get("has_k"):
            keep, weight, reason = True, 0.5, "prose-md"
        if keep and r["repo"] == X86_REPO:
            reason = "x86-rep"  # weight set after cap computed
            x86_rep_tokens += r["est_tokens"]
        elif keep and r["kind"] == "k_code":
            kept_k_tokens_nonx86 += r["est_tokens"]
        out.append([idx, keep, weight, reason, cluster_id.get(idx, -1)])

    # X86 down-weight so its weighted tokens <= cap * (non-x86 kept K tokens)
    target = X86_TOKEN_CAP * max(kept_k_tokens_nonx86, 1)
    x86_w = min(1.0, target / max(x86_rep_tokens, 1))
    for rec in out:
        if rec[3] == "x86-rep":
            rec[2] = round(x86_w, 4)

    with open(OUT, "w") as fh:
        for r, rec in zip(rows, out):
            idx, keep, weight, reason, cid = rec
            fh.write(json.dumps({
                "repo": r["repo"], "path": r["path"], "kind": r["kind"],
                "era": r["era"], "bytes": r["bytes"], "est_tokens": r["est_tokens"],
                "sha": r["_sha"], "cluster": cid, "keep": keep,
                "weight": weight, "reason": reason,
            }) + "\n")

    # report
    kept = [(rows[i], rec) for i, rec in enumerate(out) if rec[1]]
    reason_ct = collections.Counter(rec[3] for rec in out)
    drop_ct = collections.Counter(rec[3] for rec in out if not rec[1])
    raw_tok = sum(r["est_tokens"] for r in rows)
    kept_tok = sum(r["est_tokens"] for r, rec in kept)
    kept_wtok = sum(int(r["est_tokens"]*rec[2]) for r, rec in kept)
    print("\n================ CLEAN REPORT ================")
    print(f"files: {n} -> kept {len(kept)} ({len(kept)/n*100:.0f}%)")
    print(f"est tokens: raw {raw_tok/1e6:.2f}M -> kept {kept_tok/1e6:.2f}M "
          f"-> weighted {kept_wtok/1e6:.2f}M")
    print("drop reasons:", dict(drop_ct))
    print(f"X86 down-weight factor: {x86_w:.3f}")
    # kept tokens by kind/era
    kk = collections.Counter()
    kw = collections.Counter()
    for r, rec in kept:
        kk[r["kind"]] += r["est_tokens"]; kw[r["kind"]] += int(r["est_tokens"]*rec[2])
    print("kept tokens by kind (raw / weighted):")
    for k in kk: print(f"  {k:8s} {kk[k]/1e6:.2f}M / {kw[k]/1e6:.2f}M")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
