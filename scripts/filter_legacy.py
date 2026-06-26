#!/usr/bin/env python3
"""
Final corpus filter (decision A): PURGE compiler-confirmed deprecated K, apply
only the confirmed-safe `require "..."` -> `requires "..."` mechanical fix.
No K3->K7 migration (unverifiable, risks teaching plausible-but-broken K).

Deprecation markers were each validated against the installed kompile (v7.1.337):
  module <X> is      -> Outer Parser error
  including ...       -> (old import) replaced by `imports`
  <rhs> when <cond>   -> Inner Parser: unexpected token 'when'
  syntax K ::= ...    -> Cannot add constructors to hooked sort K
  Maude kw (op/eq/..) -> not K
  require "..."       -> Outer Parser error  (SAFE fix: add the trailing 's')

Reads keep=true rows from clean_manifest.jsonl; writes the surviving (and fixed)
files to data/corpus_final/<kind>/<slug>/<path> plus final_manifest.jsonl.
"""
import collections
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
SRC = {"k_code": os.path.join(CORPUS, "k_code"),
       "k_docs": os.path.join(CORPUS, "k_docs")}
CLEAN = os.path.join(CORPUS, "clean_manifest.jsonl")
FINAL_DIR = os.path.join(ROOT, "data", "corpus_final")
FINAL_MAN = os.path.join(ROOT, "data", "final_manifest.jsonl")
CHARS_PER_TOK = 3.5

_block = re.compile(r"/\*.*?\*/", re.S)
_line = re.compile(r"//[^\n]*")
_str = re.compile(r'"(?:\\.|[^"\\])*"')
def strip_code(t): return _str.sub('""', _line.sub('', _block.sub(' ', t)))

# deep-legacy detectors for .k (run on comment/string-stripped text)
DEEP_CODE = {
    "module_is": re.compile(r"\bmodule\s+[A-Za-z0-9_'-]+\s+is\b"),
    "including": re.compile(r"(?m)^\s*including\b"),
    "when":      re.compile(r"(?<![.\w])when(?![\w.])"),   # excludes dotted `.when`
    "maude_kw":  re.compile(r"(?m)^\s*(?:op|ops|eq|ceq|rl|crl|sorts?|mb)\b"),
    "syntax_K":  re.compile(r"\bsyntax\s+K\s+::="),
}
# prose-safe detectors for .md (raw text; skip `when`/maude to avoid prose FPs)
DEEP_DOCS = {
    "module_is": DEEP_CODE["module_is"],
    "including": DEEP_CODE["including"],
    "syntax_K":  DEEP_CODE["syntax_K"],
}
REQUIRE_FIX = re.compile(r'(?m)^(\s*)require(\s+")')

def src_path(r): return os.path.join(SRC[r["kind"]], r["repo"].replace("/", "__"), r["path"])
def dst_path(r): return os.path.join(FINAL_DIR, r["kind"], r["repo"].replace("/", "__"), r["path"])

def main():
    rows = [json.loads(line) for line in open(CLEAN)]
    kept = [r for r in rows if r["keep"]]
    purged_by = collections.Counter()
    purged_tok = collections.Counter()
    purged_repo = collections.Counter()
    n_fix = 0
    out_rows = []
    tok_in = tok_out = 0
    for r in kept:
        try:
            raw = open(src_path(r), encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            continue
        tok_in += r["est_tokens"]
        if r["kind"] == "k_code":
            det = DEEP_CODE
            text = strip_code(raw)
        else:
            det = DEEP_DOCS
            text = raw
        hits = [m for m, rx in det.items() if rx.search(text)]
        if hits:
            purged_by[hits[0]] += 1
            for h in hits:
                pass
            purged_tok[hits[0]] += r["est_tokens"]
            purged_repo[r["repo"]] += r["est_tokens"]
            continue
        # apply safe require->requires fix
        fixed, nsub = REQUIRE_FIX.subn(r'\1requires\2', raw)
        if nsub:
            n_fix += 1
        os.makedirs(os.path.dirname(dst_path(r)), exist_ok=True)
        with open(dst_path(r), "w", encoding="utf-8") as fh:
            fh.write(fixed)
        est = int(len(fixed) / CHARS_PER_TOK)
        tok_out += est
        out_rows.append({"repo": r["repo"], "path": r["path"], "kind": r["kind"],
                         "era": r["era"], "bytes": len(fixed.encode("utf-8", "replace")),
                         "est_tokens": est, "weight": r["weight"],
                         "require_fixed": bool(nsub), "reason": r["reason"]})
    with open(FINAL_MAN, "w") as fh:
        for o in out_rows:
            fh.write(json.dumps(o) + "\n")

    # report
    code = [o for o in out_rows if o["kind"] == "k_code"]
    docs = [o for o in out_rows if o["kind"] == "k_docs"]
    def wt(lst): return sum(o["est_tokens"]*o["weight"] for o in lst)
    print("================ FINAL FILTER (decision A) ================")
    print(f"kept files in : {len(kept)}   -> survived: {len(out_rows)}   "
          f"purged: {sum(purged_by.values())}")
    print(f"require->requires fixes applied to {n_fix} files")
    print("\npurged by marker (first-hit): " + ", ".join(
        f"{k}={purged_by[k]}f/{purged_tok[k]//1000}k" for k in purged_by))
    print(f"\ntokens: kept-in {tok_in/1e6:.2f}M -> final {tok_out/1e6:.2f}M "
          f"(purged {(tok_in-tok_out)/1e6:.2f}M)")
    print(f"final .k : {len(code):>4} files  raw {sum(o['est_tokens'] for o in code)/1e6:.2f}M  "
          f"weighted {wt(code)/1e6:.2f}M")
    print(f"final .md : {len(docs):>4} files  raw {sum(o['est_tokens'] for o in docs)/1e6:.2f}M  "
          f"weighted {wt(docs)/1e6:.2f}M")
    print(f"TOTAL weighted: {(wt(code)+wt(docs))/1e6:.2f}M tokens")
    print("\npurged tokens by repo (top 10):")
    for repo, t in purged_repo.most_common(10):
        print(f"  {repo:44s} {t//1000:>5}k")
    print(f"\nwrote {FINAL_MAN} and data/corpus_final/")

if __name__ == "__main__":
    main()
