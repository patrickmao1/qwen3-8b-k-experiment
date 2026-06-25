#!/usr/bin/env python3
"""
Scrape public K Framework source (.k) and literate/docs (.md) from known-good
GitHub repos. We deliberately do NOT trust GitHub's `language:KFramework` tag
(it is polluted by LS-DYNA keyword decks, which also use the .k extension).
Instead we seed from the kframework/runtimeverification orgs + curated community
repos, and validate every .k file by actual K syntax markers.

Technique: partial clone (--filter=blob:none --no-checkout) + sparse-checkout of
only *.k / *.md, so huge non-K blobs (e.g. X86-64's 24MB of assembly) are never
downloaded.

Output:
  data/corpus/k_code/<owner>__<repo>/<path>.k      validated real K
  data/corpus/k_docs/<owner>__<repo>/<path>.md     markdown (docs / literate K)
  data/corpus/manifest.jsonl                       one row per harvested file
  data/corpus/repos.jsonl                          one row per repo (license, status)
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
WORK = os.path.join(ROOT, "data", "_clones")
CODE_DIR = os.path.join(CORPUS, "k_code")
DOCS_DIR = os.path.join(CORPUS, "k_docs")
MANIFEST = os.path.join(CORPUS, "manifest.jsonl")
REPOS_LOG = os.path.join(CORPUS, "repos.jsonl")

# Curated seed list of GENUINE K repos. The validator filters junk regardless,
# but this keeps us off multi-hundred-MB non-K clones. "legacy" => old K3/Maude
# syntax (kept but tagged so we can down-weight/exclude for current-K training).
SEEDS = [
    # --- runtimeverification org ---
    ("runtimeverification/k", "framework"),                 # builtins, tutorials, tests
    ("runtimeverification/evm-semantics", "modern"),
    ("runtimeverification/wasm-semantics", "modern"),
    ("runtimeverification/python-semantics", "modern"),
    ("runtimeverification/plutus-core-semantics", "modern"),
    ("runtimeverification/michelson-semantics", "modern"),
    ("runtimeverification/iele-semantics", "modern"),
    ("runtimeverification/erc20-semantics", "modern"),
    ("runtimeverification/erc777-semantics", "modern"),
    ("runtimeverification/blockchain-k-plugin", "modern"),
    ("runtimeverification/verified-smart-contracts", "modern"),
    ("runtimeverification/mir-semantics", "modern"),
    ("runtimeverification/deps-semantics", "modern"),
    # --- kframework org ---
    ("kframework/c-semantics", "modern"),
    ("kframework/javascript-semantics", "modern"),
    ("kframework/java-semantics", "legacy"),
    ("kframework/ocaml-semantics", "modern"),
    ("kframework/llvm-semantics", "modern"),
    ("kframework/llvm-semantics-old", "legacy"),
    ("kframework/aadl-semantics", "modern"),
    ("kframework/alk-semantics", "modern"),
    ("kframework/cink-semantics", "modern"),
    ("kframework/jvm-semantics", "modern"),
    ("kframework/javacard-semantics", "modern"),
    ("kframework/haskell-core-semantics", "modern"),
    ("kframework/p4-semantics", "modern"),
    ("kframework/vyper-semantics", "modern"),
    ("kframework/solidity-semantics", "modern"),
    ("kframework/boogie-semantics", "modern"),
    ("kframework/eei-semantics", "modern"),
    ("kframework/orc-semantics", "modern"),
    ("kframework/modelink-semantics", "modern"),
    ("kframework/k-in-k", "modern"),
    ("kframework/kat", "modern"),
    ("kframework/X86-64-semantics", "modern"),
    ("kframework/semantic-approaches", "modern"),
    ("kframework/klab", "modern"),
    ("kframework/k-legacy", "legacy"),
    # --- community ---
    ("alk-language/k-semantics", "modern"),
]

K_BLOCK_RE = re.compile(r"```\s*k\b", re.I)

def is_real_k(text: str) -> bool:
    """True if a .k file looks like K Framework source (not an LS-DYNA deck)."""
    if "endmodule" in text:
        return True
    if re.search(r"\bsyntax\b[^\n]*::=", text):
        return True
    if re.search(r"\brule\b", text) and "=>" in text:
        return True
    if re.search(r'\brequires\b\s+"', text) and "module" in text:
        return True
    return False

def md_has_k(text: str) -> bool:
    return bool(K_BLOCK_RE.search(text)) or ("endmodule" in text) or \
           bool(re.search(r"\bsyntax\b[^\n]*::=", text))

def http_json(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "k-scraper"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception:
        return None

def run(cmd, cwd=None, timeout=600):
    return subprocess.run(cmd, cwd=cwd, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def harvest_repo(full_name, era, mf, rf):
    owner, repo = full_name.split("/")
    slug = f"{owner}__{repo}"
    dest = os.path.join(WORK, slug)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    url = f"https://github.com/{full_name}"

    # license (best-effort)
    meta = http_json(f"https://api.github.com/repos/{full_name}") or {}
    lic = ((meta.get("license") or {}).get("spdx_id")) if meta else None

    t0 = time.time()
    r = run(["git", "clone", "--depth", "1", "--filter=blob:none",
             "--no-checkout", url, dest], timeout=300)
    if r.returncode != 0:
        rf.write(json.dumps({"repo": full_name, "status": "clone_fail",
                             "log": r.stdout[-400:]}) + "\n"); rf.flush()
        print(f"[FAIL clone] {full_name}: {r.stdout[-200:]}", flush=True)
        return 0, 0
    run(["git", "sparse-checkout", "init", "--no-cone"], cwd=dest)
    run(["git", "sparse-checkout", "set", "--no-cone", "/**/*.k", "/**/*.md",
         "*.k", "*.md"], cwd=dest)
    co = run(["git", "checkout"], cwd=dest, timeout=300)
    if co.returncode != 0:
        # fall back to fetching the requested files anyway; checkout may warn
        pass

    n_k = n_md = 0
    k_bytes = md_bytes = 0
    for dirpath, _dirs, files in os.walk(dest):
        if "/.git" in dirpath:
            continue
        for fn in files:
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, dest)
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except Exception:
                continue
            nbytes = len(text.encode("utf-8", "replace"))
            if fn.endswith(".k"):
                if not is_real_k(text):
                    continue
                out = os.path.join(CODE_DIR, slug, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8") as oh:
                    oh.write(text)
                mf.write(json.dumps({"repo": full_name, "era": era, "path": rel,
                    "kind": "k_code", "bytes": nbytes, "license": lic}) + "\n")
                n_k += 1; k_bytes += nbytes
            elif fn.endswith(".md"):
                out = os.path.join(DOCS_DIR, slug, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8") as oh:
                    oh.write(text)
                mf.write(json.dumps({"repo": full_name, "era": era, "path": rel,
                    "kind": "k_docs", "bytes": nbytes, "license": lic,
                    "has_k": md_has_k(text)}) + "\n")
                n_md += 1; md_bytes += nbytes
    mf.flush()
    shutil.rmtree(dest, ignore_errors=True)  # keep only harvested text
    rf.write(json.dumps({"repo": full_name, "status": "ok", "era": era,
        "license": lic, "k_files": n_k, "k_bytes": k_bytes,
        "md_files": n_md, "md_bytes": md_bytes,
        "secs": round(time.time() - t0, 1)}) + "\n"); rf.flush()
    print(f"[ok] {full_name:48s} k={n_k:>4} ({k_bytes//1024:>5}KB) "
          f"md={n_md:>4} ({md_bytes//1024:>5}KB) {time.time()-t0:.0f}s", flush=True)
    return k_bytes, md_bytes

def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(CODE_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    tot_k = tot_md = 0
    with open(MANIFEST, "w") as mf, open(REPOS_LOG, "w") as rf:
        for i, (full, era) in enumerate(SEEDS, 1):
            print(f"--- ({i}/{len(SEEDS)}) {full} [{era}] ---", flush=True)
            try:
                kb, mb = harvest_repo(full, era, mf, rf)
                tot_k += kb; tot_md += mb
            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] {full}", flush=True)
                rf.write(json.dumps({"repo": full, "status": "timeout"}) + "\n"); rf.flush()
            except Exception as e:
                print(f"[ERR] {full}: {e}", flush=True)
                rf.write(json.dumps({"repo": full, "status": f"err:{e}"}) + "\n"); rf.flush()
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"\n=== DONE. total .k = {tot_k/1e6:.2f} MB, .md = {tot_md/1e6:.2f} MB ===", flush=True)

if __name__ == "__main__":
    main()
