#!/usr/bin/env python3
"""
Robust evaluation for the K-CPT model. Three layers, all base-vs-CPT comparable:

  L1 Perplexity   : held-out test-split perplexity, OVERALL + per-language(repo).
  L2 Compile/exec : completion benchmark (data/benchmark/*). Feed each task's
                    prefix to the model, let it COMPLETE the rules, then kompile
                    the assembled definition and krun its sample programs vs the
                    expected outputs. Metrics: compile-rate, exec-correctness.
  L3 Regression   : deprecated-syntax emission rate in completions + a kompile
                    error-type histogram.

Because the model is a *base* (completion) model, L2 is prefix-completion, not
instruction-prompted.

Usage:
  bash scripts/eval.sh --model unsloth/Qwen3-8B-Base --label base
  bash scripts/eval.sh --model outputs/cpt-qwen3-8b/adapter --label cpt
  python scripts/eval_suite.py --compare outputs/eval/base.json outputs/eval/cpt.json
Outputs: outputs/eval/<label>.json
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from kcpt import paths
from kcpt.metrics import corpus_perplexity
from kcpt.model import load_model

BENCH = paths.BENCH
OUTDIR = os.path.join(paths.OUTPUTS, "eval")
ENV = paths.ENV
SENTINEL = "// >>> COMPLETE THE RULES BELOW <<<"

# deprecated-syntax markers (compiler-confirmed), run on comment/string-stripped text
_block = re.compile(r"/\*.*?\*/", re.S)
_line = re.compile(r"//[^\n]*")
_str = re.compile(r'"(?:\\.|[^"\\])*"')


def strip_code(t):
    return _str.sub('""', _line.sub("", _block.sub(" ", t)))


DEPRECATED = {
    "when": re.compile(r"(?<![.\w])when(?![\w.])"),
    "syntax_K": re.compile(r"\bsyntax\s+K\s+::="),
    "module_is": re.compile(r"\bmodule\s+[A-Za-z0-9_'-]+\s+is\b"),
    "including": re.compile(r"(?m)^\s*including\b"),
}


def uses_deprecated(code):
    s = strip_code(code)
    return [k for k, rx in DEPRECATED.items() if rx.search(s)]


# ---------------- L2 benchmark ----------------
def load_tasks():
    tasks = []
    if not os.path.isdir(BENCH):
        return tasks
    for d in sorted(os.listdir(BENCH)):
        td = os.path.join(BENCH, d)
        full = os.path.join(td, "full.k")
        meta = os.path.join(td, "meta.json")
        if not (os.path.isfile(full) and os.path.isfile(meta)):
            continue
        text = open(full, encoding="utf-8", errors="replace").read()
        if SENTINEL not in text:
            continue
        prefix, _ = text.split(SENTINEL, 1)
        tasks.append(
            {
                "dir": td,
                "name": d,
                "prefix": prefix + SENTINEL + "\n",
                "meta": json.load(open(meta)),
            }
        )
    return tasks


def assemble(prefix, completion):
    cand = prefix + completion
    idx = cand.find("endmodule", len(prefix))
    return cand[: idx + 9] + "\n" if idx != -1 else cand  # truncate at first endmodule


def modules(text):
    decls = re.findall(r"(?m)^\s*module\s+([A-Za-z0-9_'-]+)", text)
    syn = next((m for m in decls if m.endswith("-SYNTAX")), None)
    main = next(
        (m for m in reversed(decls) if not m.endswith("-SYNTAX")),
        decls[-1] if decls else None,
    )
    return main, syn


def verify_task(task, completion):
    """kompile assembled candidate + krun its programs. Returns a result dict."""
    res = {
        "name": task["name"],
        "category": task["meta"].get("category", "?"),
        "deprecated": uses_deprecated(completion),
        "compiled": False,
        "kompile_error": None,
        "programs_total": 0,
        "programs_passed": 0,
    }
    cand = assemble(task["prefix"], completion)
    res["has_endmodule"] = "endmodule" in completion
    main, syn = modules(cand)
    with tempfile.TemporaryDirectory() as tmp:
        kf = os.path.join(tmp, "cand.k")
        open(kf, "w").write(cand)
        cmd = ["kompile", "--backend", "llvm", "-o", os.path.join(tmp, "kompiled"), kf]
        if main:
            cmd += ["--main-module", main]
        if syn:
            cmd += ["--syntax-module", syn]
        try:
            r = subprocess.run(
                cmd, cwd=tmp, env=ENV, capture_output=True, text=True, timeout=240
            )
        except subprocess.TimeoutExpired:
            res["kompile_error"] = "timeout"
            return res
        if r.returncode != 0:
            err = re.search(r"\[Error\][^\n]*", r.stdout + r.stderr)
            res["kompile_error"] = (err.group(0)[:120] if err else "unknown")[:120]
            return res
        res["compiled"] = True
        kdir = os.path.join(tmp, "kompiled")
        for prog in task["meta"].get("programs", []):
            res["programs_total"] += 1
            pf = os.path.join(task["dir"], prog["file"])
            if not os.path.isfile(pf):
                continue
            try:
                kr = subprocess.run(
                    ["krun", "--definition", kdir, pf],
                    env=ENV,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if prog.get("expected", "\0") in kr.stdout:
                    res["programs_passed"] += 1
            except subprocess.TimeoutExpired:
                pass
    return res


def generate_completions(model, tok, tasks, samples, temperature, max_new_cap):
    import torch

    outs = {}
    for t in tasks:
        # fair per-task budget: scale to the reference completion length (+headroom)
        full = open(
            os.path.join(t["dir"], "full.k"), encoding="utf-8", errors="replace"
        ).read()
        ref = full.split(SENTINEL, 1)[1] if SENTINEL in full else ""
        ref_len = len(tok(ref, add_special_tokens=False)["input_ids"]) if ref else 256
        budget = min(max_new_cap, max(256, int(ref_len * 1.6) + 32))
        ids = tok(t["prefix"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=budget,
                do_sample=samples > 1,
                temperature=temperature if samples > 1 else None,
                num_return_sequences=samples,
                pad_token_id=tok.eos_token_id,
            )
        comps = [
            tok.decode(g[ids["input_ids"].shape[1] :], skip_special_tokens=True)
            for g in gen
        ]
        outs[t["name"]] = comps
    return outs


def run_benchmark(model, tok, samples, temperature, max_new, jobs):
    import collections

    tasks = load_tasks()
    if not tasks:
        return {"note": "no benchmark tasks found", "n_tasks": 0}
    comps = generate_completions(model, tok, tasks, samples, temperature, max_new)
    # verify in parallel: a task passes@k if ANY of its samples compiles (exec = best sample)
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        fut = {}
        for t in tasks:
            for s_i, comp in enumerate(comps[t["name"]]):
                fut[ex.submit(verify_task, t, comp)] = (t["name"], s_i)
        per_task = collections.defaultdict(list)
        for f in as_completed(fut):
            name, _ = fut[f]
            per_task[name].append(f.result())
    # collapse samples -> best per task
    cat = collections.defaultdict(lambda: [0, 0])  # category -> [compiled, total]
    err_hist = collections.Counter()
    deprecated_ct = 0
    prog_pass = prog_tot = 0
    compiled_tasks = 0
    for name, rs in per_task.items():
        best = max(rs, key=lambda r: (r["compiled"], r["programs_passed"]))
        results.append(best)
        c = best["category"]
        cat[c][1] += 1
        if best["compiled"]:
            cat[c][0] += 1
            compiled_tasks += 1
            prog_pass += best["programs_passed"]
            prog_tot += best["programs_total"]
        elif best["kompile_error"]:
            err_hist[best["kompile_error"].split(":")[0]] += 1
        if any(r["deprecated"] for r in rs):
            deprecated_ct += 1
    n = len(per_task)
    return {
        "n_tasks": n,
        "samples_per_task": samples,
        "compile_rate": round(compiled_tasks / n, 3),
        "exec_correctness": round(prog_pass / prog_tot, 3) if prog_tot else None,
        "deprecated_emission_rate": round(deprecated_ct / n, 3),
        "by_category": {
            c: {"compile_rate": round(v[0] / v[1], 3), "n": v[1]}
            for c, v in sorted(cat.items())
        },
        "kompile_error_histogram": dict(err_hist.most_common()),
        "tasks": sorted(results, key=lambda r: r["name"]),
    }


# ---------------- compare ----------------
def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    la, lb = a.get("label", "A"), b.get("label", "B")
    print(f"\n=== {la}  vs  {lb} ===")
    print(f"{'metric':32s} {la:>12} {lb:>12}  delta")

    def row(name, va, vb, better_lower=False):
        if va is None or vb is None:
            print(f"{name:32s} {str(va):>12} {str(vb):>12}")
            return
        d = vb - va
        arrow = "↓" if d < 0 else ("↑" if d > 0 else "·")
        print(f"{name:32s} {va:>12} {vb:>12}  {d:+.3f} {arrow}")

    row(
        "overall_perplexity",
        a["L1"]["overall_perplexity"],
        b["L1"]["overall_perplexity"],
        True,
    )
    row("compile_rate", a["L2"]["compile_rate"], b["L2"]["compile_rate"])
    row(
        "exec_correctness",
        a["L2"].get("exec_correctness"),
        b["L2"].get("exec_correctness"),
    )
    row(
        "deprecated_emission_rate",
        a["L2"]["deprecated_emission_rate"],
        b["L2"]["deprecated_emission_rate"],
        True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument(
        "--max-new",
        type=int,
        default=2048,
        help="upper-bound generation budget; actual budget is per-task (1.6x reference length)",
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ppl-max-docs", type=int, default=0)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    os.makedirs(OUTDIR, exist_ok=True)
    model, tok = load_model(args.model, args.max_seq_length)
    rows = [json.loads(line) for line in open(os.path.join(paths.SPLITS, "test.jsonl"))]
    L1 = {} if args.skip_ppl else corpus_perplexity(
        model, tok, rows, args.max_seq_length, doc_path_fn=paths.doc_path, max_docs=args.ppl_max_docs)
    L2 = run_benchmark(
        model, tok, args.samples, args.temperature, args.max_new, args.jobs
    )
    report = {"label": args.label, "model": args.model, "L1": L1, "L2": L2}
    out = os.path.join(OUTDIR, f"{args.label}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n=== EVAL [{args.label}] ===")
    if L1:
        print("overall perplexity:", L1["overall_perplexity"])
    print(
        "compile_rate:",
        L2.get("compile_rate"),
        "| exec_correctness:",
        L2.get("exec_correctness"),
        "| deprecated_emission_rate:",
        L2.get("deprecated_emission_rate"),
    )
    print("by_category:", L2.get("by_category"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
