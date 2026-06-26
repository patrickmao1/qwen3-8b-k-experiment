#!/usr/bin/env python3
"""
One-shot raw completion against a selected model.

The model is a *base* (completion) model, so the prompt is fed verbatim and the
model continues it. Only the completion is printed to stdout (pipeable); the
prompt itself is not echoed.

Prompt source, in precedence order: --file, positional arg, then stdin.

Usage:
  bash scripts/prompt.sh -f prompts/imp.k                       # prompt from file
  bash scripts/prompt.sh "module IMP-SYNTAX"                    # prompt as arg
  echo "module FOO" | bash scripts/prompt.sh                    # prompt from stdin
  bash scripts/prompt.sh -f prompts/imp.k --model unsloth/Qwen3-8B-Base
"""

import argparse
import sys

from kcpt.model import load_model

DEFAULT_MODEL = "outputs/cpt-qwen3-8b/adapter"  # latest CPT adapter


def read_prompt(args):
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return f.read()
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def generate(model, tok, prompt, max_new, temperature):
    import torch

    ids = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        gen = model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", nargs="?", help="prompt text (or use --file / stdin)")
    ap.add_argument("-f", "--file", help="read the prompt from this file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy/deterministic; >0 enables sampling",
    )
    args = ap.parse_args()

    prompt = read_prompt(args)
    if not prompt.strip():
        sys.exit("ERROR: empty prompt (pass text as an arg, --file, or via stdin)")

    model, tok = load_model(args.model, args.max_seq_length)
    completion = generate(model, tok, prompt, args.max_new, args.temperature)
    sys.stdout.write(completion)
    if not completion.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
