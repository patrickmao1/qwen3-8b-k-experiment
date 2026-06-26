import collections
import math


def perplexity_from_loss(loss, clamp=20.0):
    return math.exp(min(loss, clamp))


def corpus_perplexity(model, tok, rows, max_seq_length, *, doc_path_fn, max_docs=0):
    """Per-document held-out perplexity, overall + per repo. Pure of globals:
    caller supplies rows and a doc_path_fn(row)->path."""
    import torch

    if max_docs:
        rows = rows[:max_docs]
    by = collections.defaultdict(lambda: [0.0, 0])
    tot = [0.0, 0]
    for r in rows:
        p = doc_path_fn(r)
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids), max_seq_length):
            chunk = ids[i:i + max_seq_length]
            if len(chunk) < 2:
                continue
            t = torch.tensor([chunk], device=model.device)
            with torch.no_grad():
                loss = model(t, labels=t).loss.item()
            n = len(chunk) - 1
            by[r["repo"]][0] += loss * n
            by[r["repo"]][1] += n
            tot[0] += loss * n
            tot[1] += n
    per_lang = {repo: round(perplexity_from_loss(s / n), 3) for repo, (s, n) in by.items() if n}
    overall = round(perplexity_from_loss(tot[0] / max(tot[1], 1)), 3)
    return {"overall_perplexity": overall, "per_language": dict(sorted(per_lang.items()))}
