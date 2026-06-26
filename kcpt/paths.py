import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(DATA, "splits")
CORPUS_FINAL = os.path.join(DATA, "corpus_final")
PACKED = os.path.join(DATA, "packed")
BENCH = os.path.join(DATA, "benchmark")
FINAL_MANIFEST = os.path.join(DATA, "final_manifest.jsonl")
OUTPUTS = os.path.join(ROOT, "outputs")
LOGS = os.path.join(ROOT, "logs")

# kompile/krun live in the nix profile; subprocesses need it on PATH.
ENV = dict(os.environ)
ENV["PATH"] = os.path.expanduser("~/.nix-profile/bin") + ":" + ENV.get("PATH", "")


def doc_path(row):
    """Manifest row -> on-disk corpus file path."""
    return os.path.join(CORPUS_FINAL, row["kind"], row["repo"].replace("/", "__"), row["path"])
