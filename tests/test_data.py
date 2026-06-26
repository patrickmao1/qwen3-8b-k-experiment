from kcpt import paths

def test_doc_path_maps_repo_slashes():
    row = {"kind": "k_code", "repo": "runtimeverification/evm-semantics", "path": "evm.k"}
    p = paths.doc_path(row)
    assert p.endswith("/corpus_final/k_code/runtimeverification__evm-semantics/evm.k")

def test_env_has_nix_on_path():
    assert ".nix-profile/bin" in paths.ENV["PATH"]
