import random
from kcpt import paths, data

def test_doc_path_maps_repo_slashes():
    row = {"kind": "k_code", "repo": "runtimeverification/evm-semantics", "path": "evm.k"}
    p = paths.doc_path(row)
    assert p.endswith("/corpus_final/k_code/runtimeverification__evm-semantics/evm.k")

def test_env_has_nix_on_path():
    assert ".nix-profile/bin" in paths.ENV["PATH"]

def test_pack_token_lists_chunks_and_inserts_eos():
    lists = [[1, 2, 3], [4, 5]]
    blocks = data.pack_token_lists(lists, seq_len=3, eos_id=0)
    # stream = 1,2,3,0,4,5,0 -> blocks of 3: [1,2,3],[0,4,5]; trailing [0] dropped
    assert blocks == [[1, 2, 3], [0, 4, 5]]

def test_pack_token_lists_drops_partial_final_block():
    blocks = data.pack_token_lists([[1, 2]], seq_len=4, eos_id=0)
    assert blocks == []  # 1,2,0 < 4

def test_weighted_copies_floor_plus_fractional():
    rng = random.Random(0)
    counts = [data.weighted_copies(1.5, rng) for _ in range(1000)]
    assert all(c in (1, 2) for c in counts)
    assert 1 in counts and 2 in counts  # both branches hit

def test_weighted_copies_integer_weight_exact():
    rng = random.Random(0)
    assert all(data.weighted_copies(2.0, rng) == 2 for _ in range(50))
