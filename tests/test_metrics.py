import math

from kcpt import metrics


def test_perplexity_from_loss_basic():
    assert math.isclose(metrics.perplexity_from_loss(0.0), 1.0)
    assert math.isclose(metrics.perplexity_from_loss(1.0), math.e, rel_tol=1e-6)

def test_perplexity_from_loss_clamps():
    # huge loss clamps to exp(20), not inf
    assert metrics.perplexity_from_loss(1e9) == math.exp(20)
