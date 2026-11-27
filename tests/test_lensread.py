import torch

from rhjl.lensread import frac_explained, hf_to_jlens, nn_omp


def test_layer_mapping():
    assert hf_to_jlens(12) == 11
    assert hf_to_jlens(1) == 0
    assert hf_to_jlens(24) is None


def test_nn_omp_recovers_sparse_nonneg_combination():
    g = torch.Generator().manual_seed(0)
    D = torch.randn(500, 64, generator=g)
    x = 2.0 * D[3] + 0.5 * D[17]
    ids, c, fit = nn_omp(x, D, k=5)
    assert set(ids[:2]) == {3, 17}
    assert frac_explained(x, fit) > 0.999
    assert (c >= 0).all()


def test_nn_omp_ignores_negative_directions():
    D = torch.eye(8)
    ids, _, fit = nn_omp(-D[2], D, k=3)
    assert ids == []
    assert frac_explained(-D[2], fit) == 0.0
