"""Versioned integer-CDF sampling, including support and batch scheduling."""
import numpy as np
import pytest
import torch

from junqi_rl.analysis.random_eval import (
    WEIGHTED_MASS_SCALE, WEIGHTED_SAMPLING_VERSION, _game_uniform,
    _integer_weight_masses, _sample_uniform, _sample_weights,
)


def test_tiny_positive_support_survives_and_illegal_weights_have_zero_mass():
    weights = torch.tensor([[1.0, 1e-300, 0.0, 100.0], [1e-200, 2e-200, 0., 1.]], dtype=torch.float64)
    legal = torch.tensor([[True, True, True, False], [True, True, True, False]])
    masses = _integer_weight_masses(weights, legal)
    assert masses.dtype == torch.int64
    assert masses.tolist() == [[WEIGHTED_MASS_SCALE, 1, 0, 0],
                                [WEIGHTED_MASS_SCALE // 2, WEIGHTED_MASS_SCALE, 0, 0]]
    assert WEIGHTED_SAMPLING_VERSION.endswith('.v2')


def test_quantization_error_is_bounded_and_totals_are_exact_float64_integers():
    rng = torch.Generator().manual_seed(819)
    weights = torch.exp(-torch.rand(4, 16641, generator=rng, dtype=torch.float64) * 100)
    legal = torch.rand(4, 16641, generator=rng) > .5
    masses = _integer_weight_masses(weights, legal)
    true_weights = weights.masked_fill(~legal, 0)
    probabilities = true_weights / true_weights.sum(-1, keepdim=True)
    quantized = masses.double() / masses.sum(-1, keepdim=True)
    total_variation = .5 * (probabilities - quantized).abs().sum(-1)
    assert (total_variation <= legal.sum(-1) / WEIGHTED_MASS_SCALE).all()
    cumulative = masses.cumsum(-1)
    torch.testing.assert_close(cumulative.double().long(), cumulative, rtol=0, atol=0)
    assert (cumulative[:, -1] < 2**53).all()


def test_weighted_samples_are_batch_partition_and_order_independent():
    rng = torch.Generator().manual_seed(273)
    weights = torch.rand(137, 173, generator=rng).pow(3)
    legal = torch.rand(137, 173, generator=rng) > .4
    legal[:, 9] = True
    seeds = torch.arange(137) + 773
    for ply, stream in ((0, 19), (4000, 1)):
        expected = _sample_weights(weights, legal, seeds, ply, stream)
        actual = torch.cat([_sample_weights(weights[a:b], legal[a:b], seeds[a:b], ply, stream)
                            for a, b in ((0, 1), (1, 96), (96, 137))])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        permutation = torch.randperm(137, generator=rng)
        permuted = _sample_weights(weights[permutation], legal[permutation], seeds[permutation], ply, stream)
        torch.testing.assert_close(permuted, expected[permutation], rtol=0, atol=0)
        assert legal.gather(1, expected.long()[:, None]).all()


def test_nonuniform_samples_follow_the_weighted_distribution():
    count = 30000
    weights = torch.tensor([[1., 3., 6., 100.]]).expand(count, -1)
    legal = torch.tensor([[True, True, True, False]]).expand(count, -1)
    actions = _sample_weights(weights, legal, torch.arange(count), 317, 19)
    observed = torch.bincount(actions.long(), minlength=4).double() / count
    np.testing.assert_allclose(observed.numpy(), [.1, .3, .6, 0], rtol=0, atol=.01)


def test_uniform_opponent_keeps_historical_float32_output():
    rng = torch.Generator().manual_seed(82)
    legal = torch.rand(8192, 37, generator=rng) > .4
    legal[:, 5] = True
    seeds = torch.arange(len(legal)) + 6100200
    for ply in (0, 31, 3999):
        cumulative = legal.float().cumsum(-1)
        threshold = _game_uniform(seeds, ply)[:, None] * cumulative[:, -1:]
        expected = (cumulative > threshold).to(torch.int32).argmax(-1).to(torch.int32)
        torch.testing.assert_close(_sample_uniform(legal, seeds, ply), expected, rtol=0, atol=0)


def test_action_count_cannot_overflow_the_exact_mass_contract():
    with pytest.raises(ValueError, match='exact float64'):
        _integer_weight_masses(torch.ones(1, 32769), torch.ones(1, 32769, dtype=torch.bool))
