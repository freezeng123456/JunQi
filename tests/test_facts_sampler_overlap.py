"""Submitting scores before host feasibility keeps samples and RNG identical."""
import numpy as np
import pytest
import torch

from junqi_rl.belief.constraints import ConstraintBudgetExceeded, compile_constraints
from tests.test_facts_joint_belief import fixture, tiny_model


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA for device-local exact parity'))])
@pytest.mark.parametrize('samples', [1, 2, 4])
def test_overlap_matches_all_logits_samples_rng_masks_witnesses_and_nodes(device, samples):
    _, inputs, _, obs = fixture()
    model = tiny_model().to(device)
    obs = obs.to(device)
    records = []
    for overlap, backend in [(False, 'scan'), (True, 'incremental')]:
        solver = compile_constraints(inputs, inventory_backend=backend)
        generator = torch.Generator(device=device).manual_seed(912)
        logits = []
        hook = model.head.register_forward_hook(lambda _m, _i, out: logits.append(out.detach().clone()))
        try:
            labels = model.sample(obs, inputs=inputs, n_samples=samples, constraints=[solver],
                                  generator=generator, overlap_constraints=overlap)
        finally:
            hook.remove()
        records.append((labels, generator.get_state(), torch.cat(logits), solver))
    for first, second in zip(records[0][:3], records[1][:3]):
        assert torch.equal(first, second)
    reference, candidate = records[0][3], records[1][3]
    assert reference.stats == candidate.stats
    assert reference._cache == candidate._cache
    assert reference.matching_cache_stats == candidate.matching_cache_stats
    assert reference._matching_cache == candidate._matching_cache
    assert set(reference._masks) == set(candidate._masks)
    for key in reference._masks:
        np.testing.assert_array_equal(reference._masks[key], candidate._masks[key])


@pytest.mark.parametrize('overlap', [False, True])
def test_constraint_failures_do_not_advance_random_generator(overlap):
    _, inputs, _, obs = fixture()
    generator = torch.Generator().manual_seed(101)
    before = generator.get_state().clone()

    class Failure:
        def allowed_next(self, prefix):
            raise ConstraintBudgetExceeded(f'budget at prefix {len(prefix)}')

    with pytest.raises(ConstraintBudgetExceeded, match='prefix 0'):
        tiny_model().sample(obs, inputs=inputs, n_samples=1, constraints=[Failure()],
                            generator=generator, overlap_constraints=overlap)
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize('overlap', [False, True])
def test_nonfinite_scores_fail_before_random_draw(overlap):
    _, inputs, _, obs = fixture()
    model = tiny_model()
    with torch.no_grad():
        model.head.bias.fill_(float('nan'))
    generator = torch.Generator().manual_seed(101)
    before = generator.get_state().clone()
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        model.sample(obs, inputs=inputs, n_samples=1, generator=generator,
                     overlap_constraints=overlap)
    assert torch.equal(before, generator.get_state())
