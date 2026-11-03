"""GPU-only acceptance for independent setup sampling and reference resume."""
import copy
import numpy as np
import pytest
import torch

cuda=pytest.importorskip('junqi_cuda')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture
def isolated_world():
    from junqi_rl.gpu_rollout import GpuRollout
    world=GpuRollout(128,include_global=False,identity_encoding='known_only')
    world.reset(seed_base=11)
    original=copy.deepcopy(world.state_dict())
    try:
        yield world
    finally:
        # Native reset banks are process-wide, including for later legacy
        # tests which explicitly exercise the default paired-bank hash.
        world.load_state_dict(original)


def test_independent_seat_reset_is_product_of_banks_and_resumes(isolated_world):
    from junqi_rl.gpu_rollout import GpuRollout,_build_setup_pool
    world=isolated_world
    pool=_build_setup_pool(8,seed=221)
    from junqi_rl.networks.arrangement_net import PIECE_TYPE_VALUE_TO_VOCAB_IDX
    vocab=np.vectorize(PIECE_TYPE_VALUE_TO_VOCAB_IDX.__getitem__)(pool.reshape(32,30))
    samples=torch.nn.functional.one_hot(torch.tensor(vocab,device='cuda'),13).float()
    world.refresh_setup_pool_from_arrangements(samples,torch.arange(32,device='cuda')%4,
        pool_size=0,independent_seats=True)
    world.reset_from_setup_pool(seed=25)
    saved=copy.deepcopy(world.state_dict())
    snapshot=world.state.checkpoint_to_host()
    # Snapshot API stores native arrays by field name.
    snapshot=snapshot['arrays']
    key='d_piece_type_arr'
    types=np.asarray(snapshot[key]).reshape(128,4,30)
    banks=pool.reshape(8,4,30)
    indices=np.stack([np.array([next(i for i in range(8) if np.array_equal(row,banks[i,s]))
                                for row in types[:,s]]) for s in range(4)],axis=1)
    assert (indices!=indices[:,:1]).any(1).mean()>.9
    cuda.upload_setup_pool(pool,False)
    world.reset_from_setup_pool(seed=25)
    again=world.state.checkpoint_to_host()['arrays']
    types=np.asarray(again[key]).reshape(128,120)
    assert all(any(np.array_equal(row,p) for p in pool) for row in types)
    world.load_state_dict(saved)
    assert world.independent_setup_seats
    world.reset_from_setup_pool(seed=25)
    restored=world.state.checkpoint_to_host()['arrays'][key]
    np.testing.assert_array_equal(restored,snapshot[key])
    # New worlds share the native reset bank and must checkpoint its actual
    # sampling mode, rather than silently recording the legacy default.
    another=GpuRollout(4,include_global=False,identity_encoding='known_only')
    assert another.independent_setup_seats and another.state_dict()['independent_setup_seats']


def test_reference_optimizer_ema_shuffle_and_history_resume_exact(tmp_path):
    from tests.test_same_seat_gpu import components,advance
    from junqi_rl.training.ema import PolicyEMA
    from junqi_rl.training.advantage_history import AdvantageHistory
    from junqi_rl.training.game_progress import capture_runtime,restore_runtime
    torch.manual_seed(101);torch.use_deterministic_algorithms(True)
    parts=components('compact_history',True,backbone='ataraxos')
    world,trainer,buf,progress,rng,cfg=parts
    trainer.cfg.train_chunk_size=2;trainer.cfg.ema_decay=.999
    trainer.ema=PolicyEMA(trainer.policy,.999)
    buf.advantage_history=AdvantageHistory(40)
    # advance's collection checks emit chronological rows; only the trainer
    # shuffles, and its RNG must restore exactly.
    buf.timestep_order='shuffle'
    advance(parts,0,reset=True)
    state=copy.deepcopy(dict(trainer=trainer.state_dict(),runtime=capture_runtime(
        world,progress,rng,cfg,0,rollout_buffer=buf)))
    expected=advance(parts,1)
    weights=copy.deepcopy(trainer.policy.state_dict());ema=copy.deepcopy(trainer.ema.state_dict())
    trainer.load_state_dict(state['trainer'])
    restore_runtime(state['runtime'],world,progress,rng,cfg,rollout_buffer=buf)
    actual=advance(parts,1)
    torch.testing.assert_close(expected[0],actual[0],atol=0,rtol=0)
    for k,v in weights.items():torch.testing.assert_close(v,trainer.policy.state_dict()[k],atol=0,rtol=0)
    for k,v in ema['model'].items():torch.testing.assert_close(v,trainer.ema.model.state_dict()[k],atol=0,rtol=0)
