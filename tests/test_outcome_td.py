"""Outcome TD must distinguish uncertainty from an actual drawn game."""
from types import SimpleNamespace

import pytest
import torch

from junqi_rl.training.seat_returns import same_seat_returns
from junqi_rl.training.seat_pending import PendingSeatTransitions
from junqi_rl.training.ppo import PPOTrainer
from junqi_rl.training.config import TrainConfig
from junqi_rl.training.game_progress import runtime_contract


def run(probs, seats, rewards=None, dones=None, td_lambda=.8, gae_lambda=.5):
    probs = torch.tensor(probs, dtype=torch.float32)[:, None, :]
    seats = torch.tensor(seats, dtype=torch.int8)[:, None]
    rewards = torch.zeros_like(seats, dtype=torch.float32) if rewards is None else torch.tensor(rewards, dtype=torch.float32)[:, None]
    dones = torch.zeros_like(seats, dtype=torch.bool) if dones is None else torch.tensor(dones, dtype=torch.bool)[:, None]
    return same_seat_returns(probs, rewards, dones, seats, torch.ones_like(dones),
                             gamma=1., gae_lambda=gae_lambda, td_lambda=td_lambda)


def test_zero_mean_uncertainty_does_not_turn_into_draw():
    adv, targets, ready, _ = run([[0, 1, 0], [.5, 0, .5]], [0, 0])
    assert adv[0].item() == 0
    assert ready[:, 0].tolist() == [True, False]
    torch.testing.assert_close(targets[0, 0], torch.tensor([.5, 0, .5]))


@pytest.mark.parametrize('reward', [-1., 0., 1.])
def test_terminal_outcome_closes_eliminated_seat_with_one_hot(reward):
    seats = [0, 1, 2, 3, 1, 2, 3, 2]
    _, targets, ready, pending = run([[.2, .3, .5]] * len(seats), seats,
        rewards=[0]*7+[reward], dones=[False]*7+[True], td_lambda=1.)
    bins = torch.tensor([int(reward if s%2 == 0 else -reward)+1 for s in seats])
    expected = torch.nn.functional.one_hot(bins, 3).float()
    torch.testing.assert_close(targets[:, 0], expected)
    assert ready.all() and (pending == -1).all()


@pytest.mark.parametrize('lam', [0., .8, 1.])
def test_lambda_mixture_matches_independent_n_step_targets(lam):
    # Seat 0 decisions at 0/2/5, seat 2 terminal. Intervening observers
    # must never become its bootstrap, including its own teammate.
    p = [[.1,.6,.3], [.8,.1,.1], [.2,.1,.7], [.9,0,.1], [0,.9,.1], [.3,.4,.3], [.6,.1,.3]]
    seats = [0,1,0,2,3,0,2]
    _, targets, _, _ = run(p,seats,rewards=[0]*6+[1],dones=[False]*6+[True],td_lambda=lam)
    expected = ((1-lam)*torch.tensor(p[2]) + lam*(1-lam)*torch.tensor(p[5])
                + lam**2*torch.tensor([0.,0.,1.]))
    torch.testing.assert_close(targets[0,0], expected)


def test_randomized_distribution_mean_matches_scalar_td_and_gae():
    gen = torch.Generator().manual_seed(72)
    p = torch.rand(97,7,3,generator=gen); p /= p.sum(-1,keepdim=True)
    seats = torch.randint(4,(97,7),generator=gen)
    dones = torch.rand(97,7,generator=gen)<.08
    rewards = (torch.randint(3,(97,7),generator=gen)-1).float()*dones
    valid = torch.ones_like(dones);valid[:4]=torch.rand(4,7,generator=gen)>.5
    dones &= valid;rewards *= dones
    args = (rewards,dones,seats,valid)
    kwargs = dict(gamma=1.,gae_lambda=.5,td_lambda=.8)
    ca,cr,cv,cp = same_seat_returns(p,*args,**kwargs)
    sa,sr,sv,sp = same_seat_returns(p[...,2]-p[...,0],*args,**kwargs)
    torch.testing.assert_close(ca,sa)
    torch.testing.assert_close(cr[...,2]-cr[...,0],sr,atol=2e-7,rtol=1e-5)
    assert torch.equal(cv,sv) and torch.equal(cp,sp)
    assert (cr >= 0).all()
    torch.testing.assert_close(cr.sum(-1),torch.ones_like(sr))


def test_cross_entropy_uses_complete_target_and_rejects_scalar_same_seat():
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.cfg = SimpleNamespace(net=SimpleNamespace(use_cat_vf=True),return_perspective='same_seat')
    logits = torch.tensor([[0.,-8.,0.]],requires_grad=True)
    target = torch.tensor([[.5,0,.5]])
    loss = trainer._value_loss(logits.log_softmax(-1),target)
    assert loss.item() < .7
    loss.backward()
    torch.testing.assert_close(logits.grad,logits.detach().softmax(-1)-target)
    draw_loss = trainer._value_loss(logits.log_softmax(-1),torch.tensor([[0.,1.,0.]]))
    assert draw_loss.item() > 8.
    with pytest.raises(ValueError,match='full outcome'):
        trainer._value_loss(logits.log_softmax(-1),torch.zeros(1))


def test_old_pending_and_runtime_contract_cannot_resume_as_outcome_td():
    scalar = PendingSeatTransitions(1,'cpu')
    cat = PendingSeatTransitions(1,'cpu',outcome_td=True)
    with pytest.raises(ValueError,match='schema'):
        cat.load_state_dict(scalar.state_dict())
    cat.load_state_dict(cat.state_dict())
    cfg = TrainConfig();cfg.ppo.return_perspective='same_seat';cfg.net.use_cat_vf=True
    assert runtime_contract(cfg)['return_learning']['td_target'] == 'outcome_distribution_v1'


def test_corrupt_outcome_probabilities_are_rejected():
    with pytest.raises(ValueError,match='normalized'):
        run([[.5,.5,.5]],[0])
    with pytest.raises(ValueError,match='terminal rewards'):
        run([[.2,.3,.5]],[0],rewards=[.2],dones=[True])
