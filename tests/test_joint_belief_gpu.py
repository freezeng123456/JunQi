"""Target-device causal decoding, full-army constraints, and gradient checks."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')


def test_v100_joint_training_and_cached_sampling():
    from tests.test_joint_belief import model, public_fixture
    from junqi_rl.belief.legacy_joint import build_joint_input,joint_training_targets
    import numpy as np
    soa,rules=public_fixture(); ids=np.array([0])
    inputs=build_joint_input(soa,rules,ids,ids).to('cuda')
    labels=joint_training_targets(soa,ids,inputs.to('cpu')).cuda()
    net=model().cuda().train(); obs=torch.randn(1,317,17,17,device='cuda'); seats=torch.tensor([0],device='cuda')
    loss=-net.teacher_logits(obs,seats,targets=labels,**inputs.kwargs()).log_softmax(-1).gather(-1,labels[...,None]).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(net.type_embedding.weight.grad).all()
    assert net.type_embedding.weight.grad.norm()>0
    net.eval(); samples=net.sample(obs,seats,n_samples=8,**inputs.kwargs())
    assert inputs.support[:,None].expand(1,8,75,12).gather(-1,samples[...,None]).all()
    for army in range(3):
        got=torch.nn.functional.one_hot(samples[:,:,army*25:(army+1)*25],12).sum(-2)
        assert (got==inputs.inventory[:,army+1,None]).all()
