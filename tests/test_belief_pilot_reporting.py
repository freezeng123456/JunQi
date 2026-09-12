import copy
import torch

from scripts.pilot_joint_belief import max_parameter_delta


def test_learning_delta_excludes_batch_normalization_counters():
    model = torch.nn.BatchNorm1d(3)
    initial = copy.deepcopy(model.state_dict())
    model.num_batches_tracked.add_(192)
    assert max_parameter_delta(model, initial) == 0
    with torch.no_grad():
        model.weight.add_(.125)
    assert max_parameter_delta(model, initial) == .125
