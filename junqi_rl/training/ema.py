"""Once-per-training-iteration EMA, kept separate from the self-play actor."""
import copy
import torch


class PolicyEMA:
    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        for shadow, parameter in zip(self.model.parameters(), model.parameters(), strict=True):
            shadow.mul_(self.decay).add_(parameter, alpha=1-self.decay)
        for shadow, value in zip(self.model.buffers(), model.buffers(), strict=True):
            shadow.copy_(value)
        self.updates += 1

    def state_dict(self):
        return dict(model=self.model.state_dict(), decay=self.decay, updates=self.updates)

    def load_state_dict(self, state):
        if state['decay'] != self.decay:
            raise ValueError("EMA decay mismatch")
        self.model.load_state_dict(state['model'])
        self.updates = int(state['updates'])
