"""Rolling advantage quantile, recomputed from behavior values and new outcomes.

Only scalar rollout data is retained; old boards are never replayed for SGD.
The parallel affine scan avoids thousands of Python-launched trace kernels.
"""
import torch


@torch.no_grad()
def history_advantages(values, rewards, dones, seats, valid, lam):
    value = values[..., 2]-values[..., 0] if values.ndim == 3 else values
    t, n = value.shape
    time = torch.arange(t, device=value.device)[:, None].expand(t, n)
    env = torch.arange(n, device=value.device)[None, :]
    padded_values = torch.cat([value, value.new_zeros(1,n)])
    result = torch.zeros_like(value)
    for seat in range(4):
        own = valid & (seats == seat)
        event = own | dones
        next_idx = torch.where(event, time, t).flip(0).cummin(0).values.flip(0)
        next_idx = torch.cat([next_idx[1:], torch.full_like(next_idx[:1],t)])
        padded_own = torch.cat([own, torch.zeros_like(own[:1])])
        own_successor = padded_own[next_idx,env]
        outcome = rewards * torch.where((seats & 1)==(seat & 1),1.,-1.)
        padded_reward = torch.cat([outcome, outcome.new_zeros(1,n)])
        target = torch.where(own_successor,padded_values[next_idx,env],padded_reward[next_idx,env])
        target = torch.where(dones,outcome,target)
        ready = own & ((next_idx<t) | dones)
        a = torch.where(ready,target-value,0.)
        b = torch.where(dones,0.,torch.where(own,lam,1.)).to(value.dtype)
        stride=1
        while stride<t:
            a[:-stride] = a[:-stride]+b[:-stride]*a[stride:]
            b[:-stride] = b[:-stride]*b[stride:]
            stride *= 2
        result = torch.where(own,a,result)
    return result


class AdvantageHistory:
    FIELDS = ('values','rewards','dones','seats','valid')

    def __init__(self, steps):
        self.steps, self.data = int(steps), {}

    def append(self, **data):
        for key in self.FIELDS:
            current = data[key].detach()
            self.data[key] = (torch.cat([self.data[key],current],0) if key in self.data else current.clone())[-self.steps:]

    def threshold(self, lam, keep_rate, minimum):
        adv = history_advantages(**self.data,lam=lam)
        chosen = adv[self.data['valid']].abs().float()
        from .distributed_reference import global_quantile
        return global_quantile(chosen, 1-keep_rate, minimum)

    def state_dict(self):
        return dict(steps=self.steps,data={k:v.cpu().clone() for k,v in self.data.items()})

    def load_state_dict(self,state,device):
        if state['steps']!=self.steps:
            raise ValueError('advantage history length mismatch')
        self.data = {k:v.to(device).clone() for k,v in state['data'].items()}
