"""Synchronous reference learner with globally weighted, uneven local batches.

Collection and runtime remain rank-local. Only scalar history, small setup
batches, and one gradient vector per logical optimizer step cross ranks.
"""
from collections import Counter
import torch
import torch.distributed as dist


def active():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size()>1


def objects(value):
    if not active():return [value]
    result=[None]*dist.get_world_size();dist.all_gather_object(result,value);return result


@torch.no_grad()
def broadcast_module(model):
    if active():
        for tensor in list(model.parameters())+list(model.buffers()):dist.broadcast(tensor,0)


def gather_rows(tensor):
    """Concatenate variable-length first dimensions, including empty ranks."""
    if not active():return tensor
    count=torch.tensor([len(tensor)],device=tensor.device,dtype=torch.long)
    counts=[torch.empty_like(count) for _ in range(dist.get_world_size())]
    dist.all_gather(counts,count);sizes=[int(x.item()) for x in counts]
    if max(sizes)==0:return tensor[:0]
    padded=tensor.new_zeros((max(sizes),)+tensor.shape[1:]);padded[:len(tensor)].copy_(tensor)
    gathered=[torch.empty_like(padded) for _ in sizes];dist.all_gather(gathered,padded)
    return torch.cat([x[:n] for x,n in zip(gathered,sizes)],0)


def global_quantile(values,q,minimum):
    values=gather_rows(values)
    answer=values.new_tensor(float(minimum))
    if len(values) and (not active() or dist.get_rank()==0):answer.copy_(values.quantile(q).clamp_min(minimum))
    if active():dist.broadcast(answer,0)
    return float(answer)


def global_count(n,device):
    result=torch.tensor(int(n),device=device,dtype=torch.long)
    if active():dist.all_reduce(result)
    return int(result)


def common_order(order,device):
    if active():
        tensor=torch.as_tensor(order,device=device,dtype=torch.long)
        dist.broadcast(tensor,0);order=tensor.cpu().numpy()
    return order


def fail_together(bad,device,message):
    flag=torch.tensor(int(bad),device=device,dtype=torch.int32)
    if active():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    if int(flag):raise FloatingPointError(message)


def sum_gradients(model):
    """Local losses already use the global denominator: SUM, never average."""
    if not active():return
    params=[p for p in model.parameters() if p.requires_grad]
    # A rank with no selected rows contributes zeros and still participates.
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    presence=flat.new_tensor([p.grad is not None for p in params])
    payload=torch.cat((flat,presence));dist.all_reduce(payload)
    flat=payload[:flat.numel()];presence=payload[flat.numel():].cpu().tolist()
    offset=0
    for i,p in enumerate(params):
        grad=flat[offset:offset+p.numel()].view_as(p)
        if not presence[i]:p.grad=None
        elif p.grad is None:p.grad=grad.clone()
        else:p.grad.copy_(grad)
        offset+=p.numel()


def reduce_metrics(metrics,keys,device,*,max_keys=()):
    if not active():return metrics
    vals=torch.stack([torch.as_tensor(metrics.get(k,0.),device=device).detach().float() for k in keys])
    dist.all_reduce(vals)
    metrics.update(zip(keys,vals.unbind()))
    for k in max_keys:
        val=torch.as_tensor(metrics.get(k,0.),device=device).detach().float();dist.all_reduce(val,op=dist.ReduceOp.MAX);metrics[k]=val
    return metrics


def setup_epoch(trainer,buffer,on_optimizer_step=None):
    """Shuffle every ready setup globally; no truncation or repeated padding."""
    from junqi_rl.arrangement.buffer import Batch
    ready=buffer.ready_flags
    mapping=dict(arrangements='arrangements',seat_idx='seat_idx',log_probs='log_probs',
                 returns='val_est',reg_returns='reg_val_est',advantages='adv_est')
    data={}
    for key,name in mapping.items():
        value=getattr(buffer,name)
        data[key]=gather_rows(value[ready])
    n=len(data['arrangements']);stats={k:[] for k in (
        'arr_train/'+name for name in ['policy_loss','value_loss','entropy_loss','kl_loss','total_loss',
        'entropy','clip_fraction','g_norm','lr','n_batches','grad_skip'])}
    rank,world=dist.get_rank(),dist.get_world_size()
    trainer._net_for_train.train()
    # cfg.batch_size is PER-RANK; the logical global batch is fixed by config.
    size=trainer.cfg.batch_size*world
    for _ in range(trainer.cfg.num_epoch_per_train):
        order=torch.randperm(n,device=buffer.device);dist.broadcast(order,0)
        for start in range(0,n,size):
            idx=order[start:start+size][rank::world]
            batch=Batch(**{k:v.index_select(0,idx) for k,v in data.items()})
            if trainer._step(batch,stats) and on_optimizer_step is not None:on_optimizer_step(trainer.net)
    if trainer.ema is not None:trainer.ema.update(trainer.net)
    out={k:sum(v)/len(v) for k,v in stats.items() if v and k!='arr_train/n_batches'}
    out['arr_train/n_batches']=sum(stats['arr_train/n_batches'])
    out['arr_train/grad_skip_total']=trainer._grad_nan_skip_count
    return out


def merge_summaries(summaries):
    """Global game accounting is a sum of distinct rank-local histories."""
    out=dict(summaries[0]);lineage=out['lineage_id']
    if any(s['lineage_id']!=lineage for s in summaries):raise ValueError('distributed lineage mismatch')
    for k in ('completed_games','tracked_environment_steps','total_completed_game_moves'):
        out[k]=sum(s[k] for s in summaries)
    for k in ('termination_reasons','winners','first_action_counts','first_source_counts'):
        c=Counter()
        for s in summaries:c.update(s[k])
        out[k]=dict(c)
    out['mean_completed_game_moves']=out['total_completed_game_moves']/max(1,out['completed_games'])
    out['world_size']=len(summaries)
    out['per_rank_completed_games']=[s['completed_games'] for s in summaries]
    return out
