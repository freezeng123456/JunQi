"""Memory-bounded updates with unchanged logical minibatches and Adam cadence.

Every chunk contributes its sample-weighted mean, then gradients are clipped
once and Adam steps once. A chunk is never an extra optimization iteration.
"""
from . import distributed_reference as distributed
import torch
from torch.nn import functional as F


def _finish(model, optimizer, limit):
    distributed.sum_gradients(model)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), limit)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite reference-update gradient")
    optimizer.step()
    return norm.detach()


def update_move_row(trainer, batch):
    cfg = trainer.cfg
    if trainer._scaler is not None or batch.value_obs_spatial is not None:
        raise ValueError("Reference chunks support FP32/BF16 filtered self-play batches")
    if batch.value_only_mask is not None and batch.value_only_mask.any():
        raise ValueError("Reference chunks do not support random-opponent value rows")
    n = len(batch.actions)
    global_n = distributed.global_count(n, trainer.device)
    if global_n == 0:
        raise ValueError("Cannot update a globally empty row")
    trainer._policy_for_train.eval()
    trainer.optimizer.zero_grad(set_to_none=True)
    dtype = trainer._amp_dtype if trainer._use_amp else torch.float32
    temp = trainer._get_temperature()
    metrics = {}
    nonfinite = False
    for start in range(0, n, cfg.train_chunk_size):
        end = min(n, start + cfg.train_chunk_size)
        weight = (end-start)/global_n
        sl = slice(start, end)
        obs, glob = batch.obs_spatial[sl].to(dtype), batch.obs_global[sl].to(dtype)
        legal, action = batch.legal_mask[sl], batch.actions[sl].long()
        trainer._assert_policy_active_actions_legal(legal, action, torch.ones_like(action, dtype=torch.bool))
        with torch.autocast(trainer.device.type, dtype=trainer._amp_dtype, enabled=trainer._use_amp):
            if batch.behavior_log_probs is not None:
                old = batch.behavior_log_probs[sl].detach()
            else:
                with torch.no_grad():
                    old = trainer._collect_policy(obs, glob, legal, actions=action)['log_probs']
            out = trainer._policy_for_train(obs, glob, legal, actions=action)
            logp = out['log_probs']
            chosen = logp.gather(-1, action[:, None]).squeeze(-1)
            pl = trainer._policy_loss(chosen, batch.old_log_probs[sl], batch.advantages[sl])
            vl = trainer._value_loss(out['value'], batch.returns[sl])
            ml, ent = trainer._entropy_loss(logp, legal)
            kl = trainer._reverse_kl(logp, old, legal)
            loss = cfg.policy_coef*pl + cfg.vf_coef*vl + temp*ml + cfg.kl_coef*kl
        if not torch.isfinite(loss):
            nonfinite = True
        else:
            (loss * weight).backward()
        ratio = chosen.detach() - batch.old_log_probs[sl]
        vals = dict(policy_loss=pl, value_loss=vl, entropy_loss=ml, entropy=ent,
                    kl_loss=kl, total_loss=loss, approx_kl=-ratio.mean())
        for k, v in vals.items():
            key = 'train/'+k
            metrics[key] = metrics.get(key, 0.) + v.detach()*weight
        peak = ratio.abs().max()
        metrics['train/kl_log_ratio_abs_max'] = torch.maximum(
            metrics.get('train/kl_log_ratio_abs_max', peak), peak)
    distributed.fail_together(nonfinite, trainer.device, "Nonfinite reference move loss")
    distributed.reduce_metrics(metrics, ['train/'+k for k in (
        'policy_loss','value_loss','entropy_loss','entropy','kl_loss','total_loss','approx_kl')],
        trainer.device, max_keys=('train/kl_log_ratio_abs_max',))
    metrics['train/g_norm'] = _finish(trainer.policy, trainer.optimizer, cfg.max_grad_norm)
    trainer.num_train_step += 1
    metrics.update({'train/lr':trainer.optimizer.param_groups[0]['lr'],
                    'train/temperature':temp, 'train/batch_size':global_n})
    return metrics


def update_setup_batch(trainer, batch, stats):
    cfg, n = trainer.cfg, len(batch.arrangements)
    device = next(trainer.net.parameters()).device
    global_n = distributed.global_count(n, device)
    if global_n == 0:
        return False
    nonfinite = False
    dtype = {'float32':torch.float32, 'bfloat16':torch.bfloat16}[cfg.autocast_dtype]
    trainer.optim.zero_grad(set_to_none=True)
    totals = {}
    for start in range(0, n, cfg.train_chunk_size):
        end = min(n, start + cfg.train_chunk_size);sl = slice(start,end);weight=(end-start)/global_n
        arrangements = batch.arrangements[sl]
        with torch.autocast(arrangements.device.type, dtype=dtype, enabled=dtype!=torch.float32):
            out = trainer._net_for_train(arrangements, batch.seat_idx[sl])
            logp = F.log_softmax(out['logits'], dim=-1)
            chosen = arrangements.argmax(-1, keepdim=True)
            new = logp.gather(-1, chosen).squeeze(-1)
            old = batch.log_probs[sl].gather(-1, chosen).squeeze(-1)
            ratio = (new-old).exp();adv = batch.advantages[sl]
            pl = -torch.minimum(adv*ratio, adv*ratio.clamp(1-cfg.clip_range,1+cfg.clip_range)).mean()
            kl = (logp.exp()*(logp-batch.log_probs[sl])).sum(-1).mean()
            vl = -(batch.returns[sl]*F.log_softmax(out['value'],dim=-1)).sum(-1).mean()
            el = F.mse_loss(out['ent_pred'].squeeze(-1),batch.reg_returns[sl])
            loss = cfg.policy_coef*pl+cfg.vf_coef*vl+cfg.ent_pred_coef*el+cfg.kl_coef*kl
        if not torch.isfinite(loss):
            nonfinite = True
        else:
            (loss*weight).backward()
        vals = dict(policy_loss=pl,value_loss=vl,entropy_loss=el,kl_loss=kl,total_loss=loss,
                    entropy=-new.sum(-1).mean(),clip_fraction=((ratio-1).abs()>cfg.clip_range).float().mean())
        for k,v in vals.items():totals[k]=totals.get(k,0.)+v.detach()*weight
    distributed.fail_together(nonfinite, device, "Nonfinite reference setup loss")
    distributed.reduce_metrics(totals, ('policy_loss','value_loss','entropy_loss',
        'kl_loss','total_loss','entropy','clip_fraction'), device)
    norm = _finish(trainer.net,trainer.optim,cfg.max_grad_norm)
    for k,v in totals.items():stats['arr_train/'+k].append(float(v))
    stats['arr_train/g_norm'].append(float(norm))
    stats['arr_train/lr'].append(trainer.optim.param_groups[0]['lr'])
    stats['arr_train/n_batches'].append(1.)
    stats['arr_train/grad_skip'].append(0.)
    trainer._total_batches += 1
    return True
