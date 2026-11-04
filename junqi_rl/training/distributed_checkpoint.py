"""Commit shared learner weights only after all rank runtime shards are durable."""
from pathlib import Path
import hashlib
import json
import uuid
import torch
import torch.distributed as dist
from ._checkpoint_io import atomic_write
from .distributed_reference import objects


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_runtime(checkpoint_path, provider):
    rank, world = dist.get_rank(), dist.get_world_size()
    # A retry must not overwrite shards referenced by the last committed manifest.
    revision = objects(uuid.uuid4().hex if rank == 0 else None)[0]
    path = Path(checkpoint_path).with_suffix(f'.rank{rank:03d}.{revision}.runtime.pt')
    result = None
    try:
        runtime = provider()
        atomic_write(str(path), lambda f: torch.save(runtime, f))
        result = dict(rank=rank, file=path.name, sha256=digest(path),
                      completed_games=runtime['games']['completed_games'])
    except Exception as exc:
        result = dict(rank=rank, error=f'{type(exc).__name__}: {exc}')
    shards = objects(result)
    if any('error' in s for s in shards):
        raise RuntimeError(f'Distributed runtime checkpoint failed: {shards}')
    return dict(version=2, world_size=world, shards=shards,
                games=dict(completed_games=sum(s['completed_games'] for s in shards)))


def load_runtime(checkpoint_path, state):
    world, rank = dist.get_world_size(), dist.get_rank()
    if state.get('version') != 2 or state.get('world_size') != world:
        raise ValueError('Exact distributed resume requires the same world size and rank runtime shards')
    if len(state['shards']) != world or [s['rank'] for s in state['shards']] != list(range(world)):
        raise ValueError('Invalid rank runtime manifest')
    item = state['shards'][rank]
    if Path(item['file']).name != item['file']:
        raise ValueError('Runtime shard must be adjacent to checkpoint')
    path = Path(checkpoint_path).parent / item['file']
    if digest(path) != item['sha256']:
        raise ValueError(f'Runtime shard checksum mismatch: rank {rank}')
    return torch.load(path, map_location='cpu', weights_only=False)


def remove_shards(checkpoint_path):
    """Only remove shards owned by this numbered checkpoint after its pruning."""
    path = Path(checkpoint_path)
    for shard in path.parent.glob(path.stem + '.rank[0-9][0-9][0-9].*.runtime.pt'):
        shard.unlink()


def trim_journals(save_dir, rollout):
    """Discard unsaved tail after a restart, before replaying those rollouts."""
    for name, key in (('training_games.jsonl','ending_rollout'), ('game_progress.jsonl','rollout'),
                      ('arrangement_progress.jsonl','rollout'), ('checkpoint_games.jsonl','rollout')):
        path = Path(save_dir) / name
        if not path.exists():
            continue
        temp = path.with_suffix('.jsonl.recovering')
        with path.open() as source, temp.open('w') as out:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break  # an interrupted final append
                if row[key] <= rollout:
                    out.write(line)
        temp.replace(path)


def verify_learners(trainer, arrangement=None):
    """Require identical parameters, EMA and update clocks on every rank."""
    modules = {'policy': trainer.policy}
    if trainer.ema is not None:
        modules['policy_ema'] = trainer.ema.model
    counters = dict(move_updates=trainer.num_train_step, rollout=trainer.num_rollout)
    if arrangement is not None:
        modules['setup'] = arrangement.net
        if arrangement.ema is not None:
            modules['setup_ema'] = arrangement.ema.model
        counters['setup_updates'] = arrangement._total_batches
    hashes = {}
    for label, module in modules.items():
        h = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hashes[label] = h.hexdigest()
    receipt = dict(hashes=hashes, counters=counters)
    all_receipts = objects(receipt)
    if any(item != receipt for item in all_receipts):
        raise RuntimeError('Learner parameters or update counters diverged across ranks')
    return dict(receipt, world_size=dist.get_world_size(), identical=True)
