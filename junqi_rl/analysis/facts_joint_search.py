"""Fresh neural hidden-army proposals and full-action facts-only search.

Worlds are rebuilt from the observer's known opening identities, sampled
identities, and public history. Full replay and root-information checks further
condition proposals beyond the domain/inventory/combat constraint compiler.
Exhausting that rejection budget is an error, never a blueprint fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from junqi_core.identity import TRACKED_TYPES
from junqi_core.knowledge import KnowledgeSnapshot
from junqi_core.observation import ObservationBuilder
from junqi_core.rules import ALL_SEATS, CAMP_INDICES, PieceType, Seat, ShowMode
from junqi_core.state import Action, GameState, MoveResult
from junqi_rl.belief.constraints import ConstraintBudgetExceeded, compile_constraints
from junqi_rl.belief.joint import build_joint_input
from junqi_rl.env import (JunqiEnv, action_id_to_src_dst, rotate_to_compact_action_id,
                          src_dst_to_action_id, unrotate_compact_action_id)
from junqi_rl.input_contract import policy_input_contract


@dataclass(frozen=True)
class SearchConfig:
    particles: int = 1
    depth: int = 1
    proposal_batches: int = 2
    beta: float = 0.5
    max_constraint_nodes: int = 100_000

    def __post_init__(self):
        for name in ('particles', 'depth', 'proposal_batches', 'max_constraint_nodes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError('Search beta must be finite and positive')


class SamplingBudgetExceeded(RuntimeError):
    """Not enough history-consistent neural samples; no action was selected."""


class WorldReconstructionError(ValueError):
    def __init__(self, reason, replayed_steps=0):
        super().__init__(reason)
        self.reason = reason
        self.replayed_steps = replayed_steps


def public_result(result: MoveResult) -> tuple:
    """Complete broadcast, excluding all debug/private type fields."""
    return (int(result.seat), tuple(result.src), tuple(result.dst), int(result.event),
            bool(result.flag_reveal_src), bool(result.flag_reveal_dst),
            bool(result.flag_captured), bool(result.terminated_after),
            result.winner_team_after, bool(result.draw_after),
            tuple(int(s) for s in result.seats_died_this_step))


def _broadcast(row):
    if len(row) != 11:
        raise ValueError('Public replay requires all 11 broadcast fields')
    return (int(row[0]), tuple(row[1]), tuple(row[2]), int(row[3]),
            bool(row[4]), bool(row[5]), bool(row[6]), bool(row[7]),
            None if row[8] is None else int(row[8]), bool(row[9]), tuple(row[10]))


def step_world(world: JunqiEnv, action_id: int) -> MoveResult:
    """Step rules directly; recording environments retain their fact ledger."""
    if world._knowledge is not None:
        return world._step_game_only(action_id)[2].move_result
    src, dst = action_id_to_src_dst(action_id)
    return world.state.step_inplace(Action(world.state.turn, src, dst))


def sampled_setups(snapshot: KnowledgeSnapshot, piece_ids, assignment):
    """Build all four armies without access to a real simulator state."""
    if snapshot.show_mode is not ShowMode.DARK:
        raise ValueError('Facts joint worlds require DARK mode')
    observer = int(snapshot.observer)
    expected = {owner * 30 + slot for owner in range(4) if owner != observer
                for slot in range(30) if slot not in CAMP_INDICES}
    ids, labels = np.asarray(piece_ids), np.asarray(assignment)
    if (ids.shape != (75,) or ids.dtype.kind not in 'iu'
            or len(set(ids.tolist())) != 75 or set(ids.tolist()) != expected):
        raise ValueError('World construction requires exactly all 75 other-army identities')
    if (labels.shape != (75,) or labels.dtype.kind not in 'iu'
            or (labels < 0).any() or (labels >= len(TRACKED_TYPES)).any()):
        raise ValueError('World identities must be 75 integer type labels')
    setup = [[PieceType.NONE] * 30 for _ in ALL_SEATS]
    for slot in range(30):
        if slot in CAMP_INDICES:
            continue
        pid = observer * 30 + slot
        label = int(snapshot.known_types[pid])
        if label < 0 or snapshot.initial_cells[pid] < 0:
            raise ValueError('Observer original identities must remain known, including dead pieces')
        setup[observer][slot] = TRACKED_TYPES[label]
    for pid, label in zip(ids, labels):
        owner, slot = divmod(int(pid), 30)
        setup[owner][slot] = TRACKED_TYPES[int(label)]
    return tuple(tuple(row) for row in setup)


def materialize(snapshot, piece_ids, assignment, history, *, max_num_moves=4000):
    """Strict reference reconstruction from observer facts and proposed labels."""
    if len(history) != snapshot.step:
        raise ValueError('Public history length differs from observer snapshot')
    setup = sampled_setups(snapshot, piece_ids, assignment)
    world = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                     identity_encoding='known_only', max_num_moves=max_num_moves)
    world._state = GameState.new_game(setup, show_mode=ShowMode.DARK)
    world._beliefs = {seat: None for seat in ALL_SEATS}
    for index, row in enumerate(history):
        expected = _broadcast(row)
        if world.state.terminated or int(world.state.turn) != expected[0]:
            raise WorldReconstructionError('public_turn_or_termination_mismatch', index)
        try:
            result = step_world(world, src_dst_to_action_id(expected[1], expected[2]))
        except (ValueError, RuntimeError) as exc:
            raise WorldReconstructionError('public_action_illegal', index) from exc
        if public_result(result) != expected:
            raise WorldReconstructionError('public_broadcast_mismatch', index + 1)
    return world


def clone_world(world: JunqiEnv) -> JunqiEnv:
    """Clone a reconstructed sampled world; never call this on the real root."""
    clone = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                     identity_encoding='known_only', max_num_moves=world.max_num_moves)
    clone._state = world.state.clone()
    clone._beliefs = {seat: None for seat in ALL_SEATS}
    return clone


def scalar_value(value):
    """Training's categorical bins are [-1, 0, +1], i.e. lose/draw/win."""
    if value.ndim == 2 and value.shape[-1] == 3:
        probabilities = value.float().exp()
        answer = probabilities[..., 2] - probabilities[..., 0]
    elif value.ndim == 1 or (value.ndim == 2 and value.shape[-1] == 1):
        answer = value.float().reshape(-1)
    else:
        raise ValueError('Value head must be scalar or [-1, 0, +1] log-probabilities')
    if not torch.isfinite(answer).all():
        raise FloatingPointError('Nonfinite blueprint value')
    return answer


def regularized_update(log_policy, q, legal, *, beta):
    """KL-regularized policy improvement; no hand-designed action magnet."""
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('Search beta must be finite and positive')
    if (log_policy.shape != q.shape or legal.shape != q.shape or not legal.any(-1).all()
            or not torch.isfinite(log_policy[legal]).all() or not torch.isfinite(q[legal]).all()):
        raise ValueError('Every legal action requires finite prior probability and searched Q')
    return (log_policy + q / beta).masked_fill(~legal, -torch.inf).log_softmax(-1)


class FactsBlueprint:
    """Batched 317/0 inference with no identity probability objects or updates."""
    def __init__(self, policy, device):
        contract = policy_input_contract(policy)
        if contract.identity_encoding != 'known_only' or contract.include_global:
            raise ValueError('Facts search requires a known_only 317/0 blueprint')
        self.policy = policy.eval()
        self.device = torch.device(device)
        self.builder = ObservationBuilder(**contract.observation_kwargs())
        self.capacity = 0
        self.query_counts = dict(policy_calls=0, policy_positions=0, value_calls=0, value_positions=0)

    def inputs(self, worlds, *, include_legal=True):
        n = len(worlds)
        if not n:
            raise ValueError('A blueprint batch cannot be empty')
        if n > self.capacity:
            self.capacity = n
            self.spatial = np.empty((n, 317, 17, 17), dtype=np.float32)
            self.global_ = np.empty((n, 0), dtype=np.float32)
            self.legal = np.zeros((n, 129 * 129), dtype=bool)
        seats = [world.state.turn for world in worlds]
        self.builder.build_observations_batch(
            [world.state for world in worlds], [None] * n, seats,
            self.spatial[:n], self.global_[:n])
        if include_legal:
            self.legal[:n].fill(False)
            for i, (world, seat) in enumerate(zip(worlds, seats)):
                for aid in world.legal_action_ids():
                    self.legal[i, rotate_to_compact_action_id(int(aid), seat)] = True
        return (torch.from_numpy(self.spatial[:n]).to(self.device),
                torch.from_numpy(self.global_[:n]).to(self.device),
                torch.from_numpy(self.legal[:n]).to(self.device))

    @torch.no_grad()
    def distribution(self, worlds):
        obs, global_, legal = self.inputs(worlds)
        if not legal.any(-1).all():
            raise ValueError('Active blueprint world has no legal action')
        self.query_counts['policy_calls'] += 1
        self.query_counts['policy_positions'] += len(worlds)
        output = self.policy(obs, global_, legal, actions=legal.long().argmax(-1))
        log_probs = output['log_probs'].float()
        if log_probs.shape != legal.shape or not torch.isfinite(log_probs[legal]).all():
            raise ValueError('Nonfinite or malformed blueprint action distribution')
        log_probs = log_probs.masked_fill(~legal, -torch.inf).log_softmax(-1)
        return log_probs, scalar_value(output['value']), legal, obs

    @torch.no_grad()
    def values(self, worlds):
        obs, global_, _ = self.inputs(worlds, include_legal=False)
        self.query_counts['value_calls'] += 1
        self.query_counts['value_positions'] += len(worlds)
        return scalar_value(self.policy.forward_value(obs, global_))


def _finished_value(world, root_seat):
    if world.state.terminated:
        return float((world.state.team_rewards() or (0, 0, 0, 0))[int(root_seat)])
    if world.state.move_counter >= world.max_num_moves:
        return 0.0
    return None


class FactsJointSearch:
    def __init__(self, blueprint: FactsBlueprint, joint, config=SearchConfig(), *, seed=0,
                 capture_diagnostics=False, inventory_backend='scan', overlap_constraints=False):
        if (getattr(joint, 'identity_encoding', None) != 'known_only'
                or getattr(joint, 'uses_global_summary', None) is not False):
            raise ValueError('Search requires a facts-only neural belief model')
        self.blueprint, self.joint, self.config = blueprint, joint.eval(), config
        if inventory_backend not in ('scan', 'incremental'):
            raise ValueError('Unsupported constraint inventory backend')
        if overlap_constraints and getattr(joint, 'belief_kind', None) != 'joint':
            raise ValueError('Constraint overlap is supported only by the joint sampler')
        self.inventory_backend, self.overlap_constraints = inventory_backend, bool(overlap_constraints)
        self.generator = torch.Generator(device=blueprint.device).manual_seed(seed)
        self.metrics = dict(decisions=0, successful_decisions=0, joint_refreshes=0,
                            proposals=0, accepted_proposals=0, rejected_proposals=0,
                            sample_budget_failures=0, public_replay_steps=0,
                            constraint_budget_failures=0,
                            rollout_world_steps=0, legal_actions=0, candidate_actions=0,
                            unique_accepted_assignments=0, constraint_seconds=0.0,
                            sampling_seconds=0.0, materialization_seconds=0.0,
                            simulation_seconds=0.0)
        self.query_counts = dict.fromkeys(blueprint.query_counts, 0)
        self.constraint_counts = {}
        self.matching_cache_counts = {}
        self.constraint_implementation = None
        self.rejection_reasons = {}
        self.latencies = []
        self.last_q = None
        # Opt-in development telemetry. Match callers keep the original memory
        # and transfer costs, and telemetry never supplies inference inputs.
        self.capture_diagnostics = bool(capture_diagnostics)
        self.last_decision = None
        self.last_particle_diagnostics = None

    @torch.no_grad()
    def particles(self, env, history, obs, root_legal):
        observer = env.state.turn
        snapshot = env.knowledge_snapshot(observer)
        if snapshot.step != len(history) or env.state.move_counter != len(history):
            raise ValueError('Facts search requires the complete current public history')
        inputs = build_joint_input([snapshot])
        solver_kwargs = {'max_search_nodes': self.config.max_constraint_nodes}
        if self.inventory_backend != 'scan':
            solver_kwargs['inventory_backend'] = self.inventory_backend
        solver = compile_constraints(inputs, **solver_kwargs)
        self.constraint_implementation = getattr(solver, 'implementation', None)
        root_spatial = obs[0].detach().cpu().numpy().copy()
        root_legal = root_legal[0].detach().cpu().numpy().copy()
        worlds, identities = [], set()
        accepted = [] if self.capture_diagnostics else None
        self.metrics['joint_refreshes'] += 1
        previous_solver_stats = dict(solver.stats)
        previous_matching_stats = dict(getattr(solver, 'matching_cache_stats', {}))
        for _ in range(self.config.proposal_batches):
            count = self.config.particles - len(worlds)
            started = time.perf_counter()
            try:
                sample_kwargs = {'overlap_constraints': True} if self.overlap_constraints else {}
                samples = self.joint.sample(obs, inputs=inputs, n_samples=count,
                                            generator=self.generator, constraints=[solver], **sample_kwargs)
            except ConstraintBudgetExceeded:
                self.metrics['constraint_budget_failures'] += 1
                raise
            finally:
                self.metrics['sampling_seconds'] += time.perf_counter() - started
                for key, value in solver.stats.items():
                    self.constraint_counts[key] = self.constraint_counts.get(key, 0) + value - previous_solver_stats.get(key, 0)
                previous_solver_stats = dict(solver.stats)
                for key, value in getattr(solver, 'matching_cache_stats', {}).items():
                    self.matching_cache_counts[key] = self.matching_cache_counts.get(key, 0) + value - previous_matching_stats.get(key, 0)
                previous_matching_stats = dict(getattr(solver, 'matching_cache_stats', {}))
            self.metrics['constraint_seconds'] += float(
                getattr(self.joint, 'last_sample_metrics', {}).get('constraint_seconds', 0.0))
            if tuple(samples.shape) != (1, count, 75):
                raise ValueError('Joint sampler must return every one of the 75 identities')
            labels = samples[0].detach().cpu().numpy()
            self.metrics['proposals'] += len(labels)
            for assignment in labels:
                # A neural sampler violating its own declared support is a bug,
                # not an ordinary full-history rejection to silently absorb.
                solver.validate(assignment)
                started = time.perf_counter()
                reason = None
                try:
                    world = materialize(snapshot, inputs.piece_ids[0], assignment, history,
                                        max_num_moves=env.max_num_moves)
                    self.metrics['public_replay_steps'] += world.state.move_counter
                    observation = self.blueprint.builder.build(world.state, None, observer)
                    if not np.array_equal(observation.spatial, root_spatial):
                        reason = 'root_observation_mismatch'
                    else:
                        legal = np.zeros(129 * 129, dtype=bool)
                        for aid in world.legal_action_ids():
                            legal[rotate_to_compact_action_id(int(aid), observer)] = True
                        if not np.array_equal(legal, root_legal):
                            reason = 'root_legal_actions_mismatch'
                except WorldReconstructionError as exc:
                    self.metrics['public_replay_steps'] += exc.replayed_steps
                    reason = exc.reason
                finally:
                    self.metrics['materialization_seconds'] += time.perf_counter() - started
                if reason is not None:
                    self.metrics['rejected_proposals'] += 1
                    self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
                    continue
                worlds.append(world)
                if accepted is not None:
                    accepted.append(assignment.tolist())
                identity = tuple(int(x) for x in assignment)
                if identity not in identities:
                    self.metrics['unique_accepted_assignments'] += 1
                    identities.add(identity)
                self.metrics['accepted_proposals'] += 1
            if len(worlds) == self.config.particles:
                if accepted is not None:
                    self.last_particle_diagnostics = {
                        'piece_ids': inputs.piece_ids[0].tolist(),
                        'assignments': accepted,
                        'support_validated': True,
                        'public_replay_validated': True,
                        'root_observation_validated': True,
                        'root_legal_actions_validated': True,
                        'constraint_stats': dict(solver.stats),
                        'matching_cache_stats': dict(getattr(solver, 'matching_cache_stats', {})),
                        'constraint_implementation': self.constraint_implementation,
                        'inventory_backend': self.inventory_backend,
                    }
                return worlds
        self.metrics['sample_budget_failures'] += 1
        raise SamplingBudgetExceeded(
            f'Only {len(worlds)}/{self.config.particles} complete history-consistent neural '
            f'worlds after {self.config.proposal_batches} proposal batches; no action selected')

    @torch.no_grad()
    def action(self, env, history):
        began = time.perf_counter()
        if self.capture_diagnostics:
            self.last_decision = self.last_particle_diagnostics = None
        queries_before = dict(self.blueprint.query_counts)
        self.metrics['decisions'] += 1
        if env.state.terminated or env.state.move_counter >= env.max_num_moves:
            raise ValueError('Cannot search a terminated or externally truncated game')
        if env.builder.identity_encoding != 'known_only' or env.builder.global_dims != 0:
            raise ValueError('Facts search requires a known_only 317/0 root environment')
        seat = env.state.turn
        root_team = int(seat) & 1
        log_pi, _, legal, obs = self.blueprint.distribution([env])
        # CPU tensors alias reusable staging buffers. Retain the root tensors
        # before world validation or later policy batches overwrite the buffers.
        log_pi, legal, obs = log_pi.clone(), legal.clone(), obs.clone()
        worlds = self.particles(env, history, obs, legal)
        began_simulation = time.perf_counter()
        candidates = legal[0].nonzero().flatten()
        self.metrics['legal_actions'] += len(candidates)
        self.metrics['candidate_actions'] += len(candidates)
        simulations = []
        for aid in candidates.cpu().tolist():
            action = unrotate_compact_action_id(aid, seat)
            for world in worlds:
                simulated = clone_world(world)
                step_world(simulated, action)
                simulations.append(simulated)
        self.metrics['rollout_world_steps'] += len(simulations)
        for _ in range(self.config.depth - 1):
            active = [w for w in simulations if _finished_value(w, seat) is None]
            if not active:
                break
            probabilities = self.blueprint.distribution(active)[0].exp()
            chosen = torch.multinomial(probabilities, 1, generator=self.generator).flatten().cpu().tolist()
            for world, aid in zip(active, chosen):
                step_world(world, unrotate_compact_action_id(aid, world.state.turn))
            self.metrics['rollout_world_steps'] += len(active)
        returns = np.empty(len(simulations), dtype=np.float32)
        unfinished, positions = [], []
        for i, world in enumerate(simulations):
            result = _finished_value(world, seat)
            if result is None:
                unfinished.append(world)
                positions.append(i)
            else:
                returns[i] = result
        if unfinished:
            values = self.blueprint.values(unfinished).detach().cpu().numpy()
            for i, world, value in zip(positions, unfinished, values):
                returns[i] = value if (int(world.state.turn) & 1) == root_team else -value
        estimates = returns.reshape(len(candidates), len(worlds)).mean(-1)
        q = torch.full_like(log_pi, torch.nan)
        q[0, candidates] = torch.tensor(estimates, device=q.device)
        distribution = regularized_update(log_pi, q, legal, beta=self.config.beta)
        self.last_q = q.detach().cpu()
        self.metrics['simulation_seconds'] += time.perf_counter() - began_simulation
        action = int(torch.multinomial(distribution.exp(), 1, generator=self.generator)[0, 0])
        self.metrics['successful_decisions'] += 1
        for key, value in self.blueprint.query_counts.items():
            self.query_counts[key] += value - queries_before[key]
        self.latencies.append(time.perf_counter() - began)
        if self.capture_diagnostics:
            # Save legal entries only: illegal Q is deliberately NaN and its
            # log probability is -inf, neither belongs in portable JSON.
            ids = candidates.cpu().tolist()
            self.last_decision = {
                'root_seat': int(seat), 'beta': self.config.beta,
                'legal_compact_action_ids': ids,
                'legal_world_action_ids': [unrotate_compact_action_id(aid, seat) for aid in ids],
                'particle_returns': returns.reshape(len(ids), len(worlds)).tolist(),
                'q': estimates.tolist(),
                'pi0_log_probabilities': log_pi[0, candidates].cpu().tolist(),
                'pi0_probabilities': log_pi[0, candidates].exp().cpu().tolist(),
                'search_probabilities': distribution[0, candidates].exp().cpu().tolist(),
                'selected_compact_action_id': action,
                'particles': self.last_particle_diagnostics,
            }
        return unrotate_compact_action_id(action, seat)

    def report(self):
        return {**self.metrics, 'config': asdict(self.config),
                'inventory_backend': self.inventory_backend,
                'constraint_implementation': self.constraint_implementation,
                'joint_overlap_constraints': self.overlap_constraints,
                'matching_cache_counts': dict(self.matching_cache_counts),
                'blueprint_queries': dict(self.query_counts), 'constraint_queries': dict(self.constraint_counts),
                'belief_kind': getattr(self.joint, 'belief_kind', 'unspecified'),
                'rejection_reasons': dict(self.rejection_reasons),
                'acceptance_rate': self.metrics['accepted_proposals'] / max(1, self.metrics['proposals']),
                'action_coverage': self.metrics['candidate_actions'] / max(1, self.metrics['legal_actions']),
                'decision_seconds_total': sum(self.latencies),
                'decision_seconds_p50': float(np.median(self.latencies)) if self.latencies else 0.0,
                'decision_seconds_p95': float(np.quantile(self.latencies, .95)) if self.latencies else 0.0,
                'backend': 'full-public-history CPU reconstruction; batched facts-only torch inference',
                'proposal_semantics': getattr(self.joint, 'distribution_semantics', 'neural proposals')
                    + '; fresh samples conditioned by full replay and root observation'}


# Architecture-neutral name for new callers; historical imports remain valid.
FactsBeliefSearch = FactsJointSearch
