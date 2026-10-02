"""Resolve policy observation requirements once, before allocating an environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from junqi_core.observation import validate_identity_encoding


@dataclass(frozen=True)
class PolicyInputContract:
    identity_encoding: str
    include_global: bool

    def observation_kwargs(self) -> dict[str, Any]:
        return {"identity_encoding": self.identity_encoding,
                "include_global": self.include_global}


def policy_input_contract(*policies: Any) -> PolicyInputContract:
    """Resolve a shared observation, rejecting different identity semantics.

    Policies explicitly declare both attributes. Historical checkpoint loaders
    construct a policy with legacy attributes; inference never guesses them.
    Legacy spatial-only policies may share the same spatial encoding with a
    policy needing globals, so globals are computed when either needs them.
    """
    if not policies:
        raise ValueError("At least one policy is required")
    contracts = []
    for policy in policies:
        encoding = getattr(policy, "identity_encoding", None)
        include_global = getattr(policy, "uses_global_summary", None)
        if type(encoding) is not str or type(include_global) is not bool:
            raise ValueError("Policy must declare identity_encoding and uses_global_summary; "
                             "load historical weights through the checkpoint adapter")
        validate_identity_encoding(encoding, include_global)
        contracts.append(PolicyInputContract(encoding, include_global))
    if any(c.identity_encoding != contracts[0].identity_encoding for c in contracts[1:]):
        raise ValueError("Cross-schema evaluation needs separate observation adapters")
    return PolicyInputContract(contracts[0].identity_encoding,
                               any(c.include_global for c in contracts))
