"""Identity learning interfaces.

``joint`` packages observer facts and public history without rule probabilities.
Historical marginal trainers and probability publication remain available as
lazy compatibility exports; importing the facts interface does not load them.
"""

from importlib import import_module

_EXPORTS = {
    "BeliefBuffer": ("buffer", "BeliefBuffer"),
    "BeliefSample": ("buffer", "BeliefSample"),
    "RevealTracker": ("reveal_tracker", "RevealTracker"),
    "refresh_beliefs_neural": ("inference", "refresh_beliefs_neural"),
    "belief_logits_to_upload_shape": ("inference", "belief_logits_to_upload_shape"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value

__all__ = [
    "BeliefBuffer",
    "BeliefSample",
    "RevealTracker",
    "refresh_beliefs_neural",
    "belief_logits_to_upload_shape",
]
