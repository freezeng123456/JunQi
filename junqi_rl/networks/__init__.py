"""junqi_rl.networks — Neural network architectures for 四国军棋 RL.

Networks
--------
JunqiNet
    Main policy + value network.  Input: 412-channel 17×17 spatial tensor
    (canonical frame) + 28-dim global vector.  Output: policy logits over the
    16,641-dim compact action space (129 on-board cells, src × dst) + scalar
    (or categorical) value estimate.

TransformerBlock
    Pre-norm residual transformer layer with multi-head self-attention and
    position-wise feed-forward network.

SpatialEncoder
    CNN stem that converts (C, 17, 17) spatial observations to a flat token
    sequence ready for the transformer trunk.

JunqiNetConfig
    Frozen dataclass describing all hyper-parameters.
"""

from .junqi_net import JunqiNet, JunqiNetConfig

__all__ = ["JunqiNet", "JunqiNetConfig"]
