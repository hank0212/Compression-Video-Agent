"""VidCom2 video-token pruning for vLLM 0.19, as an out-of-tree patch."""
from .retention import compute_retention_mask  # noqa: F401
