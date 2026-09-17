"""Install VidCom2 as vLLM's video pruning method, without touching site-packages.

vLLM 0.19 ships only EVS. Upstream added VidCom2 as a second
`--video-pruning-method` (PR #47750) but that is `main`-only and in no released
wheel. The change is pure Python, so instead of a source build we swap the one
function the model actually calls.

WHY THE PATCH TARGET IS `qwen3_vl`, NOT `multimodal.evs`
--------------------------------------------------------
`qwen3_vl.py` does `from vllm.multimodal.evs import compute_retention_mask` at
module scope, which binds the function object into the qwen3_vl namespace at
import time. Rebinding `vllm.multimodal.evs.compute_retention_mask` afterwards
would therefore have NO effect on the model. The name must be replaced in every
module that imported it.

`compute_retained_tokens_count` is deliberately left as EVS's: the processor uses
it to size the placeholder run before the mask exists, so sharing it is what
guarantees the counts agree.

Usage
-----
    FA_PRUNE_METHOD=vidcom2 python -m longvt_compression.vidcom2_vllm.serve <vllm args>

or, in-process:
    from longvt_compression.vidcom2_vllm import patch; patch.install()
"""

import os

# Modules that do `from vllm.multimodal.evs import compute_retention_mask`.
_TARGETS = (
    "vllm.model_executor.models.qwen3_vl",
    "vllm.model_executor.models.qwen3_vl_moe",
    "vllm.model_executor.models.qwen2_5_vl",
)

_installed = False


def install(force: bool = False) -> bool:
    """Rebind compute_retention_mask to the VidCom2 implementation.

    Returns True if the patch was applied. No-op unless FA_PRUNE_METHOD=vidcom2
    (or force=True), so the same launcher serves the stock EVS baseline too.
    """
    global _installed
    if _installed:
        return True
    method = os.environ.get("FA_PRUNE_METHOD", "evs").lower()
    if not force and method != "vidcom2":
        print(f"[vidcom2_vllm] FA_PRUNE_METHOD={method!r} -> leaving stock EVS in place")
        return False

    import importlib

    from .retention import compute_retention_mask

    import vllm.multimodal.evs as evs
    evs.compute_retention_mask_evs = evs.compute_retention_mask   # keep the original
    evs.compute_retention_mask = compute_retention_mask

    patched = []
    for name in _TARGETS:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "compute_retention_mask"):
            mod.compute_retention_mask = compute_retention_mask
            patched.append(name.rsplit(".", 1)[-1])

    if not patched:
        raise RuntimeError(
            "vidcom2 patch applied to no module -- vLLM's layout changed. "
            "Check that qwen3_vl.py still does "
            "`from vllm.multimodal.evs import compute_retention_mask`."
        )
    _installed = True
    print(f"[vidcom2_vllm] VidCom2 retention installed into: {', '.join(patched)}")
    return True
