"""vLLM general-plugin entry point.

Registered under the `vllm.general_plugins` group, which vLLM loads in EVERY
process -- API server, engine core AND workers. This is the only mechanism that
works here: vLLM 0.19 SPAWNS the engine core (note the multiprocessing
resource_tracker alongside `VLLM::EngineCore`), so the child re-imports vllm from
scratch and any monkeypatch applied in the parent process is silently lost. The
model would then keep running stock EVS while the launcher cheerfully reported
that VidCom2 was installed.
"""

from .patch import install


def register():
    install()
