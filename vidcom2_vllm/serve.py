"""vLLM OpenAI server with VidCom2 pruning patched in before startup.

    FA_PRUNE_METHOD=vidcom2 python -m longvt_compression.vidcom2_vllm.serve \
        <model> --video-pruning-rate 0.25 --port 8011 ...

Identical to `vllm serve` in every other respect -- argv is forwarded untouched.
With FA_PRUNE_METHOD unset (or =evs) this is exactly stock vLLM, which is how the
EVS control arm is served from the same launcher.
"""
import sys

from . import patch


def main():
    patch.install()
    from vllm.entrypoints.cli.main import main as vllm_main
    if len(sys.argv) > 1 and sys.argv[1] != "serve":
        sys.argv.insert(1, "serve")
    vllm_main()


if __name__ == "__main__":
    main()
