"""
ProSE-X v2 ↔ vLLM Bridge

Integrates prosex's promotion-centric sparse KV cache management
into vLLM's inference engine. During decode, only "promoted" KV cache
blocks participate in attention, reducing memory bandwidth while
maintaining output quality through prosex's 5-stage pipeline.

Usage:
    from vllm import LLM
    from src.bridge import integrate_with_vllm

    # Create vLLM engine with required settings
    llm = LLM(
        model="Qwen/Qwen2-7B-Instruct",
        enforce_eager=True,           # Required: no CUDA graphs
        enable_prefix_caching=False,  # Required: conflicts with sparse attn
    )

    # Install prosex bridge
    hook = integrate_with_vllm(llm, "configs/bridge/default.yaml")

    # Use normally — promotion pipeline runs automatically
    outputs = llm.generate(prompts, sampling_params)

    # Get stats
    print(hook.get_stats())
"""

from src.bridge.config import BridgeConfig
from src.bridge.signature import (
    compute_chunk_signature,
    compute_query_signature,
    signature_similarity,
)
from src.bridge.chunk_builder import build_chunks_from_request, classify_anchor_tail
from src.bridge.block_mapper import BlockChunkMapper
from src.bridge.request_state import BridgeStateManager, ProseRequestState
from src.bridge.pipeline_bridge import ProsePipelineBridge
from src.bridge.vllm_hook import ProseVLLMHook

__all__ = [
    # Config
    "BridgeConfig",
    # Signature
    "compute_chunk_signature",
    "compute_query_signature",
    "signature_similarity",
    # Chunk builder
    "build_chunks_from_request",
    "classify_anchor_tail",
    # Block mapper
    "BlockChunkMapper",
    # State management
    "BridgeStateManager",
    "ProseRequestState",
    # Pipeline bridge
    "ProsePipelineBridge",
    # vLLM hook
    "ProseVLLMHook",
]


def integrate_with_vllm(
    llm_engine: object,
    bridge_config_path: str,
) -> ProseVLLMHook:
    """
    Install the prosex bridge on a vLLM engine.

    This is the main entry point for integrating prosex with vLLM.
    It locates the GPU model runner inside the engine and installs
    hooks for promotion-based block table filtering.

    Args:
        llm_engine: A vLLM LLM instance or LLMEngine instance.
        bridge_config_path: Path to bridge YAML configuration file.

    Returns:
        ProseVLLMHook instance (can be used to get stats or uninstall).

    Raises:
        ValueError: If the vLLM engine type is not recognized.
        RuntimeError: If the GPU model runner cannot be found.
        FileNotFoundError: If the config file doesn't exist.

    Example:
        >>> from vllm import LLM
        >>> from src.bridge import integrate_with_vllm
        >>>
        >>> llm = LLM(model="Qwen/Qwen2-7B-Instruct",
        ...           enforce_eager=True,
        ...           enable_prefix_caching=False)
        >>> hook = integrate_with_vllm(llm, "configs/bridge/default.yaml")
        >>> outputs = llm.generate(["Hello, world!"])
        >>> print(hook.get_stats())
        >>> hook.uninstall()
    """
    import os

    if not os.path.exists(bridge_config_path):
        raise FileNotFoundError(
            f"Bridge config file not found: {bridge_config_path}"
        )

    # Load configuration
    config = BridgeConfig.from_yaml(bridge_config_path)

    # Find the model runner
    model_runner = _find_model_runner(llm_engine)

    if model_runner is None:
        raise RuntimeError(
            "Could not find GPU model runner in the vLLM engine. "
            "Make sure you are using a GPU-enabled vLLM installation "
            "and the engine has been initialized."
        )

    # Install hook
    hook = ProseVLLMHook(config)
    hook.install(model_runner)

    return hook


def _find_model_runner(engine: object) -> object:
    """
    Walk the vLLM engine object graph to find the GPU model runner.

    vLLM V1 architecture:
      LLM → LLMEngine → EngineCore → EngineCoreClient → ... → GPUModelRunner

    We try multiple paths to handle different vLLM versions and configurations.
    """
    # Path 1: LLM.llm_engine.engine_core.model_executor (or similar)
    if hasattr(engine, "llm_engine"):
        engine = engine.llm_engine

    # Path 2: LLMEngine → engine_core
    engine_core = None
    for attr in ("engine_core", "_engine_core", "engine"):
        engine_core = getattr(engine, attr, None)
        if engine_core is not None:
            break

    if engine_core is not None:
        # Path 3: EngineCore → model_executor
        executor = None
        for attr in ("model_executor", "_executor", "executor"):
            executor = getattr(engine_core, attr, None)
            if executor is not None:
                break

        if executor is not None:
            # Path 4: Executor → workers → model_runner
            # Try different executor types
            for attr in ("workers", "_workers", "worker"):
                workers = getattr(executor, attr, None)
                if workers is not None:
                    if isinstance(workers, list) and len(workers) > 0:
                        worker = workers[0]
                    else:
                        worker = workers

                    for runner_attr in ("model_runner", "gpu_model_runner", "_model_runner"):
                        runner = getattr(worker, runner_attr, None)
                        if runner is not None:
                            return runner

            # Direct model_runner on executor
            for runner_attr in ("model_runner", "gpu_model_runner", "_model_runner"):
                runner = getattr(executor, runner_attr, None)
                if runner is not None:
                    return runner

    # Path 5: Engine might have a direct reference
    for attr in ("model_runner", "_model_runner", "gpu_runner"):
        runner = getattr(engine, attr, None)
        if runner is not None:
            return runner

    # Path 6: Last resort — search all attributes recursively (shallow)
    for attr_name in dir(engine):
        if attr_name.startswith("_"):
            continue
        try:
            attr_value = getattr(engine, attr_name)
            if hasattr(attr_value, "execute_model") and hasattr(attr_value, "block_size"):
                return attr_value
        except Exception:
            pass

    return None
