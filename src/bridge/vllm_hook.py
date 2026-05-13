"""
vLLM Model Runner Hook: injects prosex's promotion pipeline into vLLM's
inference loop via runtime monkey-patching at well-defined hook points.

Design principles:
1. ZERO modifications to vLLM source files
2. Minimal hook surface (2-3 patch points)
3. Graceful fallback on errors (full attention if pipeline fails)
4. Thread-safe state management
5. Clean uninstall capability for testing

Hook points:
  A. GPUModelRunner._update_states
     → Detect request add/remove lifecycle events

  B. GPUModelRunner._build_attention_metadata (or prepare_inputs)
     → Filter block tables based on promotion decisions before
       attention metadata is built

Usage:
    from vllm import LLM
    from src.bridge import integrate_with_vllm

    llm = LLM(model="...", enforce_eager=True, enable_prefix_caching=False)
    hook = integrate_with_vllm(llm, "configs/bridge/default.yaml")
    # ... use llm normally ...
    hook.uninstall()
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from src.bridge.config import BridgeConfig
from src.bridge.request_state import BridgeStateManager

logger = logging.getLogger(__name__)

# Sentinel for "not yet initialized"
_UNSET = object()


class ProseVLLMHook:
    """
    Hooks into vLLM's GPU model runner to enable prosex KV cache filtering.

    Installs persistence at these injection points:
    1. _update_states — wrapped to track request lifecycle
    2. _build_attention_metadata — wrapped to filter block tables
    """

    def __init__(self, bridge_config: BridgeConfig):
        self.config = bridge_config
        self.state_manager: Optional[BridgeStateManager] = None
        self._model_runner: Any = None
        self._installed: bool = False

        # Saved originals for uninstall
        self._original_update_states: Optional[Callable] = None
        self._original_build_attn_metadata: Optional[Callable] = None

        # Model info (populated during install)
        self._num_layers: int = 32
        self._num_kv_heads: int = 8
        self._head_dim: int = 128
        self._dtype_size: int = 2

        # Tracking for per-step request discovery
        self._known_request_ids: Set[str] = set()
        self._step_count: int = 0

        # Stats
        self.total_hook_calls: int = 0
        self.total_filtered_steps: int = 0
        self.total_errors: int = 0

    # ------------------------------------------------------------------
    # Install / Uninstall
    # ------------------------------------------------------------------

    def install(self, model_runner: Any) -> "ProseVLLMHook":
        """
        Install hooks on a vLLM GPU model runner.

        Args:
            model_runner: vLLM GPUModelRunner instance
                          (from vllm.v1.worker.gpu_model_runner).

        Returns:
            self (for method chaining).

        Raises:
            RuntimeError: If hooks are already installed.
        """
        if self._installed:
            raise RuntimeError("ProseVLLMHook is already installed. Call uninstall() first.")

        self._model_runner = model_runner

        # Extract model configuration from the runner
        self._extract_model_info(model_runner)

        # Initialize state manager
        self.state_manager = BridgeStateManager(self.config)

        # ---- Hook A: _update_states (request lifecycle) ----
        if hasattr(model_runner, "_update_states"):
            self._original_update_states = model_runner._update_states
            hook_self = self  # capture for closure

            def patched_update_states(*args: Any, **kwargs: Any) -> Any:
                result = hook_self._original_update_states(*args, **kwargs)
                try:
                    hook_self._on_states_updated()
                except Exception as e:
                    logger.error(f"Error in _on_states_updated: {e}", exc_info=True)
                    hook_self.total_errors += 1
                return result

            model_runner._update_states = patched_update_states
            logger.info("Hooked _update_states for request lifecycle tracking")
        else:
            logger.warning(
                "Could not find _update_states on model runner. "
                "Request lifecycle tracking will use fallback method."
            )

        # ---- Hook B: _build_attention_metadata (block table filtering) ----
        if hasattr(model_runner, "_build_attention_metadata"):
            self._original_build_attn_metadata = model_runner._build_attention_metadata
            hook_self = self  # capture for closure

            def patched_build_attn_metadata(*args: Any, **kwargs: Any) -> Any:
                try:
                    hook_self._before_attention_metadata()
                except Exception as e:
                    logger.error(
                        f"Error in _before_attention_metadata: {e}", exc_info=True
                    )
                    hook_self.total_errors += 1
                return hook_self._original_build_attn_metadata(*args, **kwargs)

            model_runner._build_attention_metadata = patched_build_attn_metadata
            logger.info("Hooked _build_attention_metadata for block table filtering")
        else:
            logger.warning(
                "Could not find _build_attention_metadata on model runner. "
                "Block table filtering will not be active."
            )

        # Store hook reference on model runner for external access
        model_runner._prose_hook = self

        self._installed = True
        logger.info(
            f"ProseVLLMHook installed successfully. "
            f"enable_promotion={self.config.enable_promotion}, "
            f"chunk_size={self.config.chunk_size}"
        )
        return self

    def uninstall(self) -> None:
        """
        Remove hooks and restore original methods.

        Safe to call multiple times.
        """
        if not self._installed or self._model_runner is None:
            return

        if self._original_update_states is not None:
            self._model_runner._update_states = self._original_update_states
            self._original_update_states = None

        if self._original_build_attn_metadata is not None:
            self._model_runner._build_attention_metadata = (
                self._original_build_attn_metadata
            )
            self._original_build_attn_metadata = None

        if hasattr(self._model_runner, "_prose_hook"):
            delattr(self._model_runner, "_prose_hook")

        self._model_runner = None
        self._installed = False
        logger.info("ProseVLLMHook uninstalled")

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _on_states_updated(self) -> None:
        """
        Called after vLLM updates request states each scheduler step.

        Detects new/removed requests and updates the bridge state manager.
        """
        if self.state_manager is None:
            return

        # Discover current active requests from the model runner's input batch
        active_ids = self._get_active_request_ids()
        if active_ids is None:
            return

        # Detect new requests
        new_ids = active_ids - self._known_request_ids
        for req_id in new_ids:
            self._on_new_request(req_id)

        # Detect removed requests
        removed_ids = self._known_request_ids - active_ids
        for req_id in removed_ids:
            self.state_manager.on_request_removed(req_id)

        # Periodic orphan cleanup (every 100 steps)
        self._step_count += 1
        if self._step_count % 100 == 0:
            self.state_manager.cleanup_orphaned(active_ids)

        self._known_request_ids = active_ids

    def _before_attention_metadata(self) -> None:
        """
        Called just before attention metadata is built.

        This is where we filter block tables based on promotion decisions.
        We modify the InputBatch's block_table in-place so that when
        _build_attention_metadata computes the GPU block table tensor,
        it uses our filtered block IDs.
        """
        if self.state_manager is None or self._model_runner is None:
            return

        if not self.config.enable_promotion:
            return

        self.total_hook_calls += 1

        try:
            self._filter_block_tables_for_batch()
        except Exception as e:
            logger.error(f"Block table filtering failed: {e}", exc_info=True)
            self.total_errors += 1
            # Continue with full block tables (no filtering)

    # ------------------------------------------------------------------
    # Block table filtering
    # ------------------------------------------------------------------

    def _filter_block_tables_for_batch(self) -> None:
        """
        Filter block tables for all active decode requests.

        Modifies the InputBatch block tables in-place:
        - Prefill requests: no filtering (full attention)
        - Decode requests: compact block table to promoted blocks only

        This is the critical path — it needs to be fast and never crash.
        """
        model_runner = self._model_runner
        input_batch = getattr(model_runner, "_input_batch", None)
        if input_batch is None:
            # Try alternate name
            input_batch = getattr(model_runner, "input_batch", None)
        if input_batch is None:
            return

        # Get request IDs and block tables from input batch
        req_ids = self._get_request_ids_from_batch(input_batch)
        if not req_ids:
            return

        # Check which requests are in prefill vs decode
        is_prefill_map = self._get_prefill_map(input_batch, req_ids)

        for req_idx, req_id in enumerate(req_ids):
            if req_id is None:
                continue

            # Skip prefill requests
            if is_prefill_map.get(req_id, False):
                continue

            state = self.state_manager.get_state(req_id)
            if state is None:
                continue

            # Get current block table for this request
            full_block_table = self._get_request_block_table(
                input_batch, req_idx, req_id
            )
            if full_block_table is None:
                continue

            # Determine new token for this step
            new_token_id = self._get_new_token_id(input_batch, req_idx)
            if new_token_id is None:
                # Can't determine new token, skip filtering but mark step
                continue

            # Run pipeline via state manager
            try:
                filtered_blocks, new_seq_len, was_filtered = (
                    self.state_manager.on_decode_step(
                        request_id=req_id,
                        new_token_id=new_token_id,
                        current_block_table=full_block_table,
                        total_tokens=state.total_tokens + 1,
                    )
                )

                if was_filtered:
                    self.total_filtered_steps += 1
                    # Update the block table in-place
                    self._set_request_block_table(
                        input_batch, req_idx, req_id, filtered_blocks
                    )
                    # Update seq_len
                    self._set_request_seq_len(input_batch, req_idx, req_id, new_seq_len)

            except Exception as e:
                logger.error(
                    f"Pipeline failed for request {req_id}: {e}", exc_info=True
                )
                # Fall through — full block table remains

    # ------------------------------------------------------------------
    # Request lifecycle helpers
    # ------------------------------------------------------------------

    def _on_new_request(self, request_id: str) -> None:
        """
        Handle a new request entering the system.

        Extracts prompt tokens and initializes prosex state.
        """
        if self.state_manager is None:
            return

        # Try to get prompt tokens from the model runner
        prompt_token_ids = self._get_prompt_tokens(request_id)
        if prompt_token_ids is None:
            logger.warning(
                f"Could not extract prompt tokens for request {request_id}. "
                f"Using empty prompt — promotion will be inactive."
            )
            prompt_token_ids = []

        self.state_manager.on_request_added(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            dtype_size=self._dtype_size,
        )

        # Mark prefill as "complete" immediately — we don't filter during prefill
        # The actual block mapping will be built on first decode step
        state = self.state_manager.get_state(request_id)
        if state and prompt_token_ids:
            state.prefill_complete = True

    # ------------------------------------------------------------------
    # Data extraction from vLLM internals
    # ------------------------------------------------------------------

    def _extract_model_info(self, model_runner: Any) -> None:
        """
        Extract model dimensions from the model runner.

        Attempts multiple strategies to find model config:
        1. model_runner.vllm_config.model_config
        2. model_runner.model_config
        3. model_runner._model.config (HF config)
        """
        try:
            # Strategy 1: vllm_config
            vllm_config = getattr(model_runner, "vllm_config", None)
            if vllm_config is not None:
                model_config = getattr(vllm_config, "model_config", None)
                if model_config is not None:
                    hf_config = getattr(model_config, "hf_config", None)
                    if hf_config is not None:
                        self._num_layers = getattr(
                            hf_config, "num_hidden_layers",
                            getattr(hf_config, "num_layers", 32)
                        )
                        self._num_kv_heads = getattr(
                            hf_config, "num_key_value_heads",
                            getattr(hf_config, "num_attention_heads", 8)
                        )
                        self._head_dim = getattr(
                            hf_config, "head_dim",
                            getattr(hf_config, "hidden_size", 1024) // max(
                                getattr(hf_config, "num_attention_heads", 8), 1
                            )
                        )
                        # Try to detect dtype
                        dtype = getattr(model_config, "dtype", "float16")
                        self._dtype_size = 2 if "16" in str(dtype) else (
                            1 if "8" in str(dtype) else 4
                        )
                        logger.info(
                            f"Model info extracted: {self._num_layers} layers, "
                            f"{self._num_kv_heads} KV heads, "
                            f"{self._head_dim} head_dim, "
                            f"dtype_size={self._dtype_size}"
                        )
                        return

            # Strategy 2: model_config directly
            model_config = getattr(model_runner, "model_config", None)
            if model_config is not None:
                # Similar extraction...
                pass

            # Strategy 3: _model.config
            model = getattr(model_runner, "_model", None)
            if model is not None:
                hf_config = getattr(model, "config", None)
                if hf_config is not None:
                    self._num_layers = getattr(
                        hf_config, "num_hidden_layers",
                        getattr(hf_config, "num_layers", 32)
                    )
                    self._num_kv_heads = getattr(
                        hf_config, "num_key_value_heads",
                        getattr(hf_config, "num_attention_heads", 8)
                    )
                    self._head_dim = getattr(hf_config, "head_dim", 128)
                    logger.info(
                        f"Model info extracted from HF config: "
                        f"{self._num_layers}L, {self._num_kv_heads}KVh, {self._head_dim}d"
                    )
                    return

        except Exception as e:
            logger.warning(f"Could not extract model info: {e}. Using defaults.")

        logger.info(f"Using default model dimensions: "
                     f"layers={self._num_layers}, kv_heads={self._num_kv_heads}, "
                     f"head_dim={self._head_dim}")

    def _get_active_request_ids(self) -> Optional[Set[str]]:
        """Get currently active request IDs from the model runner."""
        if self._model_runner is None:
            return None

        # Try input_batch
        input_batch = getattr(self._model_runner, "_input_batch", None)
        if input_batch is None:
            input_batch = getattr(self._model_runner, "input_batch", None)

        if input_batch is not None:
            req_ids = getattr(input_batch, "req_ids", None)
            if req_ids is not None:
                return {r for r in req_ids if r is not None}

        return set()

    def _get_request_ids_from_batch(self, input_batch: Any) -> List[Optional[str]]:
        """Get ordered request IDs from an input batch."""
        req_ids = getattr(input_batch, "req_ids", None)
        if req_ids is None:
            return []
        if isinstance(req_ids, list):
            return req_ids
        if isinstance(req_ids, torch.Tensor):
            return [str(r.item()) if r is not None else None for r in req_ids]
        return []

    def _get_prefill_map(
        self, input_batch: Any, req_ids: List[Optional[str]]
    ) -> Dict[str, bool]:
        """
        Determine which requests are in prefill vs decode.

        In vLLM V1, during prefill, the number of query tokens per request
        is > 1. During decode, it's exactly 1.
        """
        prefill_map: Dict[str, bool] = {}

        # Strategy: check num_tokens or query_start_loc
        # During prefill, query counts are larger
        query_start_loc = getattr(input_batch, "query_start_loc", None)
        num_computed_tokens = getattr(input_batch, "num_computed_tokens", None)

        for idx, req_id in enumerate(req_ids):
            if req_id is None:
                continue

            is_prefill = False
            if query_start_loc is not None and isinstance(query_start_loc, torch.Tensor):
                if idx + 1 < len(query_start_loc):
                    q_start = query_start_loc[idx].item()
                    q_end = query_start_loc[idx + 1].item()
                    num_query_tokens = q_end - q_start
                    is_prefill = num_query_tokens > 1

            if num_computed_tokens is not None:
                if isinstance(num_computed_tokens, list) and idx < len(num_computed_tokens):
                    if num_computed_tokens[idx] == 0:
                        is_prefill = True
                elif isinstance(num_computed_tokens, torch.Tensor):
                    if idx < len(num_computed_tokens) and num_computed_tokens[idx].item() == 0:
                        is_prefill = True

            prefill_map[req_id] = is_prefill

        return prefill_map

    def _get_request_block_table(
        self, input_batch: Any, req_idx: int, req_id: str
    ) -> Optional[List[int]]:
        """
        Extract the block table for a specific request from the input batch.

        Returns ordered list of physical block IDs (no -1 padding).
        """
        # Try block_table attribute
        block_table = getattr(input_batch, "block_table", None)
        if block_table is None:
            return None

        # block_table is a MultiGroupBlockTable — access per-group data
        # For simplicity, we look at the first (primary) kv cache group
        if hasattr(block_table, "block_tables"):
            # List of per-group BlockTable objects
            if len(block_table.block_tables) > 0:
                bt = block_table.block_tables[0]
                if hasattr(bt, "get_request_blocks"):
                    blocks = bt.get_request_blocks(req_idx)
                    if blocks is not None:
                        if isinstance(blocks, torch.Tensor):
                            blocks = blocks.tolist()
                        return [b for b in blocks if isinstance(b, int) and b >= 0]

        # Try as 2D tensor
        if isinstance(block_table, torch.Tensor):
            if block_table.dim() == 2 and req_idx < block_table.shape[0]:
                row = block_table[req_idx]
                return [int(b) for b in row.tolist() if int(b) >= 0]

        # Try per-request dict
        if isinstance(block_table, dict):
            blocks = block_table.get(req_id) or block_table.get(req_idx)
            if blocks is not None:
                if isinstance(blocks, torch.Tensor):
                    blocks = blocks.tolist()
                return [b for b in blocks if isinstance(b, int) and b >= 0]

        return None

    def _set_request_block_table(
        self,
        input_batch: Any,
        req_idx: int,
        req_id: str,
        filtered_blocks: List[int],
    ) -> None:
        """
        Update the block table for a specific request with filtered blocks.

        This is best-effort — we try multiple paths to set the block table.
        If none work, we log a warning and the request runs with full attention.
        """
        block_table = getattr(input_batch, "block_table", None)
        if block_table is None:
            return

        # Try setting via MultiGroupBlockTable
        if hasattr(block_table, "block_tables") and len(block_table.block_tables) > 0:
            bt = block_table.block_tables[0]
            if hasattr(bt, "set_request_blocks"):
                bt.set_request_blocks(req_idx, filtered_blocks)
                return
            if hasattr(bt, "block_table"):
                # Internal 2D tensor
                tensor = bt.block_table
                if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:
                    if req_idx < tensor.shape[0]:
                        # Pad filtered_blocks to match width
                        width = tensor.shape[1]
                        padded = filtered_blocks + [-1] * (width - len(filtered_blocks))
                        tensor[req_idx] = torch.tensor(
                            padded[:width], dtype=tensor.dtype, device=tensor.device
                        )
                        return

        # Try as 2D tensor directly
        if isinstance(block_table, torch.Tensor) and block_table.dim() == 2:
            if req_idx < block_table.shape[0]:
                width = block_table.shape[1]
                padded = filtered_blocks + [-1] * (width - len(filtered_blocks))
                block_table[req_idx] = torch.tensor(
                    padded[:width], dtype=block_table.dtype, device=block_table.device
                )
                return

        logger.debug(f"Could not set block table for request {req_id}")

    def _set_request_seq_len(
        self,
        input_batch: Any,
        req_idx: int,
        req_id: str,
        new_seq_len: int,
    ) -> None:
        """Update the sequence length for a request after promotion filtering."""
        seq_lens = getattr(input_batch, "seq_lens", None)
        if seq_lens is None:
            seq_lens = getattr(input_batch, "seq_lens_cpu", None)

        if seq_lens is not None:
            if isinstance(seq_lens, torch.Tensor) and req_idx < len(seq_lens):
                seq_lens[req_idx] = new_seq_len
            elif isinstance(seq_lens, list) and req_idx < len(seq_lens):
                seq_lens[req_idx] = new_seq_len

    def _get_new_token_id(self, input_batch: Any, req_idx: int) -> Optional[int]:
        """Get the new token ID for a decode request."""
        # Try token_ids from input batch
        token_ids = getattr(input_batch, "token_ids", None)
        if token_ids is not None:
            if isinstance(token_ids, torch.Tensor):
                if req_idx < len(token_ids):
                    return int(token_ids[req_idx].item())

        # Try query_start_loc to determine query tokens
        query_start_loc = getattr(input_batch, "query_start_loc", None)
        if query_start_loc is not None and isinstance(query_start_loc, torch.Tensor):
            if req_idx + 1 < len(query_start_loc):
                q_start = query_start_loc[req_idx].item()
                q_end = query_start_loc[req_idx + 1].item()
                # For decode, there's exactly 1 query token
                # We don't know the actual token ID, but we can indicate
                # that a decode is happening
                if q_end - q_start == 1:
                    # Use a hash of request_id as a proxy token for signature
                    return hash(req_idx) & 0x7FFFFFFF

        return None

    def _get_prompt_tokens(self, request_id: str) -> Optional[List[int]]:
        """
        Extract prompt token IDs for a request.

        Attempts to find from the model runner's request tracking.
        """
        model_runner = self._model_runner
        if model_runner is None:
            return None

        # Try requests dict on model runner
        requests = getattr(model_runner, "requests", None)
        if requests is not None and isinstance(requests, dict):
            req = requests.get(request_id)
            if req is not None:
                prompt_token_ids = getattr(req, "prompt_token_ids", None)
                if prompt_token_ids is not None:
                    return list(prompt_token_ids)
                # Alternative attribute names
                all_token_ids = getattr(req, "all_token_ids", None)
                if all_token_ids is not None:
                    return list(all_token_ids)

        return None

    # ------------------------------------------------------------------
    # Public API for external callers
    # ------------------------------------------------------------------

    def get_promoted_block_ids(self, request_id: str) -> List[int]:
        """Get currently promoted block IDs for a request."""
        if self.state_manager is None:
            return []
        return self.state_manager.get_promoted_block_ids(request_id)

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        stats = {
            "total_hook_calls": self.total_hook_calls,
            "total_filtered_steps": self.total_filtered_steps,
            "total_errors": self.total_errors,
            "installed": self._installed,
            "known_requests": len(self._known_request_ids),
        }
        if self.state_manager is not None:
            stats.update(self.state_manager.get_stats())
        return stats
