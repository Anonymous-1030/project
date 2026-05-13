"""
Bridge configuration for ProSE-X v2 + vLLM integration.

All parameters are explicit - no magic constants.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import os
import yaml

from src.config import ProSEXv2Config


@dataclass
class BridgeConfig:
    """
    Configuration for the prosex ↔ vLLM bridge.

    Contains both the prosex promotion pipeline config and
    bridge-specific parameters for the vLLM integration.
    """

    # === ProSE-X pipeline config ===
    prosex_config: ProSEXv2Config = field(default_factory=ProSEXv2Config)

    # === Bridge master controls ===
    enable_promotion: bool = True
    """Master switch. When False, all attention is full (no filtering)."""

    pipeline_run_every_n_steps: int = 1
    """Run the promotion pipeline every N decode steps.
       On non-pipeline steps, reuse previous result with TTL decay."""

    # === Chunking ===
    anchor_ratio: float = 0.1
    """Fraction of chunks classified as anchors (always visible)."""

    chunk_size: int = 512
    """Tokens per prosex chunk. Must match prosex_config.chunk_size."""

    # === Signature computation ===
    signature_method: str = "ngram_hash"
    """Method for computing chunk/query signatures.
       Supported: 'ngram_hash', 'simhash'."""

    query_signature_window: int = 32
    """Number of recent tokens to use when computing query signature."""

    ngram_n_values: tuple = (1, 2, 3)
    """N-gram sizes for signature computation."""

    signature_dim: int = 128
    """Dimension of chunk/query signature vectors."""

    # === Attention control ===
    prefill_always_full: bool = True
    """Always use full attention during prefill (no promotion filtering)."""

    max_promoted_chunks_per_request: int = 20
    """Hard cap on number of promoted chunks per request."""

    fallback_on_pipeline_error: bool = True
    """When True, fall back to full attention if pipeline raises."""

    # === vLLM compatibility ===
    vllm_block_size: int = 16
    """vLLM block size (tokens per block). Default: 16."""

    enforce_eager: bool = True
    """Disable CUDA graphs (required for dynamic block tables)."""

    disable_prefix_caching: bool = True
    """Disable vLLM prefix caching (conflicts with sparse attention)."""

    # === Memory tracking ===
    track_promoted_memory_bytes: bool = True
    """Track and log promoted memory bytes per request."""

    # === Logging ===
    log_pipeline_decisions: bool = True
    """Log pipeline promotion/eviction decisions per step."""

    log_level: str = "INFO"

    def __post_init__(self):
        """Validate and synchronize config."""
        # Sync chunk_size between bridge and prosex configs
        if self.prosex_config.chunk_size != self.chunk_size:
            self.prosex_config.chunk_size = self.chunk_size

        # Validate ratios
        if not (0.0 <= self.anchor_ratio <= 1.0):
            raise ValueError(f"anchor_ratio must be in [0, 1], got {self.anchor_ratio}")

        if self.pipeline_run_every_n_steps < 1:
            raise ValueError(
                f"pipeline_run_every_n_steps must be >= 1, got {self.pipeline_run_every_n_steps}"
            )

        if self.query_signature_window < 1:
            raise ValueError(
                f"query_signature_window must be >= 1, got {self.query_signature_window}"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "BridgeConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BridgeConfig":
        """Create configuration from a dictionary."""
        config = cls()

        # Top-level bridge settings
        bridge_data = d.get("bridge", {})
        for key, value in bridge_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # ProSE-X pipeline config
        prosex_data = d.get("prosex", {})
        if prosex_data:
            config.prosex_config = ProSEXv2Config.from_dict(prosex_data)

        # vLLM settings (nested under bridge namespace)
        vllm_data = d.get("vllm", {})
        for key, value in vllm_data.items():
            bridge_key = f"vllm_{key}"
            if hasattr(config, bridge_key):
                setattr(config, bridge_key, value)

        # Apply any top-level overrides
        for key in ("enable_promotion", "anchor_ratio", "chunk_size",
                     "pipeline_run_every_n_steps", "query_signature_window"):
            if key in d:
                setattr(config, key, d[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "bridge": {
                "enable_promotion": self.enable_promotion,
                "pipeline_run_every_n_steps": self.pipeline_run_every_n_steps,
                "anchor_ratio": self.anchor_ratio,
                "chunk_size": self.chunk_size,
                "signature_method": self.signature_method,
                "query_signature_window": self.query_signature_window,
                "prefill_always_full": self.prefill_always_full,
                "max_promoted_chunks_per_request": self.max_promoted_chunks_per_request,
                "fallback_on_pipeline_error": self.fallback_on_pipeline_error,
                "track_promoted_memory_bytes": self.track_promoted_memory_bytes,
                "log_pipeline_decisions": self.log_pipeline_decisions,
            },
            "prosex": self.prosex_config.to_dict(),
            "vllm": {
                "block_size": self.vllm_block_size,
                "enforce_eager": self.enforce_eager,
                "disable_prefix_caching": self.disable_prefix_caching,
            },
        }
