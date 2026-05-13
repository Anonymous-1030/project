"""
ProSE-X 2.0: Promotion-centric Sparse KV Architecture for Long-Context LLM Inference

A strict, artifact-grade, no-trick research prototype.

Key Principles:
1. Separation of offline label generation from online inference
2. No oracle information at inference time
3. All modules configurable through explicit config files
4. Machine-readable logging for all experiments
5. Built for ablation - every module individually swappable
"""

__version__ = "2.0.0"
__author__ = "ProSE Project Team"
