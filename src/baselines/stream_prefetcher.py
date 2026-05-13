"""
StreamPrefetcher — Formalized Hardware Stream Prefetch Baseline.

A generic hardware stream prefetcher that detects sequential chunk-access
strides and prefetches next-N chunks.  No access to attention scores or
content metadata.

KEY BEHAVIOR (fetch-then-verify):
  The stream prefetcher aggressively prefetches stride-detected chunks from
  CXL, but the actual GPU attention pattern may not follow the stride
  perfectly (needle jumps, RULER multi-key).  Prefetched chunks that the
  GPU does NOT actually attend to become invalid traffic (15-35%).

This formalizes the StreamPrefetcherPolicy from e2e_eval_runner.py with
full CXL queue integration for use in the unified baseline framework.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import (
    CXLQueueSimulator, CXLQueueConfig, BaselineCXLSession, StepStats
)


class StreamPrefetcherPolicy(BaselinePolicy):
    """Formalized hardware stream prefetcher with CXL queue integration.

    Models an aggressive sequential prefetcher:
      1. Detect stride from recent access history
      2. Prefetch stride_depth chunks ahead (and behind for bidir)
      3. GPU actually attends to a subset → rest = invalid traffic

    On irregular workloads (needle, RULER), stride detection frequently
    fails, causing either zero prefetch or wrong-stride prefetch.
    """

    name = "StreamPrefetcher"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        prefetch_depth: int = 3,
        prefetch_behind: int = 1,
        stream_threshold: int = 2,
        hbm_capacity_chunks: int = 16,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.prefetch_depth = prefetch_depth
        self.prefetch_behind = prefetch_behind
        self.stream_threshold = stream_threshold
        self.hbm_capacity = hbm_capacity_chunks
        self.cxl_session: Optional[BaselineCXLSession] = None

        self.access_history: List[int] = []
        self.stream_detected: bool = False
        self._last_stride: int = 0
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.access_history.clear()
        self.stream_detected = False
        self._last_stride = 0
        self.step_count = 0

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        self.step_count = step

        if self.cxl_session is None:
            self.cxl_session = BaselineCXLSession(self.cxl_config)

        anchor_set = set(anchor_ids)

        # ── Record access (prefetcher only sees addresses, not scores) ──
        if chunk_attention_masses:
            top_chunk = max(chunk_attention_masses.items(), key=lambda x: x[1])[0]
            self.access_history.append(int(top_chunk))
        else:
            self.access_history.append(0)

        # ── Stream detection ──
        prefetch_targets = []
        last_chunk = self.access_history[-1]

        if len(self.access_history) >= 3:
            stride = self.access_history[-1] - self.access_history[-2]
            prev_stride = self.access_history[-2] - self.access_history[-3]
            if stride == prev_stride and abs(stride) <= 2:
                self.stream_detected = True
                self._last_stride = stride
            else:
                self.stream_detected = False

        if self.stream_detected:
            stride = self._last_stride
            # Prefetch ahead
            for i in range(1, self.prefetch_depth + 1):
                nxt = last_chunk + i * stride
                if 0 <= nxt < num_chunks and nxt not in anchor_set:
                    prefetch_targets.append(nxt)
            # Prefetch behind (bidirectional)
            for i in range(1, self.prefetch_behind + 1):
                prv = last_chunk - i * stride
                if 0 <= prv < num_chunks and prv not in anchor_set:
                    prefetch_targets.append(prv)

        # ── Temporal locality: keep recent history ──
        recent = []
        for h in self.access_history[-self.stream_threshold:]:
            if 0 <= h < num_chunks and h not in anchor_set:
                recent.append(h)

        # ── Aggressive fill to cover budget gaps ──
        # If stream + recent < budget, fill with sequential neighbors
        sequential_fill = []
        if len(prefetch_targets) + len(recent) < budget_chunks:
            for offset in range(1, budget_chunks + 3):
                cid = last_chunk + offset
                if 0 <= cid < num_chunks and cid not in anchor_set and cid not in recent and cid not in prefetch_targets:
                    sequential_fill.append(cid)
                if len(prefetch_targets) + len(recent) + len(sequential_fill) >= budget_chunks * 3:
                    break

        # ── CXL FETCH: DMA all speculatively-fetched chunks ──
        all_fetches = list(dict.fromkeys(prefetch_targets + recent + sequential_fill))
        if not all_fetches:
            all_fetches = [c for c in range(budget_chunks) if c not in anchor_set]

        if all_fetches:
            self.cxl_session.cxl.submit_payload_fetch(all_fetches, 0)

            # ── GPU ATTENDED: which chunks were actually used ──
            if chunk_attention_masses:
                attn_threshold = sorted(chunk_attention_masses.values(), reverse=True)[
                    min(budget_chunks, len(chunk_attention_masses) - 1)
                ] * 0.3 if len(chunk_attention_masses) > budget_chunks else 0.001
                actually_used = [int(cid) for cid, mass in chunk_attention_masses.items()
                               if mass >= attn_threshold and int(cid) not in anchor_set]
            else:
                actually_used = recent[:budget_chunks]

            # Mark invalid: prefetched but not actually used by GPU
            used_set = set(actually_used)
            invalid = [c for c in all_fetches if c not in used_set]
            if invalid:
                self.cxl_session.cxl.mark_chunks_invalid(invalid)
            # Mark used
            used_in_fetch = [c for c in all_fetches if c in used_set]
            self.cxl_session.cxl.mark_chunks_used(used_in_fetch if used_in_fetch else all_fetches[:budget_chunks])
        else:
            actually_used = []

        # ── Final selection ──
        selected = anchor_set | set(all_fetches[:budget_chunks * 2])

        # ── Gold ──
        if chunk_attention_masses:
            sorted_by_attn = sorted(chunk_attention_masses.items(),
                                    key=lambda x: x[1], reverse=True)
            gold = [int(cid) for cid, _ in sorted_by_attn[:budget_chunks]
                   if int(cid) not in anchor_set]
        else:
            gold = list(selected - anchor_set)[:budget_chunks]

        self.cxl_session.end_step(list(selected), gold)
        self.cxl_session.advance_step()

        return sorted(selected)

    def get_stats(self) -> Optional[StepStats]:
        if self.cxl_session is None:
            return None
        stats_list = [r.cxl_stats for r in self.cxl_session.step_results]
        if not stats_list:
            return None
        total = StepStats()
        for s in stats_list:
            for field_name in s.__dataclass_fields__:
                current = getattr(total, field_name, 0)
                added = getattr(s, field_name, 0)
                setattr(total, field_name, current + added)
        return total

    def get_mean_recovery(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))
