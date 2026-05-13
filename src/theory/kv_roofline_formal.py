"""Formal KV-cache roofline derivation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict


@dataclass
class ChunkUtilityRecord:
    chunk_id: int
    utility: float
    attention_mass: float
    bytes_transferred: int


class FormalKVCacheRoofline:
    """Formalizes the KV-cache operational intensity used in the paper."""

    def compute_operational_intensity(self, records: Iterable[ChunkUtilityRecord]) -> float:
        records = list(records)
        denom = sum(max(r.bytes_transferred, 0) for r in records)
        if denom <= 0:
            return 0.0
        numer = sum(max(r.utility, 0.0) * max(r.attention_mass, 0.0) for r in records)
        return numer / denom

    def ridge_condition(self, compute_ceiling: float, bandwidth_ceiling: float) -> float:
        if bandwidth_ceiling <= 0:
            return float("inf")
        return compute_ceiling / bandwidth_ceiling

    def classify_bound(self, oi: float, compute_ceiling: float, bandwidth_ceiling: float) -> str:
        ridge = self.ridge_condition(compute_ceiling, bandwidth_ceiling)
        return "bandwidth" if oi < ridge else "compute"

    def summarize(self, records: List[ChunkUtilityRecord], compute_ceiling: float, bandwidth_ceiling: float) -> Dict[str, float | str]:
        oi = self.compute_operational_intensity(records)
        return {
            "operational_intensity": oi,
            "ridge_point": self.ridge_condition(compute_ceiling, bandwidth_ceiling),
            "bound": self.classify_bound(oi, compute_ceiling, bandwidth_ceiling),
            "total_bytes": sum(r.bytes_transferred for r in records),
            "total_weighted_utility": sum(r.utility * r.attention_mass for r in records),
        }