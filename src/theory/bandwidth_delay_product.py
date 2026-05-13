"""Bandwidth-Delay Product (BDP) Theorem for KV Promotion.

Adapts the classical networking BDP concept to KV cache promotion over
CXL/PCIe interconnects.  Provides hardware sizing guidance for the
prefetch buffer depth in the KCMC.

Definition (KV Promotion BDP):
  BDP_KV = B_link × D_decision

  where:
    B_link     = effective interconnect bandwidth (bytes/s) after protocol overhead
    D_decision = promotion decision latency (seconds), defined as the time from
                 "chunk identified as needed" to "chunk available in HBM"
               = D_score + D_schedule + D_transfer + D_decompress

Theorem 8 (Optimal Prefetch Buffer Depth):
  The minimum prefetch buffer depth to fully hide promotion latency is:
      N_buf* = ⌈BDP_KV / chunk_bytes⌉

  With N_buf < N_buf*, the per-step exposed latency is:
      L_exposed = max(0, D_decision - N_buf × chunk_bytes / B_link)

  With N_buf ≥ N_buf*, promotion is fully pipelined (zero exposed latency),
  assuming prefetch predictions are correct.

  Proof: The prefetch engine must keep N_buf chunks "in flight" (being
  transferred or staged) to saturate the link.  Each chunk occupies the
  link for chunk_bytes / B_link seconds.  To cover D_decision seconds of
  pipeline depth, we need N_buf ≥ D_decision / (chunk_bytes / B_link)
  = D_decision × B_link / chunk_bytes = BDP_KV / chunk_bytes.  ∎

Theorem 9 (Throughput-Latency Tradeoff under Queuing):
  Model the CXL memory controller as an M/D/1 queue with:
    - Arrival rate: λ = chunks_per_step / T_step
    - Service time: S = chunk_bytes / B_link + D_dram
    - Offered load: ρ = λ × S

  Average promotion latency:
    L_avg = S + ρ·S / (2·(1-ρ))     for ρ < 1  (stable)
    L_avg → ∞                         for ρ ≥ 1  (unstable)

  This gives the throughput-latency curve for Figure 9 in the paper.

Theorem 10 (Multi-Tenant BDP Scaling):
  With K tenants sharing bandwidth B_link under fair scheduling:
    - Per-tenant bandwidth: B_eff = B_link / K
    - Per-tenant BDP: BDP_tenant = B_eff × D_decision = BDP_KV / K
    - Per-tenant buffer: N_tenant = ⌈BDP_tenant / chunk_bytes⌉
    - Total buffer:      N_total = K × N_tenant

  Key insight: total buffer scales linearly with K, but each tenant's
  buffer shrinks as 1/K.  At K > BDP_KV / chunk_bytes, each tenant
  needs only 1 buffer slot (minimum), and contention dominates.

References:
  - Jacobson, "Congestion Avoidance and Control", SIGCOMM 1988 (BDP)
  - Kim et al., "Pond: CXL-Based Memory Pooling", ASPLOS 2023
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BDPResult:
    """Result of BDP computation for a hardware configuration."""
    bandwidth_gbps: float
    decision_latency_us: float
    bdp_bytes: int
    chunk_bytes: int
    optimal_buffer_depth: int
    # Latency components
    score_latency_us: float = 0.0
    schedule_latency_us: float = 0.0
    transfer_latency_us: float = 0.0
    decompress_latency_us: float = 0.0


@dataclass
class ThroughputLatencyPoint:
    """One point on the throughput-latency curve."""
    offered_load_rho: float
    arrival_rate: float       # chunks/us
    service_time_us: float
    avg_latency_us: float
    queue_wait_us: float
    stable: bool


@dataclass
class MultiTenantBDPResult:
    """BDP analysis for multi-tenant scenario."""
    num_tenants: int
    total_bandwidth_gbps: float
    per_tenant_bw_gbps: float
    per_tenant_bdp_bytes: int
    per_tenant_buffer_depth: int
    total_buffer_depth: int
    total_buffer_bytes: int
    contention_dominated: bool  # True when per-tenant buffer = 1


@dataclass
class HardwareSizingRecommendation:
    """Recommended hardware parameters from BDP analysis."""
    config_name: str
    bandwidth_gbps: float
    decision_latency_us: float
    bdp_bytes: int
    recommended_buffer_depth: int
    buffer_sram_bytes: int
    buffer_area_mm2: float     # CACTI 7nm estimate
    buffer_power_mw: float


# Pre-computed hardware configurations
_HW_CONFIGS = {
    "H100-SXM-CXL3.0": {"bw": 64.0, "lat_us": 2.0, "chunk": 65536},
    "H100-PCIe-CXL2.0": {"bw": 32.0, "lat_us": 5.0, "chunk": 65536},
    "A100-PCIe":         {"bw": 32.0, "lat_us": 7.0, "chunk": 65536},
    "L40S-PCIe":         {"bw": 32.0, "lat_us": 8.0, "chunk": 65536},
}


class BDPAnalyzer:
    """Bandwidth-Delay Product analyzer for KV promotion.

    Provides hardware sizing guidance and throughput-latency analysis.
    """

    def __init__(
        self,
        bandwidth_gbps: float = 64.0,
        chunk_bytes: int = 65536,
        # Decision latency components
        score_latency_us: float = 0.01,     # PPU scoring: ~10ns
        schedule_latency_us: float = 0.005,  # Budget check: ~5ns
        transfer_latency_us: float = 1.0,    # CXL transfer
        decompress_latency_us: float = 0.5,  # NMD decompression
    ):
        self.bandwidth_gbps = bandwidth_gbps
        self.chunk_bytes = chunk_bytes
        self.score_latency_us = score_latency_us
        self.schedule_latency_us = schedule_latency_us
        self.transfer_latency_us = transfer_latency_us
        self.decompress_latency_us = decompress_latency_us

    @property
    def decision_latency_us(self) -> float:
        """Total decision latency D_decision."""
        return (self.score_latency_us + self.schedule_latency_us +
                self.transfer_latency_us + self.decompress_latency_us)

    # ------------------------------------------------------------------
    # Core BDP computation
    # ------------------------------------------------------------------
    def compute_bdp(
        self,
        bandwidth_gbps: Optional[float] = None,
        decision_latency_us: Optional[float] = None,
    ) -> int:
        """Compute BDP in bytes.

        BDP = B_link × D_decision
        """
        bw = bandwidth_gbps or self.bandwidth_gbps
        lat = decision_latency_us or self.decision_latency_us
        # bw in GB/s = bytes/us × 1e3, lat in us
        bdp = bw * (1024**3) / 1e6 * lat
        return int(math.ceil(bdp))

    def optimal_buffer_depth(
        self,
        bdp_bytes: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
    ) -> int:
        """Compute optimal prefetch buffer depth N_buf*.

        N_buf* = ⌈BDP / chunk_bytes⌉
        """
        bdp = bdp_bytes or self.compute_bdp()
        cb = chunk_bytes or self.chunk_bytes
        return max(1, math.ceil(bdp / cb))

    def exposed_latency(
        self,
        buffer_depth: int,
        bdp_bytes: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
        bandwidth_gbps: Optional[float] = None,
    ) -> float:
        """Compute exposed latency for a given buffer depth.

        L_exposed = max(0, D_decision - N_buf × chunk_bytes / B_link)

        Returns latency in microseconds.
        """
        bdp = bdp_bytes or self.compute_bdp()
        cb = chunk_bytes or self.chunk_bytes
        bw = bandwidth_gbps or self.bandwidth_gbps

        # Time covered by buffer
        buffer_coverage_us = buffer_depth * cb / (bw * 1024**3 / 1e6)
        return max(0.0, self.decision_latency_us - buffer_coverage_us)

    def compute_full_bdp_result(self) -> BDPResult:
        """Compute full BDP result with all components."""
        bdp = self.compute_bdp()
        return BDPResult(
            bandwidth_gbps=self.bandwidth_gbps,
            decision_latency_us=self.decision_latency_us,
            bdp_bytes=bdp,
            chunk_bytes=self.chunk_bytes,
            optimal_buffer_depth=self.optimal_buffer_depth(bdp),
            score_latency_us=self.score_latency_us,
            schedule_latency_us=self.schedule_latency_us,
            transfer_latency_us=self.transfer_latency_us,
            decompress_latency_us=self.decompress_latency_us,
        )

    # ------------------------------------------------------------------
    # Throughput-latency curve (M/D/1 queuing)
    # ------------------------------------------------------------------
    def throughput_latency_curve(
        self,
        service_time_us: Optional[float] = None,
        load_range: Optional[List[float]] = None,
    ) -> List[ThroughputLatencyPoint]:
        """Compute M/D/1 throughput-latency curve.

        Args:
            service_time_us: Per-chunk service time (default: transfer + decompress).
            load_range: List of offered load values ρ ∈ (0, 1).
        """
        S = service_time_us or (self.transfer_latency_us + self.decompress_latency_us)
        if load_range is None:
            load_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                          0.9, 0.92, 0.95, 0.97, 0.99]

        points = []
        for rho in load_range:
            if rho <= 0:
                continue
            lam = rho / S  # arrival rate
            stable = rho < 1.0
            if stable:
                # M/D/1: W = ρ·S / (2·(1-ρ))
                queue_wait = rho * S / (2.0 * (1.0 - rho))
                avg_lat = S + queue_wait
            else:
                queue_wait = float('inf')
                avg_lat = float('inf')

            points.append(ThroughputLatencyPoint(
                offered_load_rho=rho,
                arrival_rate=lam,
                service_time_us=S,
                avg_latency_us=avg_lat,
                queue_wait_us=queue_wait,
                stable=stable,
            ))
        return points

    # ------------------------------------------------------------------
    # Multi-tenant BDP
    # ------------------------------------------------------------------
    def multi_tenant_bdp(
        self,
        num_tenants: int,
        bandwidth_gbps: Optional[float] = None,
        decision_latency_us: Optional[float] = None,
        chunk_bytes: Optional[int] = None,
    ) -> MultiTenantBDPResult:
        """Compute per-tenant and total BDP for K tenants.

        Per-tenant BW = B_total / K
        Per-tenant BDP = (B_total / K) × D_decision
        """
        bw = bandwidth_gbps or self.bandwidth_gbps
        lat = decision_latency_us or self.decision_latency_us
        cb = chunk_bytes or self.chunk_bytes
        K = max(num_tenants, 1)

        per_bw = bw / K
        per_bdp = self.compute_bdp(per_bw, lat)
        per_buf = max(1, math.ceil(per_bdp / cb))
        total_buf = K * per_buf

        # Contention dominated when per-tenant buffer = 1
        total_bdp = self.compute_bdp(bw, lat)
        contention = per_buf <= 1 and K > math.ceil(total_bdp / cb)

        return MultiTenantBDPResult(
            num_tenants=K,
            total_bandwidth_gbps=bw,
            per_tenant_bw_gbps=per_bw,
            per_tenant_bdp_bytes=per_bdp,
            per_tenant_buffer_depth=per_buf,
            total_buffer_depth=total_buf,
            total_buffer_bytes=total_buf * cb,
            contention_dominated=contention,
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------
    def sensitivity_analysis(
        self,
        bandwidth_range: Optional[List[float]] = None,
        chunk_size_range: Optional[List[int]] = None,
        latency_range: Optional[List[float]] = None,
    ) -> List[Dict[str, float]]:
        """Sweep parameters and compute BDP + buffer depth.

        Returns list of dicts with parameter values and results.
        """
        if bandwidth_range is None:
            bandwidth_range = [16.0, 32.0, 64.0, 128.0]
        if chunk_size_range is None:
            chunk_size_range = [16384, 32768, 65536, 131072]
        if latency_range is None:
            latency_range = [1.0, 2.0, 5.0, 10.0]

        results = []
        for bw in bandwidth_range:
            for cs in chunk_size_range:
                for lat in latency_range:
                    bdp = self.compute_bdp(bw, lat)
                    buf = max(1, math.ceil(bdp / cs))
                    results.append({
                        "bandwidth_gbps": bw,
                        "chunk_bytes": cs,
                        "decision_latency_us": lat,
                        "bdp_bytes": bdp,
                        "buffer_depth": buf,
                        "buffer_sram_bytes": buf * cs,
                    })
        return results

    # ------------------------------------------------------------------
    # Hardware sizing recommendation
    # ------------------------------------------------------------------
    def hardware_sizing_recommendation(
        self,
        target_exposed_latency_us: float = 0.0,
    ) -> HardwareSizingRecommendation:
        """Recommend buffer depth to achieve target exposed latency.

        If target = 0, recommends N_buf* (fully hidden).
        """
        bdp = self.compute_bdp()
        bw_bytes_per_us = self.bandwidth_gbps * 1024**3 / 1e6

        if target_exposed_latency_us <= 0:
            depth = self.optimal_buffer_depth(bdp, self.chunk_bytes)
        else:
            # L_exposed = D_decision - N × chunk / B
            # N = (D_decision - L_target) × B / chunk
            needed_coverage = self.decision_latency_us - target_exposed_latency_us
            if needed_coverage <= 0:
                depth = 1
            else:
                depth = max(1, math.ceil(
                    needed_coverage * bw_bytes_per_us / self.chunk_bytes
                ))

        buf_bytes = depth * self.chunk_bytes
        # CACTI 7nm: ~0.0015 mm² per KB for single-port SRAM
        buf_area = buf_bytes / 1024 * 0.0015
        # Power: ~2 mW per KB at 1 GHz
        buf_power = buf_bytes / 1024 * 2.0

        return HardwareSizingRecommendation(
            config_name="custom",
            bandwidth_gbps=self.bandwidth_gbps,
            decision_latency_us=self.decision_latency_us,
            bdp_bytes=bdp,
            recommended_buffer_depth=depth,
            buffer_sram_bytes=buf_bytes,
            buffer_area_mm2=buf_area,
            buffer_power_mw=buf_power,
        )

    @staticmethod
    def precomputed_sizing_table() -> List[HardwareSizingRecommendation]:
        """Pre-computed sizing for common hardware configurations."""
        results = []
        for name, cfg in _HW_CONFIGS.items():
            analyzer = BDPAnalyzer(
                bandwidth_gbps=cfg["bw"],
                chunk_bytes=cfg["chunk"],
            )
            rec = analyzer.hardware_sizing_recommendation()
            rec.config_name = name
            results.append(rec)
        return results
