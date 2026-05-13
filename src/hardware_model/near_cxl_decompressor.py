"""Near-memory decompressor model for CXL-attached KV storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class NearCXLDecompressor:
    """Simple analytical model for compressed KV transfer + decompression."""

    lanes: int = 64
    compressed_bits: int = 4
    output_bits: int = 16
    frequency_ghz: float = 1.2
    efficiency: float = 0.85
    local_buffer_kb: int = 32

    @property
    def compression_ratio(self) -> float:
        return self.output_bits / max(self.compressed_bits, 1)

    def decompression_throughput_gbps(self) -> float:
        bits_per_cycle = self.lanes * self.output_bits
        return bits_per_cycle * self.frequency_ghz * self.efficiency / 8.0

    def transfer_time_us(self, compressed_bytes: int, link_bandwidth_gbps: float) -> float:
        if compressed_bytes <= 0:
            return 0.0
        return (compressed_bytes / (link_bandwidth_gbps * (1024**3))) * 1e6

    def decompression_time_us(self, output_bytes: int) -> float:
        if output_bytes <= 0:
            return 0.0
        return (output_bytes / (self.decompression_throughput_gbps() * (1024**3))) * 1e6

    def model_fetch(self, output_bytes: int, link_bandwidth_gbps: float = 64.0) -> Dict[str, float]:
        compressed_bytes = int(output_bytes / self.compression_ratio)
        transfer_us = self.transfer_time_us(compressed_bytes, link_bandwidth_gbps)
        decompress_us = self.decompression_time_us(output_bytes)
        exposed_us = max(transfer_us, decompress_us)
        return {
            "output_bytes": output_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "link_bandwidth_gbps": link_bandwidth_gbps,
            "transfer_time_us": transfer_us,
            "decompression_time_us": decompress_us,
            "effective_fetch_time_us": exposed_us,
            "saved_link_bytes": output_bytes - compressed_bytes,
        }