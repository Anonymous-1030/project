#!/usr/bin/env python3
"""
QFC Engine Power & Area Analysis
Estimates based on TSMC 4nm N4P ULVT standard cell characterization
"""

import json

#=============================================================================
# TSMC 4nm N4P ULVT Parameters (from public literature)
#=============================================================================
TECH_NODE = "TSMC 4nm N4P ULVT"

# Standard cell parameters (typical values)
DFF_AREA_UM2 = 0.8          # µm² per D flip-flop
NAND2_AREA_UM2 = 0.25       # µm² per 2-input NAND
NOR2_AREA_UM2 = 0.25
INV_AREA_UM2 = 0.12
MUX2_AREA_UM2 = 0.35
ADDER_AREA_UM2 = 2.5        # per bit
MULT_16_AREA_UM2 = 120      # 16×16 multiplier
ICG_AREA_UM2 = 25           # Integrated clock gating cell

# Power parameters (at 1GHz, 0.75V, 25°C)
DFF_POWER_UW = 0.8          # µW per DFF (dynamic)
COMB_POWER_PER_GATE_UW = 0.3  # µW per equivalent gate
MULT_16_POWER_UW = 30       # µW per 16×16 multiplier (typical activity)
ADDER_32_POWER_UW = 30      # µW per 32-bit adder (typical activity)
LEAKAGE_PER_UM_UA = 0.1     # nA per µm gate width (leakage)
VOLTAGE_V = 0.75            # Operating voltage

# GPU die sizes for comparison
GPU_DIES = {
    "NVIDIA RTX 4090 (AD102)": 608,    # mm^2
    "NVIDIA RTX 4080 (AD103)": 379,
    "NVIDIA RTX 4070 (AD104)": 295,
    "AMD RX 7900 XTX (Navi 31)": 531,
    "AMD RX 7800 XT (Navi 32)": 350,
    "Mid-range Target": 300,
}

#=============================================================================
# QFC Engine Component Analysis
#=============================================================================

def analyze_query_buffers():
    """8 MAC arrays × 512-entry × 16-bit query buffers"""
    entries_per_mac = 512
    bits_per_entry = 16
    num_macs = 8
    total_dff = entries_per_mac * bits_per_entry * num_macs
    area = total_dff * DFF_AREA_UM2
    # 18% average activity (aggressive clock gating, not all MACs active)
    power = (total_dff * DFF_POWER_UW * 0.18) / 1000
    return {
        "name": "Query Buffers (8×512×16b)",
        "num_dff": total_dff,
        "area_um2": area,
        "area_mm2": area / 1e6,
        "power_dynamic_mw": power,
        "percentage": 0
    }

def analyze_mac_datapath():
    """32-wide 16×16 MAC partial multiplier units"""
    num_macs = 8
    width = 32
    total_mult = num_macs * width
    area = total_mult * MULT_16_AREA_UM2
    # 30% activity factor during MAC compute windows
    power = (total_mult * MULT_16_POWER_UW * 0.3) / 1000
    return {
        "name": "MAC Datapath (8×32 multipliers)",
        "num_mult": total_mult,
        "area_um2": area,
        "area_mm2": area / 1e6,
        "power_dynamic_mw": power,
        "percentage": 0
    }

def analyze_adder_trees():
    """32→1 reduction adder trees per MAC (31 32-bit adders each)"""
    num_macs = 8
    adders_per_mac = 31
    bits = 32
    area = num_macs * adders_per_mac * bits * ADDER_AREA_UM2
    # 35% toggle activity during reduction
    power = (num_macs * adders_per_mac * ADDER_32_POWER_UW * 0.35) / 1000
    return {
        "name": "Adder Trees (8×31 adders)",
        "area_um2": area,
        "area_mm2": area / 1e6,
        "power_dynamic_mw": power,
        "percentage": 0
    }

def analyze_engine_controller():
    """Request FIFO, round-robin arbiter, CXL interface logic"""
    # 16-deep request FIFO, arbiters, result mux, CXL adapters
    gates = 6000
    area = gates * NAND2_AREA_UM2 + (4 * 32 * DFF_AREA_UM2)  # stats counters
    power = (gates * COMB_POWER_PER_GATE_UW) / 1000
    return {
        "name": "Engine Controller & Stats",
        "num_gates": gates,
        "area_um2": area,
        "area_mm2": area / 1e6,
        "power_dynamic_mw": power,
        "percentage": 0
    }

def analyze_clocking():
    """Clock distribution and gating for 8 MACs + engine"""
    icg_area = 9 * ICG_AREA_UM2  # 8 MACs + engine
    clock_tree_area = 600 * INV_AREA_UM2  # Larger buffer tree for wide design
    area = icg_area + clock_tree_area
    # Clock network power scales with number of flops
    total_flops = (8 * 512 * 16) + 500
    clock_power = (total_flops * DFF_POWER_UW * 0.08) / 1000
    return {
        "name": "Clocking (ICG + Tree)",
        "area_um2": area,
        "area_mm2": area / 1e6,
        "power_dynamic_mw": clock_power,
        "percentage": 0
    }

def analyze_interconnect():
    """Estimate routing/interconnect overhead"""
    base_area = (analyze_query_buffers()["area_um2"] +
                 analyze_mac_datapath()["area_um2"] +
                 analyze_adder_trees()["area_um2"] +
                 analyze_engine_controller()["area_um2"])
    overhead = base_area * 0.12  # 12% overhead
    return {
        "name": "Interconnect (12% overhead)",
        "area_um2": overhead,
        "area_mm2": overhead / 1e6,
        "power_dynamic_mw": 0.8,  # Small dynamic contribution
        "percentage": 0
    }

#=============================================================================
# Main Analysis
#=============================================================================

def main():
    print("=" * 70)
    print("QFC Engine Physical Implementation Analysis")
    print(f"Technology: {TECH_NODE}")
    print("=" * 70)

    # Analyze all components
    components = [
        analyze_query_buffers(),
        analyze_mac_datapath(),
        analyze_adder_trees(),
        analyze_engine_controller(),
        analyze_clocking(),
        analyze_interconnect()
    ]

    # Calculate totals
    total_area_um2 = sum(c["area_um2"] for c in components)
    total_power_dyn_mw = sum(c["power_dynamic_mw"] for c in components)

    # Calculate percentages
    for c in components:
        c["percentage"] = (c["area_um2"] / total_area_um2) * 100

    # Leakage power (static)
    # Scales roughly with total area
    leakage_power_mw = 2.5

    # Print area breakdown
    print("\n### Area Breakdown ###")
    print("-" * 70)
    print(f"{'Component':<35} {'Area (um2)':>12} {'Area (mm2)':>12} {'%':>8}")
    print("-" * 70)

    for c in components:
        print(f"{c['name']:<35} {c['area_um2']:>12.0f} {c['area_mm2']:>12.6f} {c['percentage']:>7.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<35} {total_area_um2:>12.0f} {total_area_um2/1e6:>12.6f} {'100.0%':>8}")
    print("=" * 70)
    print(f"\nTarget Budget: 0.300 mm^2  |  Margin: {(300000 - total_area_um2)/1e6:.6f} mm^2")

    # Power breakdown
    print("\n### Power Analysis @ 1GHz, 0.75V ###")
    print("-" * 70)
    print(f"{'Component':<35} {'Dynamic (mW)':>15}")
    print("-" * 70)

    for c in components:
        print(f"{c['name']:<35} {c['power_dynamic_mw']:>15.2f}")

    print("-" * 70)
    print(f"{'Subtotal (Dynamic)':<35} {total_power_dyn_mw:>15.2f}")
    print(f"{'Leakage (Static)':<35} {leakage_power_mw:>15.2f}")
    print("-" * 70)
    print(f"{'TOTAL POWER':<35} {total_power_dyn_mw + leakage_power_mw:>15.2f} mW")
    print("=" * 70)
    print(f"\nTarget Budget: 25.0 mW  |  Margin: {25.0 - (total_power_dyn_mw + leakage_power_mw):.2f} mW")

    # GPU die comparison
    qfc_area_mm2 = total_area_um2 / 1e6

    print("\n### GPU Die Overhead Comparison ###")
    print("-" * 70)
    print(f"{'GPU':<40} {'Die (mm2)':>12} {'Overhead %':>12}")
    print("-" * 70)

    for gpu, die_size in GPU_DIES.items():
        overhead = (qfc_area_mm2 / die_size) * 100
        marker = " ***" if overhead < 0.1 else ""
        print(f"{gpu:<40} {die_size:>12} {overhead:>11.4f}%{marker}")

    print("-" * 70)
    print(f"\n*** QFC Engine area: {qfc_area_mm2:.6f} mm2")
    print(f"*** Target: <0.3 mm^2 / <25 mW")
    print(f"*** Achievement: {(qfc_area_mm2 / 300) * 100:.4f}% for 300mm2 GPU")
    print("=" * 70)

    # Save results
    results = {
        "technology": TECH_NODE,
        "total_area_um2": total_area_um2,
        "total_area_mm2": qfc_area_mm2,
        "total_power_mw": total_power_dyn_mw + leakage_power_mw,
        "dynamic_power_mw": total_power_dyn_mw,
        "static_power_mw": leakage_power_mw,
        "components": components,
        "gpu_overhead": {gpu: (qfc_area_mm2 / die_size) * 100 for gpu, die_size in GPU_DIES.items()}
    }

    with open("../results/qfc_power_area_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to ../results/qfc_power_area_analysis.json")

if __name__ == "__main__":
    main()
