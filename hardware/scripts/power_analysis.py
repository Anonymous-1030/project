#!/usr/bin/env python3
"""
PHT Engine Power & Area Analysis
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
LEAKAGE_PER_UM_UA = 0.1     # nA per µm gate width (leakage)
VOLTAGE_V = 0.75            # Operating voltage

# GPU die sizes for comparison
GPU_DIES = {
    "NVIDIA RTX 4090 (AD102)": 608,    # mm²
    "NVIDIA RTX 4080 (AD103)": 379,
    "NVIDIA RTX 4070 (AD104)": 295,
    "AMD RX 7900 XTX (Navi 31)": 531,
    "AMD RX 7800 XT (Navi 32)": 350,
    "Mid-range Target": 300,
}

#=============================================================================
# PHT Engine Component Analysis
#=============================================================================

def analyze_register_file():
    """Analyze 1024-entry × 20-bit register file"""
    entries = 1024
    bits_per_entry = 20
    total_bits = entries * bits_per_entry
    
    # Each bit = 1 DFF
    num_dff = total_bits
    area = num_dff * DFF_AREA_UM2
    
    # Decoder: 10-to-1024, roughly 1000 gates
    decoder_area = 1000 * NAND2_AREA_UM2
    
    # Output MUX: 1024-to-1, 20-bit wide
    # Roughly log2(1024) × 20 × MUX2 area
    mux_area = 10 * 20 * MUX2_AREA_UM2
    
    total_area = area + decoder_area + mux_area
    
    power_dynamic = num_dff * DFF_POWER_UW  # Switching all DFFs
    
    return {
        "name": "Register File (1024×20)",
        "num_dff": num_dff,
        "area_um2": total_area,
        "area_mm2": total_area / 1e6,
        "power_dynamic_mw": power_dynamic / 1000,
        "percentage": 0
    }

def analyze_ema_unit():
    """Analyze EMA computation unit"""
    # Components:
    # - 18-bit shifter (×4)
    # - 20-bit adder (importance)
    # - 20×16 multiplier
    # - Comparator (anchor floor)
    # - Pipeline registers
    
    shifter_area = 20 * 2  # Minimal area
    adder_area = 20 * ADDER_AREA_UM2
    multiplier_area = MULT_16_AREA_UM2 * 1.5  # 20×16 approx
    comparator_area = 16 * 4  # 16-bit compare
    pipeline_dff = (10 + 1 + 1 + 1 + 20 + 20) * DFF_AREA_UM2
    
    total_area = shifter_area + adder_area + multiplier_area + comparator_area + pipeline_dff
    num_gates = 2500  # Estimated equivalent gates
    
    return {
        "name": "EMA Computation Unit",
        "num_gates": num_gates,
        "area_um2": total_area,
        "area_mm2": total_area / 1e6,
        "power_dynamic_mw": (num_gates * COMB_POWER_PER_GATE_UW) / 1000,
        "percentage": 0
    }

def analyze_control():
    """Control logic and statistics counters"""
    # 3 × 32-bit counters
    counter_dff = 3 * 32 * DFF_AREA_UM2
    counter_logic = 3 * 100 * NAND2_AREA_UM2  # Counter logic
    
    # State machine + control
    control_area = 500 * NAND2_AREA_UM2
    
    total_area = counter_dff + counter_logic + control_area
    num_gates = 800
    
    return {
        "name": "Control & Statistics",
        "num_gates": num_gates,
        "area_um2": total_area,
        "area_mm2": total_area / 1e6,
        "power_dynamic_mw": (num_gates * COMB_POWER_PER_GATE_UW) / 1000,
        "percentage": 0
    }

def analyze_clocking():
    """Clock distribution and gating"""
    # 1 ICG cell
    # Clock tree (estimated)
    icg_area = ICG_AREA_UM2
    clock_tree_area = 100 * INV_AREA_UM2  # Buffer tree
    
    total_area = icg_area + clock_tree_area
    
    # Clock power (major consumer)
    # 1024 entries × 20 bits = 20480 DFFs to clock
    clock_power = 20480 * DFF_POWER_UW * 0.3  # 30% activity
    
    return {
        "name": "Clocking (ICG + Tree)",
        "area_um2": total_area,
        "area_mm2": total_area / 1e6,
        "power_dynamic_mw": clock_power / 1000,
        "percentage": 0
    }

def analyze_interconnect():
    """Estimate routing/interconnect overhead"""
    # Typically 10-15% of logic area
    base_area = (analyze_register_file()["area_um2"] + 
                 analyze_ema_unit()["area_um2"])
    overhead = base_area * 0.12  # 12% overhead
    
    return {
        "name": "Interconnect (12% overhead)",
        "area_um2": overhead,
        "area_mm2": overhead / 1e6,
        "power_dynamic_mw": 0.5,  # Small dynamic contribution
        "percentage": 0
    }

#=============================================================================
# Main Analysis
#=============================================================================

def main():
    print("=" * 70)
    print("PHT Engine Physical Implementation Analysis")
    print(f"Technology: {TECH_NODE}")
    print("=" * 70)
    
    # Analyze all components
    components = [
        analyze_register_file(),
        analyze_ema_unit(),
        analyze_control(),
        analyze_clocking(),
        analyze_interconnect()
    ]
    
    # Calculate totals
    total_area_um2 = sum(c["area_um2"] for c in components)
    total_power_dyn_mw = sum(c["power_dynamic_mw"] for c in components)
    
    # Calculate percentages
    for c in components:
        c["percentage"] = (c["area_um2"] / total_area_um2) * 100
    
    # Add leakage power (static)
    # Leakage: 100 nA per µm × 10,000 µm total gate width × 0.75V = 0.75 mW
    leakage_power_mw = 2.1
    
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
    
    # GPU die comparison
    pht_area_mm2 = total_area_um2 / 1e6
    
    print("\n### GPU Die Overhead Comparison ###")
    print("-" * 70)
    print(f"{'GPU':<40} {'Die (mm2)':>12} {'Overhead %':>12}")
    print("-" * 70)
    
    for gpu, die_size in GPU_DIES.items():
        overhead = (pht_area_mm2 / die_size) * 100
        marker = " ***" if overhead < 0.1 else ""
        print(f"{gpu:<40} {die_size:>12} {overhead:>11.4f}%{marker}")
    
    print("-" * 70)
    print(f"\n*** PHT Engine area: {pht_area_mm2:.6f} mm2")
    print(f"*** Target: < 0.1% die overhead")
    print(f"*** Achievement: {(pht_area_mm2 / 300) * 100:.4f}% for 300mm2 GPU")
    print("=" * 70)
    
    # Save results
    results = {
        "technology": TECH_NODE,
        "total_area_um2": total_area_um2,
        "total_area_mm2": pht_area_mm2,
        "total_power_mw": total_power_dyn_mw + leakage_power_mw,
        "dynamic_power_mw": total_power_dyn_mw,
        "static_power_mw": leakage_power_mw,
        "components": components,
        "gpu_overhead": {gpu: (pht_area_mm2 / die_size) * 100 for gpu, die_size in GPU_DIES.items()}
    }
    
    with open("../results/power_area_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to ../results/power_area_analysis.json")

if __name__ == "__main__":
    main()
