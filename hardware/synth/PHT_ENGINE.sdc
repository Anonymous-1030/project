#=============================================================================
# PHT Engine Design Constraints
# Technology: TSMC 4nm N4P Ultra-Low Threshold Voltage (ULVT)
# Library: tscs4ulvt (TSMC Standard Cell Library)
#=============================================================================

#-----------------------------------------------------------------------------
# Clock Definition
#-----------------------------------------------------------------------------
create_clock -name clk -period 1.0 [get_ports clk]

set_clock_uncertainty -setup 0.05 [get_clocks clk]
set_clock_uncertainty -hold 0.02 [get_clocks clk]
set_clock_transition 0.02 [get_clocks clk]

#-----------------------------------------------------------------------------
# Input Delays (relative to clock)
# Assumes driving cells are from same library
#-----------------------------------------------------------------------------
set_input_delay -clock clk -max 0.08 [get_ports query_chunk_id]
set_input_delay -clock clk -min 0.02 [get_ports query_chunk_id]
set_input_delay -clock clk -max 0.08 [get_ports query_valid]
set_input_delay -clock clk -min 0.02 [get_ports query_valid]

set_input_delay -clock clk -max 0.10 [get_ports upd_chunk_id]
set_input_delay -clock clk -min 0.02 [get_ports upd_chunk_id]
set_input_delay -clock clk -max 0.10 [get_ports upd_new_value]
set_input_delay -clock clk -min 0.02 [get_ports upd_new_value]
set_input_delay -clock clk -max 0.08 [get_ports upd_valid]
set_input_delay -clock clk -min 0.02 [get_ports upd_valid]

set_input_delay -clock clk -max 0.08 [get_ports anchor_chunk_id]
set_input_delay -clock clk -min 0.02 [get_ports anchor_chunk_id]
set_input_delay -clock clk -max 0.05 [get_ports anchor_set]
set_input_delay -clock clk -min 0.01 [get_ports anchor_set]
set_input_delay -clock clk -max 0.05 [get_ports anchor_clear]
set_input_delay -clock clk -min 0.01 [get_ports anchor_clear]

#-----------------------------------------------------------------------------
# Output Delays
#-----------------------------------------------------------------------------
set_output_delay -clock clk -max 0.08 [get_ports query_pht_value]
set_output_delay -clock clk -min 0.02 [get_ports query_pht_value]
set_output_delay -clock clk -max 0.05 [get_ports query_ready]
set_output_delay -clock clk -min 0.01 [get_ports query_ready]

set_output_delay -clock clk -max 0.08 [get_ports stat_access_count]
set_output_delay -clock clk -max 0.08 [get_ports stat_hit_count]
set_output_delay -clock clk -max 0.08 [get_ports stat_anchor_count]

#-----------------------------------------------------------------------------
# Load Capacitance (fF)
#-----------------------------------------------------------------------------
set_load 4.0 [get_ports query_pht_value]
set_load 2.0 [get_ports query_ready]
set_load 3.0 [get_ports stat_*]

#-----------------------------------------------------------------------------
# Power Constraints
#-----------------------------------------------------------------------------
set_max_dynamic_power 15e-3   ;# 15 mW
set_max_leakage_power 5e-3    ;# 5 mW
set_max_total_power 20e-3     ;# 20 mW

#-----------------------------------------------------------------------------
# Area Constraint
#-----------------------------------------------------------------------------
set_max_area 65000            ;# 0.065 mm² in µm²

#-----------------------------------------------------------------------------
# Operating Conditions
#-----------------------------------------------------------------------------
# SS: Slow-Slow corner (worst timing)
# FF: Fast-Fast corner (best timing, worst power)
# TT: Typical-Typical (nominal)
set_operating_conditions -analysis_type bc_wc

#-----------------------------------------------------------------------------
# Wire Load Model (for pre-layout)
#-----------------------------------------------------------------------------
set_wire_load_model -name "Zero" -library tscs4ulvt

#-----------------------------------------------------------------------------
# Clock Gating Setup
#-----------------------------------------------------------------------------
set_clock_gating_style -max_fanout 32 -positive_edge_logic {latch}
set_clock_gating_check -setup 0.2 -hold 0.1

#-----------------------------------------------------------------------------
# Multicycle Paths (EMA computation pipeline)
#-----------------------------------------------------------------------------
# EMA computation spans 3 cycles
set_multicycle_path -setup 3 -from [get_pins pht_regfile[*].ema] -to [get_pins new_entry_s2.ema]
set_multicycle_path -hold 2 -from [get_pins pht_regfile[*].ema] -to [get_pins new_entry_s2.ema]

#-----------------------------------------------------------------------------
# False Paths
#-----------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]
set_false_path -to [get_ports stat_*]  ;# Statistics are not timing-critical

#-----------------------------------------------------------------------------
# DRV Constraints (Design Rule Violations)
#-----------------------------------------------------------------------------
set_max_fanout 20 [get_designs PHT_ENGINE]
set_max_transition 0.05 [get_designs PHT_ENGINE]
set_max_capacitance 0.10 [get_designs PHT_ENGINE]
