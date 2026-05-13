#=============================================================================
# QFC Engine Design Constraints
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
#-----------------------------------------------------------------------------
# CXL Upstream (GPU -> CXL)
set_input_delay -clock clk -max 0.10 [get_ports cxl_up_data]
set_input_delay -clock clk -min 0.02 [get_ports cxl_up_data]
set_input_delay -clock clk -max 0.08 [get_ports cxl_up_valid]
set_input_delay -clock clk -min 0.02 [get_ports cxl_up_valid]

# CXL Downstream ready (from GPU)
set_input_delay -clock clk -max 0.08 [get_ports cxl_down_ready]
set_input_delay -clock clk -min 0.02 [get_ports cxl_down_ready]

# KV Memory return data
set_input_delay -clock clk -max 0.10 [get_ports kv_mem_rdata]
set_input_delay -clock clk -min 0.02 [get_ports kv_mem_rdata]
set_input_delay -clock clk -max 0.08 [get_ports kv_mem_rdata_valid]
set_input_delay -clock clk -min 0.02 [get_ports kv_mem_rdata_valid]

# Static configuration (treated as quasi-static)
set_input_delay -clock clk -max 0.20 [get_ports cfg_mac_compute_cycles]
set_input_delay -clock clk -min 0.05 [get_ports cfg_mac_compute_cycles]
set_input_delay -clock clk -max 0.20 [get_ports cfg_num_active_macs]
set_input_delay -clock clk -min 0.05 [get_ports cfg_num_active_macs]

#-----------------------------------------------------------------------------
# Output Delays
#-----------------------------------------------------------------------------
# CXL Upstream ready (to GPU)
set_output_delay -clock clk -max 0.08 [get_ports cxl_up_ready]
set_output_delay -clock clk -min 0.02 [get_ports cxl_up_ready]

# CXL Downstream (CXL -> GPU)
set_output_delay -clock clk -max 0.10 [get_ports cxl_down_data]
set_output_delay -clock clk -min 0.02 [get_ports cxl_down_data]
set_output_delay -clock clk -max 0.08 [get_ports cxl_down_valid]
set_output_delay -clock clk -min 0.02 [get_ports cxl_down_valid]

# KV Memory request
set_output_delay -clock clk -max 0.10 [get_ports kv_mem_addr]
set_output_delay -clock clk -min 0.02 [get_ports kv_mem_addr]
set_output_delay -clock clk -max 0.08 [get_ports kv_mem_req_valid]
set_output_delay -clock clk -min 0.02 [get_ports kv_mem_req_valid]
set_output_delay -clock clk -max 0.08 [get_ports kv_mem_rdata_ready]
set_output_delay -clock clk -min 0.02 [get_ports kv_mem_rdata_ready]

# Statistics (non-timing-critical)
set_output_delay -clock clk -max 0.15 [get_ports stat_total_requests]
set_output_delay -clock clk -max 0.15 [get_ports stat_qfc_requests]
set_output_delay -clock clk -max 0.15 [get_ports stat_mac_busy_cycles]
set_output_delay -clock clk -max 0.15 [get_ports stat_mac_status]

#-----------------------------------------------------------------------------
# Load Capacitance (fF)
#-----------------------------------------------------------------------------
set_load 5.0 [get_ports cxl_up_data]
set_load 2.0 [get_ports cxl_up_ready]
set_load 5.0 [get_ports cxl_down_data]
set_load 2.0 [get_ports cxl_down_valid]
set_load 3.0 [get_ports kv_mem_addr]
set_load 2.0 [get_ports kv_mem_req_valid]
set_load 3.0 [get_ports stat_*]

#-----------------------------------------------------------------------------
# Power Constraints
#-----------------------------------------------------------------------------
set_max_dynamic_power 20e-3   ;# 20 mW
set_max_leakage_power 5e-3    ;# 5 mW
set_max_total_power 25e-3     ;# 25 mW

#-----------------------------------------------------------------------------
# Area Constraint
#-----------------------------------------------------------------------------
set_max_area 300000           ;# 0.3 mm² in µm²

#-----------------------------------------------------------------------------
# Operating Conditions
#-----------------------------------------------------------------------------
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
# Multicycle Paths
#-----------------------------------------------------------------------------
# MAC compute pipeline spans 128 cycles (state-machine controlled)
set_multicycle_path -setup 128 -from [get_pins gen_mac[*].mac_inst.state] -to [get_pins gen_mac[*].mac_inst.compute_result]
set_multicycle_path -hold 127 -from [get_pins gen_mac[*].mac_inst.state] -to [get_pins gen_mac[*].mac_inst.compute_result]

#-----------------------------------------------------------------------------
# False Paths
#-----------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]
set_false_path -to [get_ports stat_*]  ;# Statistics are not timing-critical
set_false_path -from [get_ports cfg_mac_compute_cycles]
set_false_path -from [get_ports cfg_num_active_macs]

#-----------------------------------------------------------------------------
# DRV Constraints (Design Rule Violations)
#-----------------------------------------------------------------------------
set_max_fanout 20 [get_designs QFC_ENGINE]
set_max_transition 0.05 [get_designs QFC_ENGINE]
set_max_capacitance 0.10 [get_designs QFC_ENGINE]
