#!/bin/tclsh
#=============================================================================
# QFC Engine Synthesis Script
# Tool: Synopsys Design Compiler / Cadence Genus
# Target: TSMC 4nm N4P ULVT
#=============================================================================

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------
set TOP_MODULE "QFC_ENGINE"
set RTL_DIR "../rtl"
set RESULT_DIR "../results"
set REPORT_DIR "../results/reports"

# Library paths (adjust for your PDK installation)
set TARGET_LIB "tscs4ulvtbc.db"
set LINK_LIB "* $TARGET_LIB"

#-----------------------------------------------------------------------------
# Read Design
#-----------------------------------------------------------------------------
analyze -format sverilog ${RTL_DIR}/ICG.sv
analyze -format sverilog ${RTL_DIR}/QFC_MAC_ARRAY.sv
analyze -format sverilog ${RTL_DIR}/QFC_ENGINE.sv

elaborate ${TOP_MODULE}
current_design ${TOP_MODULE}
link

#-----------------------------------------------------------------------------
# Constraints
#-----------------------------------------------------------------------------
source QFC_ENGINE.sdc

#-----------------------------------------------------------------------------
# Compile
#-----------------------------------------------------------------------------
# High-effort optimization for area
compile_ultra -area_high_effort_script

# Power optimization
check_power
create_clock_tree
compile_ultra -gate_clock -retime

#-----------------------------------------------------------------------------
# Reports
#-----------------------------------------------------------------------------
file mkdir ${REPORT_DIR}

report_area -hierarchy > ${REPORT_DIR}/area.rpt
report_area -physical > ${REPORT_DIR}/area_physical.rpt
report_power -analysis_effort high > ${REPORT_DIR}/power.rpt
report_power -hierarchy > ${REPORT_DIR}/power_hier.rpt
report_timing -nworst 10 -max_paths 10 > ${REPORT_DIR}/timing.rpt
report_timing -path_type full -delay_type max > ${REPORT_DIR}/timing_max.rpt
report_timing -path_type full -delay_type min > ${REPORT_DIR}/timing_min.rpt
report_cell > ${REPORT_DIR}/cells.rpt
report_design > ${REPORT_DIR}/design.rpt
report_qor > ${REPORT_DIR}/qor.rpt
report_clock_gating > ${REPORT_DIR}/clock_gating.rpt
report_constraints -all_violators > ${REPORT_DIR}/violations.rpt
report_resources > ${REPORT_DIR}/resources.rpt
report_reference > ${REPORT_DIR}/reference.rpt

#-----------------------------------------------------------------------------
# Output
#-----------------------------------------------------------------------------
write -format ddc -hierarchy -output ${RESULT_DIR}/${TOP_MODULE}.ddc
write -format verilog -hierarchy -output ${RESULT_DIR}/${TOP_MODULE}_netlist.v
write_sdf ${RESULT_DIR}/${TOP_MODULE}.sdf
write_sdc ${RESULT_DIR}/${TOP_MODULE}_out.sdc

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------
puts "==================================================================="
puts "Synthesis Complete for ${TOP_MODULE}"
puts "==================================================================="
exec cat ${REPORT_DIR}/area.rpt | tail -20
puts "==================================================================="
exec cat ${REPORT_DIR}/power.rpt | tail -20
puts "==================================================================="
