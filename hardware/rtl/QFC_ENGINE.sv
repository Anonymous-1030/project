//=============================================================================
// QFC (Query-Forwarding Compute) Engine
// Simplified robust version for iverilog compatibility
//=============================================================================
`timescale 1ns/1ps

module QFC_ENGINE (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clk_en,

    input  wire [511:0] cxl_up_data,
    input  wire         cxl_up_valid,
    output wire         cxl_up_ready,

    output wire [511:0] cxl_down_data,
    output wire         cxl_down_valid,
    input  wire         cxl_down_ready,

    output wire [31:0]  kv_mem_addr,
    output wire         kv_mem_req_valid,
    input  wire [511:0] kv_mem_rdata,
    input  wire         kv_mem_rdata_valid,
    output wire         kv_mem_rdata_ready,

    input  wire [15:0]  cfg_mac_compute_cycles,
    input  wire [7:0]   cfg_num_active_macs,
    output reg  [31:0]  stat_total_requests,
    output reg  [31:0]  stat_qfc_requests,
    output reg  [31:0]  stat_mac_busy_cycles,
    output wire [7:0]   stat_mac_status
);

    localparam NUM_MACS = 8;

    //=========================================================================
    // Clock gating (bypassed for iverilog compatibility)
    //=========================================================================
    wire gated_clk = clk;

    //=========================================================================
    // Request slot allocation
    // Each MAC has a pending command slot
    //=========================================================================
    reg [31:0] cmd_slot_addr [0:NUM_MACS-1];
    reg [9:0]  cmd_slot_id   [0:NUM_MACS-1];
    reg        cmd_slot_valid [0:NUM_MACS-1];
    wire [NUM_MACS-1:0] mac_cmd_ready;

    // Find first available slot
    integer alloc_i;
    reg [2:0] alloc_mac;
    reg alloc_found;
    always @(*) begin
        alloc_found = 1'b0;
        alloc_mac = 3'd0;
        for (alloc_i = 0; alloc_i < NUM_MACS; alloc_i = alloc_i + 1) begin
            if (!alloc_found && !cmd_slot_valid[alloc_i]) begin
                alloc_mac = alloc_i[2:0];
                alloc_found = 1'b1;
            end
        end
    end

    //=========================================================================
    // Query tracking
    //=========================================================================
    reg [2:0] query_target_mac;
    reg [3:0] query_beat_counter;
    reg       query_in_progress;
    reg       accepting_query;

    assign cxl_up_ready = !query_in_progress || accepting_query;

    //=========================================================================
    // Main control FSM
    //=========================================================================
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_RECV_QUERY = 2'd1;
    reg [1:0] ctrl_state;

    integer rst_i;
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state <= ST_IDLE;
            query_in_progress <= 1'b0;
            query_target_mac <= 3'd0;
            query_beat_counter <= 4'b0;
            accepting_query <= 1'b0;
            for (rst_i = 0; rst_i < NUM_MACS; rst_i = rst_i + 1)
                cmd_slot_valid[rst_i] <= 1'b0;
        end else begin
            case (ctrl_state)
                ST_IDLE: begin
                    if (cxl_up_valid && alloc_found) begin
                        // Accept header
                        cmd_slot_addr[alloc_mac] <= cxl_up_data[31:0];
                        cmd_slot_id[alloc_mac]   <= cxl_up_data[41:32];
                        query_target_mac <= alloc_mac;
                        query_in_progress <= 1'b1;
                        query_beat_counter <= 4'b0;
                        accepting_query <= 1'b1;
                        ctrl_state <= ST_RECV_QUERY;
                    end
                end

                ST_RECV_QUERY: begin
                    if (cxl_up_valid && cxl_up_ready) begin
                        if (query_beat_counter == 4'd15) begin
                            accepting_query <= 1'b0;
                            ctrl_state <= ST_IDLE;
                        end else begin
                            query_beat_counter <= query_beat_counter + 1;
                        end
                    end else if (!cxl_up_valid && query_beat_counter == 4'd0 && !accepting_query) begin
                        // Safety: if no query data arrived, stay idle
                        ctrl_state <= ST_IDLE;
                    end
                end

                default: ctrl_state <= ST_IDLE;
            endcase
        end
    end

    //=========================================================================
    // MAC Array Instantiation
    //=========================================================================
    wire [31:0]  mac_result_data [0:NUM_MACS-1];
    wire [9:0]   mac_result_req_id [0:NUM_MACS-1];
    wire [NUM_MACS-1:0] mac_result_valid;
    wire [NUM_MACS-1:0] mac_result_ready;
    wire [NUM_MACS-1:0] mac_busy_vec;
    wire [31:0]  mac_cycle_count [0:NUM_MACS-1];
    wire [31:0]  mac_kv_addr [0:NUM_MACS-1];
    wire [NUM_MACS-1:0] mac_kv_req_valid;

    wire [NUM_MACS-1:0] mac_query_valid;
    wire [511:0]        mac_query_data [0:NUM_MACS-1];
    wire [3:0]          mac_query_beat [0:NUM_MACS-1];

    assign mac_query_valid[0] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd0) && cxl_up_valid && cxl_up_ready;
    assign mac_query_valid[1] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd1) && cxl_up_valid && cxl_up_ready;
    assign mac_query_valid[2] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd2) && cxl_up_valid && cxl_up_ready;
    assign mac_query_valid[3] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd3) && cxl_up_valid && cxl_up_ready;
    assign mac_query_valid[4] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd4) && cxl_up_valid && cxl_up_ready;
    assign mac_query_valid[5] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd5) && cxl_up_valid && cxl_up_ready;
    assign mac_query_valid[6] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd6) && cxl_up_valid && cxl_up_ready;
    assign mac_query_valid[7] = (ctrl_state == ST_RECV_QUERY) && (query_target_mac == 3'd7) && cxl_up_valid && cxl_up_ready;

    genvar g;
    generate
        for (g = 0; g < NUM_MACS; g = g + 1) begin : gen_mac
            assign mac_query_data[g]  = cxl_up_data;
            assign mac_query_beat[g]  = query_beat_counter;

            QFC_MAC_ARRAY mac_inst (
                .clk            (clk),
                .rst_n          (rst_n),
                .clk_en         (clk_en),
                .cmd_valid      (cmd_slot_valid[g]),
                .cmd_chunk_addr (cmd_slot_addr[g]),
                .cmd_request_id (cmd_slot_id[g]),
                .cmd_ready      (mac_cmd_ready[g]),
                .query_data     (mac_query_data[g]),
                .query_valid    (mac_query_valid[g]),
                .query_beat     (mac_query_beat[g]),
                .query_ready    (),
                .kv_addr        (mac_kv_addr[g]),
                .kv_req_valid   (mac_kv_req_valid[g]),
                .kv_rdata       (kv_mem_rdata),
                .kv_rdata_valid (kv_mem_rdata_valid && mac_busy_vec[g]),
                .result_data    (mac_result_data[g]),
                .result_req_id  (mac_result_req_id[g]),
                .result_valid   (mac_result_valid[g]),
                .result_ready   (mac_result_ready[g]),
                .mac_busy       (mac_busy_vec[g]),
                .mac_cycle_count(mac_cycle_count[g])
            );
        end
    endgenerate

    // cmd_slot_valid is a single-cycle pulse
    reg [NUM_MACS-1:0] cmd_slot_alloc_pulse;
    always @(posedge gated_clk) begin
        cmd_slot_alloc_pulse <= {NUM_MACS{1'b0}};
        if (ctrl_state == ST_IDLE && cxl_up_valid && alloc_found)
            cmd_slot_alloc_pulse[alloc_mac] <= 1'b1;
    end

    integer clear_i;
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            for (clear_i = 0; clear_i < NUM_MACS; clear_i = clear_i + 1)
                cmd_slot_valid[clear_i] <= 1'b0;
        end else begin
            for (clear_i = 0; clear_i < NUM_MACS; clear_i = clear_i + 1)
                cmd_slot_valid[clear_i] <= cmd_slot_alloc_pulse[clear_i];
        end
    end

    //=========================================================================
    // KV memory arbitration (priority)
    //=========================================================================
    assign kv_mem_addr =
        mac_kv_req_valid[0] ? mac_kv_addr[0] :
        mac_kv_req_valid[1] ? mac_kv_addr[1] :
        mac_kv_req_valid[2] ? mac_kv_addr[2] :
        mac_kv_req_valid[3] ? mac_kv_addr[3] :
        mac_kv_req_valid[4] ? mac_kv_addr[4] :
        mac_kv_req_valid[5] ? mac_kv_addr[5] :
        mac_kv_req_valid[6] ? mac_kv_addr[6] :
        mac_kv_req_valid[7] ? mac_kv_addr[7] : 32'b0;
    assign kv_mem_req_valid = |mac_kv_req_valid;
    assign kv_mem_rdata_ready = 1'b1;

    //=========================================================================
    // Result arbitration
    //=========================================================================
    reg [2:0] result_arbiter;
    reg [NUM_MACS-1:0] result_grant;
    integer r_i;
    reg r_found;

    always @(*) begin
        result_grant = {NUM_MACS{1'b0}};
        r_found = 1'b0;
        for (r_i = 0; r_i < NUM_MACS; r_i = r_i + 1) begin
            if (!r_found && mac_result_valid[(r_i + result_arbiter) % NUM_MACS]) begin
                result_grant[(r_i + result_arbiter) % NUM_MACS] = 1'b1;
                r_found = 1'b1;
            end
        end
    end

    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n)
            result_arbiter <= 3'd0;
        else if (cxl_down_valid && cxl_down_ready)
            result_arbiter <= result_arbiter + 1;
    end

    reg [31:0] selected_result;
    reg [9:0]  selected_req_id;
    integer sel_i;
    always @(*) begin
        selected_result = 32'b0;
        selected_req_id = 10'b0;
        for (sel_i = 0; sel_i < NUM_MACS; sel_i = sel_i + 1) begin
            if (result_grant[sel_i]) begin
                selected_result = mac_result_data[sel_i];
                selected_req_id = mac_result_req_id[sel_i];
            end
        end
    end

    assign cxl_down_data = {448'b0, selected_req_id, selected_result};
    assign cxl_down_valid = (result_grant != {NUM_MACS{1'b0}});

    generate
        for (g = 0; g < NUM_MACS; g = g + 1) begin : gen_result_ready
            assign mac_result_ready[g] = result_grant[g] && cxl_down_ready;
        end
    endgenerate

    //=========================================================================
    // Statistics
    //=========================================================================
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            stat_total_requests <= 32'b0;
            stat_qfc_requests <= 32'b0;
            stat_mac_busy_cycles <= 32'b0;
        end else begin
            if (cxl_up_valid && cxl_up_ready && ctrl_state == ST_IDLE && alloc_found) begin
                stat_total_requests <= stat_total_requests + 1;
                stat_qfc_requests <= stat_qfc_requests + 1;
            end
            if (|mac_busy_vec)
                stat_mac_busy_cycles <= stat_mac_busy_cycles + 1;
        end
    end

    assign stat_mac_status = mac_busy_vec;

endmodule
