//=============================================================================
// QFC MAC Array
// Purpose: Single parallel MAC array for Query-Forwarding Compute Engine
//=============================================================================
`timescale 1ns/1ps

module QFC_MAC_ARRAY (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clk_en,

    input  wire        cmd_valid,
    input  wire [31:0] cmd_chunk_addr,
    input  wire [9:0]  cmd_request_id,
    output wire        cmd_ready,

    input  wire [511:0] query_data,
    input  wire         query_valid,
    input  wire [3:0]   query_beat,
    output wire         query_ready,

    output wire [31:0]  kv_addr,
    output wire         kv_req_valid,
    input  wire [511:0] kv_rdata,
    input  wire         kv_rdata_valid,

    output wire [31:0]  result_data,
    output wire [9:0]   result_req_id,
    output wire         result_valid,
    input  wire         result_ready,

    output wire         mac_busy,
    output reg  [31:0]  mac_cycle_count
);

    localparam MAC_WIDTH = 32;
    localparam QUERY_BEATS = 16;
    localparam CHUNK_ROWS = 64;
    localparam ROW_CYCLES = 16;
    localparam COMPUTE_CYCLES = CHUNK_ROWS * ROW_CYCLES;

    //=========================================================================
    // Clock gating (bypassed for iverilog compatibility)
    //=========================================================================
    wire gated_clk = clk;

    //=========================================================================
    // Command FIFO (depth 2) - using separate arrays for iverilog compatibility
    //=========================================================================
    reg [31:0] cmd_fifo_addr [0:1];
    reg [9:0]  cmd_fifo_id   [0:1];
    reg [1:0]  cmd_fifo_cnt;
    reg        cmd_fifo_wr;
    reg        cmd_fifo_rd;

    assign cmd_ready = (cmd_fifo_cnt < 2);
    assign cmd_fifo_wr = cmd_valid && cmd_ready;

    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_fifo_cnt <= 2'b0;
        end else begin
            if (cmd_fifo_wr && !cmd_fifo_rd)
                cmd_fifo_cnt <= cmd_fifo_cnt + 1;
            else if (!cmd_fifo_wr && cmd_fifo_rd)
                cmd_fifo_cnt <= cmd_fifo_cnt - 1;

            if (cmd_fifo_wr) begin
                if (cmd_fifo_cnt == 0 || (cmd_fifo_cnt == 1 && !cmd_fifo_rd)) begin
                    cmd_fifo_addr[0] <= cmd_chunk_addr;
                    cmd_fifo_id[0]   <= cmd_request_id;
                end else begin
                    cmd_fifo_addr[1] <= cmd_chunk_addr;
                    cmd_fifo_id[1]   <= cmd_request_id;
                end
            end
        end
    end

    //=========================================================================
    // State Machine
    //=========================================================================
    localparam [2:0] MAC_IDLE = 3'd0;
    localparam [2:0] MAC_LOAD_QUERY = 3'd1;
    localparam [2:0] MAC_COMPUTE = 3'd2;
    localparam [2:0] MAC_OUTPUT = 3'd3;

    reg [2:0] state;
    reg [2:0] next_state;

    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n)
            state <= MAC_IDLE;
        else
            state <= next_state;
    end

    //=========================================================================
    // Query Vector Buffer (512 x 16-bit)
    //=========================================================================
    reg [15:0] query_buffer [0:511];
    reg [4:0]  query_beat_cnt;
    wire       query_load_done;

    assign query_ready = (state == MAC_IDLE) || (state == MAC_LOAD_QUERY);

    always @(posedge gated_clk) begin
        if (query_valid && query_ready) begin
            query_buffer[{query_beat, 5'd0}]  <= query_data[15:0];
            query_buffer[{query_beat, 5'd1}]  <= query_data[31:16];
            query_buffer[{query_beat, 5'd2}]  <= query_data[47:32];
            query_buffer[{query_beat, 5'd3}]  <= query_data[63:48];
            query_buffer[{query_beat, 5'd4}]  <= query_data[79:64];
            query_buffer[{query_beat, 5'd5}]  <= query_data[95:80];
            query_buffer[{query_beat, 5'd6}]  <= query_data[111:96];
            query_buffer[{query_beat, 5'd7}]  <= query_data[127:112];
            query_buffer[{query_beat, 5'd8}]  <= query_data[143:128];
            query_buffer[{query_beat, 5'd9}]  <= query_data[159:144];
            query_buffer[{query_beat, 5'd10}] <= query_data[175:160];
            query_buffer[{query_beat, 5'd11}] <= query_data[191:176];
            query_buffer[{query_beat, 5'd12}] <= query_data[207:192];
            query_buffer[{query_beat, 5'd13}] <= query_data[223:208];
            query_buffer[{query_beat, 5'd14}] <= query_data[239:224];
            query_buffer[{query_beat, 5'd15}] <= query_data[255:240];
            query_buffer[{query_beat, 5'd16}] <= query_data[271:256];
            query_buffer[{query_beat, 5'd17}] <= query_data[287:272];
            query_buffer[{query_beat, 5'd18}] <= query_data[303:288];
            query_buffer[{query_beat, 5'd19}] <= query_data[319:304];
            query_buffer[{query_beat, 5'd20}] <= query_data[335:320];
            query_buffer[{query_beat, 5'd21}] <= query_data[351:336];
            query_buffer[{query_beat, 5'd22}] <= query_data[367:352];
            query_buffer[{query_beat, 5'd23}] <= query_data[383:368];
            query_buffer[{query_beat, 5'd24}] <= query_data[399:384];
            query_buffer[{query_beat, 5'd25}] <= query_data[415:400];
            query_buffer[{query_beat, 5'd26}] <= query_data[431:416];
            query_buffer[{query_beat, 5'd27}] <= query_data[447:432];
            query_buffer[{query_beat, 5'd28}] <= query_data[463:448];
            query_buffer[{query_beat, 5'd29}] <= query_data[479:464];
            query_buffer[{query_beat, 5'd30}] <= query_data[495:480];
            query_buffer[{query_beat, 5'd31}] <= query_data[511:496];
        end
    end

    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            query_beat_cnt <= 4'b0;
        end else if (query_valid && query_ready) begin
            if (query_beat == 4'd15)
                query_beat_cnt <= 5'd16;
            else if (query_beat_cnt < 5'd16)
                query_beat_cnt <= query_beat + 1;
        end else if (cmd_fifo_rd) begin
            query_beat_cnt <= 4'b0;
        end
    end

    assign query_load_done = (query_beat_cnt == 5'd16);

    //=========================================================================
    // Active command
    //=========================================================================
    reg [31:0] active_addr;
    reg [9:0]  active_id;

    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            active_addr <= 32'b0;
            active_id <= 10'b0;
        end else if (cmd_fifo_rd) begin
            active_addr <= cmd_fifo_addr[0];
            active_id <= cmd_fifo_id[0];
        end
    end

    //=========================================================================
    // Compute Datapath
    //=========================================================================
    reg [15:0] compute_cycle_cnt;
    reg [31:0] compute_result;
    wire       compute_done;

    reg [5:0]  row_idx;
    reg [4:0]  dim_idx;
    reg [31:0] row_accumulator;

    wire [31:0] mac_partial [0:MAC_WIDTH-1];
    wire [31:0] adder_tree_l1 [0:15];
    wire [31:0] adder_tree_l2 [0:7];
    wire [31:0] adder_tree_l3 [0:3];
    wire [31:0] adder_tree_l4 [0:1];
    wire [31:0] adder_tree_l5;

    wire [15:0] query_elem_vec [0:MAC_WIDTH-1];
    wire [15:0] kv_elem_vec [0:MAC_WIDTH-1];
    wire [8:0]  query_idx_vec [0:MAC_WIDTH-1];

    genvar g;
    generate
        for (g = 0; g < MAC_WIDTH; g = g + 1) begin : gen_mac_unit
            wire [31:0] query_ext;
            wire [31:0] kv_ext;

            assign query_idx_vec[g] = {row_idx, dim_idx, g[4:0]};
            assign query_elem_vec[g] = query_buffer[query_idx_vec[g]];
            assign kv_elem_vec[g]    = kv_rdata[g*16 + 15 : g*16];
            assign query_ext  = {16'b0, query_elem_vec[g]};
            assign kv_ext     = {16'b0, kv_elem_vec[g]};
            assign mac_partial[g] = query_ext * kv_ext;
        end
    endgenerate

    generate
        for (g = 0; g < 16; g = g + 1) begin : gen_l1
            assign adder_tree_l1[g] = mac_partial[g*2] + mac_partial[g*2 + 1];
        end
        for (g = 0; g < 8; g = g + 1) begin : gen_l2
            assign adder_tree_l2[g] = adder_tree_l1[g*2] + adder_tree_l1[g*2 + 1];
        end
        for (g = 0; g < 4; g = g + 1) begin : gen_l3
            assign adder_tree_l3[g] = adder_tree_l2[g*2] + adder_tree_l2[g*2 + 1];
        end
        for (g = 0; g < 2; g = g + 1) begin : gen_l4
            assign adder_tree_l4[g] = adder_tree_l3[g*2] + adder_tree_l3[g*2 + 1];
        end
        assign adder_tree_l5 = adder_tree_l4[0] + adder_tree_l4[1];
    endgenerate

    // State transitions
    always @(*) begin
        next_state = state;
        cmd_fifo_rd = 1'b0;
        case (state)
            MAC_IDLE: begin
                if (cmd_fifo_cnt > 0 && query_load_done) begin
                    next_state = MAC_COMPUTE;
                    cmd_fifo_rd = 1'b1;
                end else if (cmd_fifo_cnt > 0 && !query_load_done) begin
                    next_state = MAC_LOAD_QUERY;
                    cmd_fifo_rd = 1'b1;
                end
            end

            MAC_LOAD_QUERY: begin
                if (query_load_done)
                    next_state = MAC_COMPUTE;
            end

            MAC_COMPUTE: begin
                if (compute_done)
                    next_state = MAC_OUTPUT;
            end

            MAC_OUTPUT: begin
                if (result_ready)
                    next_state = MAC_IDLE;
            end

            default: next_state = MAC_IDLE;
        endcase
    end

    assign compute_done = (compute_cycle_cnt == COMPUTE_CYCLES - 1);

    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_cycle_cnt <= 16'b0;
            row_idx <= 6'b0;
            dim_idx <= 5'b0;
            row_accumulator <= 32'b0;
            compute_result <= 32'b0;
        end else begin
            case (state)
                MAC_COMPUTE: begin
                    compute_cycle_cnt <= compute_cycle_cnt + 1;

                    if (dim_idx == ROW_CYCLES - 1) begin
                        dim_idx <= 5'b0;
                        row_idx <= row_idx + 1;
                        row_accumulator <= row_accumulator + adder_tree_l5;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end

                    if (compute_done)
                        compute_result <= row_accumulator + adder_tree_l5;
                end

                default: begin
                    compute_cycle_cnt <= 16'b0;
                    row_idx <= 6'b0;
                    dim_idx <= 5'b0;
                    row_accumulator <= 32'b0;
                end
            endcase
        end
    end

    assign kv_addr = active_addr + {row_idx, dim_idx, 6'b0};
    assign kv_req_valid = (state == MAC_COMPUTE);

    assign result_data = compute_result;
    assign result_req_id = active_id;
    assign result_valid = (state == MAC_OUTPUT);

    assign mac_busy = (state != MAC_IDLE);

    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n)
            mac_cycle_count <= 32'b0;
        else if (mac_busy)
            mac_cycle_count <= mac_cycle_count + 1;
    end

endmodule
