//=============================================================================
// PHT Engine with Debug Output
//=============================================================================
`timescale 1ns/1ps

module PHT_ENGINE (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clk_en,
    
    input  wire [9:0]  query_chunk_id,
    input  wire        query_valid,
    output reg  [15:0] query_pht_value,
    output wire        query_ready,
    
    input  wire [9:0]  upd_chunk_id,
    input  wire [15:0] upd_new_value,
    input  wire        upd_valid,
    input  wire        upd_is_promoted,
    input  wire        upd_importance,
    
    input  wire [9:0]  anchor_chunk_id,
    input  wire        anchor_set,
    input  wire        anchor_clear,
    
    output reg  [31:0] stat_access_count,
    output reg  [31:0] stat_hit_count,
    output reg  [31:0] stat_anchor_count
);

    localparam NUM_ENTRIES = 1024;
    localparam EMA_WIDTH = 16;
    localparam FLAGS_WIDTH = 3;
    localparam [EMA_WIDTH-1:0] ANCHOR_FLOOR = 16'h826C;
    
    wire gated_clk;
    
    ICG icg_inst (
        .CK  (clk),
        .E   (clk_en),
        .SE  (1'b0),
        .GCK (gated_clk)
    );
    
    reg anchor_array [0:NUM_ENTRIES-1];
    reg [EMA_WIDTH-1:0] ema_array [0:NUM_ENTRIES-1];
    reg [FLAGS_WIDTH-1:0] flags_array [0:NUM_ENTRIES-1];
    
    integer init_i;
    initial begin
        for (init_i = 0; init_i < NUM_ENTRIES; init_i = init_i + 1) begin
            anchor_array[init_i] = 1'b0;
            ema_array[init_i] = 16'h0000;
            flags_array[init_i] = 3'b000;
        end
    end
    
    // Pipeline Stage 1
    reg [9:0] upd_addr_s1;
    reg       upd_valid_s1;
    reg       upd_is_promoted_s1;
    reg       upd_importance_s1;
    reg       old_anchor_s1;
    reg [15:0] old_ema_s1;
    
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            upd_valid_s1 <= 1'b0;
        end else begin
            upd_addr_s1 <= upd_chunk_id;
            upd_valid_s1 <= upd_valid;
            upd_is_promoted_s1 <= upd_is_promoted;
            upd_importance_s1 <= upd_importance;
            old_anchor_s1 <= anchor_array[upd_chunk_id];
            old_ema_s1 <= ema_array[upd_chunk_id];
        end
    end
    
    // Debug output for Stage 1
    always @(posedge gated_clk) begin
        if (upd_valid_s1)
            $display("  [S1] addr=%0d old_ema=0x%04x promoted=%b importance=%b", 
                upd_addr_s1, old_ema_s1, upd_is_promoted_s1, upd_importance_s1);
    end
    
    // Pipeline Stage 2: Compute
    reg [9:0] upd_addr_s2;
    reg       upd_valid_s2;
    reg       new_anchor_s2;
    reg [15:0] new_ema_s2;
    
    wire [17:0] ema_extended = {2'b00, old_ema_s1};
    wire [19:0] ema_times_4 = ema_extended << 2;
    wire [19:0] importance_val = upd_importance_s1 ? 20'h10000 : 20'h0;
    wire [19:0] ema_with_importance = upd_is_promoted_s1 ? 
        (ema_times_4 + importance_val) : ema_times_4;
    wire [35:0] mult_result = ema_with_importance * 16'h3333;
    wire [17:0] new_ema_raw = mult_result[31:14];
    
    reg [15:0] new_ema_final;
    reg new_anchor_bit;
    
    always @(*) begin
        if (old_anchor_s1 && (new_ema_raw[15:0] < ANCHOR_FLOOR))
            new_ema_final = ANCHOR_FLOOR;
        else
            new_ema_final = new_ema_raw[15:0];
        
        if (anchor_set && (anchor_chunk_id == upd_addr_s1))
            new_anchor_bit = 1'b1;
        else if (anchor_clear && (anchor_chunk_id == upd_addr_s1))
            new_anchor_bit = 1'b0;
        else
            new_anchor_bit = old_anchor_s1;
    end
    
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            upd_valid_s2 <= 1'b0;
        end else begin
            upd_addr_s2 <= upd_addr_s1;
            upd_valid_s2 <= upd_valid_s1;
            new_anchor_s2 <= new_anchor_bit;
            new_ema_s2 <= new_ema_final;
        end
    end
    
    // Debug output for Stage 2
    always @(posedge gated_clk) begin
        if (upd_valid_s1) begin
            $display("  [S2-CALC] ema_ext=0x%05x ema_x4=0x%05x ema_w_imp=0x%05x mult=0x%09x raw=0x%05x final=0x%04x",
                ema_extended, ema_times_4, ema_with_importance, mult_result, new_ema_raw, new_ema_final);
        end
        if (upd_valid_s2)
            $display("  [S2] addr=%0d new_ema=0x%04x", upd_addr_s2, new_ema_s2);
    end
    
    // Pipeline Stage 3: Write back
    always @(posedge gated_clk) begin
        if (upd_valid_s2) begin
            anchor_array[upd_addr_s2] <= new_anchor_s2;
            ema_array[upd_addr_s2] <= new_ema_s2;
            $display("  [S3] WRITE addr=%0d value=0x%04x", upd_addr_s2, new_ema_s2);
        end
        if ((anchor_set || anchor_clear) && !upd_valid && !upd_valid_s1 && !upd_valid_s2) begin
            anchor_array[anchor_chunk_id] <= anchor_set;
        end
    end
    
    // Query Path
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            query_pht_value <= 16'd0;
        end else if (query_valid) begin
            query_pht_value <= ema_array[query_chunk_id];
            $display("  [QUERY] addr=%0d value=0x%04x", query_chunk_id, ema_array[query_chunk_id]);
        end
    end
    
    assign query_ready = 1'b1;
    
    // Statistics
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            stat_access_count <= 32'd0;
            stat_hit_count <= 32'd0;
            stat_anchor_count <= 32'd0;
        end else begin
            if (query_valid)
                stat_access_count <= stat_access_count + 1;
            if (upd_valid)
                stat_hit_count <= stat_hit_count + 1;
            if (anchor_set && !anchor_clear)
                stat_anchor_count <= stat_anchor_count + 1;
            else if (anchor_clear && !anchor_set)
                stat_anchor_count <= stat_anchor_count - 1;
        end
    end

endmodule
