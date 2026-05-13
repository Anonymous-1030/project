//=============================================================================
// PHT (Promotion History Table) Engine
// Purpose: Hardware-accelerated PHT for ProSE policy engine
// Target: TSMC 4nm N4P ULVT @ 1GHz
//=============================================================================
`timescale 1ns/1ps

module PHT_ENGINE (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clk_en,           // Dynamic clock gating enable
    
    // Query Interface (from Streaming Multiprocessors)
    input  wire [9:0]  query_chunk_id,   // Chunk index (0-1023)
    input  wire        query_valid,      // Query request valid
    output reg  [15:0] query_pht_value,  // 16-bit EMA value output
    output wire        query_ready,      // Always ready for query
    
    // Update Interface (from Policy Engine)
    input  wire [9:0]  upd_chunk_id,     // Chunk to update
    input  wire [15:0] upd_new_value,    // New EMA value from policy
    input  wire        upd_valid,        // Update request valid
    input  wire        upd_is_promoted,  // Whether chunk was selected
    input  wire        upd_importance,   // Current importance (0 or 1)
    
    // Anchor Control Interface
    input  wire [9:0]  anchor_chunk_id,
    input  wire        anchor_set,       // Set anchor bit (priority)
    input  wire        anchor_clear,     // Clear anchor bit
    
    // Statistics (for performance monitoring)
    output reg  [31:0] stat_access_count,  // Total query count
    output reg  [31:0] stat_hit_count,     // Total update count
    output reg  [31:0] stat_anchor_count   // Current anchor count
);

    //=========================================================================
    // Parameters & Constants
    //=========================================================================
    localparam NUM_ENTRIES = 1024;
    localparam ADDR_WIDTH = 10;
    localparam EMA_WIDTH = 16;
    localparam FLAGS_WIDTH = 3;
    
    // Fixed-point constants (0.16 format)
    localparam [EMA_WIDTH-1:0] ANCHOR_FLOOR = 16'h826C;  // 0.51
    
    //=========================================================================
    // Clock Gating
    //=========================================================================
    wire gated_clk;
    
    ICG icg_inst (
        .CK  (clk),
        .E   (clk_en),
        .SE  (1'b0),
        .GCK (gated_clk)
    );
    
    //=========================================================================
    // Register File: Split arrays for Verilog compatibility
    //=========================================================================
    reg anchor_array [0:NUM_ENTRIES-1];
    reg [EMA_WIDTH-1:0] ema_array [0:NUM_ENTRIES-1];
    reg [FLAGS_WIDTH-1:0] flags_array [0:NUM_ENTRIES-1];
    
    // Initialize arrays (for simulation)
    integer init_i;
    initial begin
        for (init_i = 0; init_i < NUM_ENTRIES; init_i = init_i + 1) begin
            anchor_array[init_i] = 1'b0;
            ema_array[init_i] = 16'h0000;
            flags_array[init_i] = 3'b000;
        end
    end
    
    //=========================================================================
    // Pipeline Stage 1: Read old value
    //=========================================================================
    reg [ADDR_WIDTH-1:0] upd_addr_s1;
    reg                  upd_valid_s1;
    reg                  upd_is_promoted_s1;
    reg                  upd_importance_s1;
    reg                  old_anchor_s1;
    reg [EMA_WIDTH-1:0]  old_ema_s1;
    reg [FLAGS_WIDTH-1:0] old_flags_s1;
    
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            upd_valid_s1 <= 1'b0;
            upd_addr_s1 <= 10'd0;
            upd_is_promoted_s1 <= 1'b0;
            upd_importance_s1 <= 1'b0;
            old_anchor_s1 <= 1'b0;
            old_ema_s1 <= 16'd0;
            old_flags_s1 <= 3'd0;
        end else begin
            upd_addr_s1 <= upd_chunk_id;
            upd_valid_s1 <= upd_valid;
            upd_is_promoted_s1 <= upd_is_promoted;
            upd_importance_s1 <= upd_importance;
            old_anchor_s1 <= anchor_array[upd_chunk_id];
            old_ema_s1 <= ema_array[upd_chunk_id];
            old_flags_s1 <= flags_array[upd_chunk_id];
        end
    end
    
    //=========================================================================
    // Pipeline Stage 2: Compute new EMA
    //=========================================================================
    reg [ADDR_WIDTH-1:0] upd_addr_s2;
    reg                  upd_valid_s2;
    reg                  new_anchor_s2;
    reg [EMA_WIDTH-1:0]  new_ema_s2;
    reg [FLAGS_WIDTH-1:0] new_flags_s2;
    
    // EMA computation
    wire [17:0] ema_extended = {2'b00, old_ema_s1};
    wire [19:0] ema_times_4 = ema_extended << 2;
    wire [19:0] importance_val = upd_importance_s1 ? 20'h10000 : 20'h0;
    wire [19:0] ema_with_importance = upd_is_promoted_s1 ? 
        (ema_times_4 + importance_val) : ema_times_4;
    wire [35:0] mult_result = ema_with_importance * 16'h3333;
    wire [19:0] new_ema_raw = mult_result[35:16];  // Divide by 2^16 (= multiply by 0.2)
    
    reg [15:0] new_ema_final;
    reg new_anchor_bit;
    
    always @(*) begin
        // Anchor floor enforcement
        if (old_anchor_s1 && (new_ema_raw[15:0] < ANCHOR_FLOOR))
            new_ema_final = ANCHOR_FLOOR;
        else
            new_ema_final = new_ema_raw[15:0];
        
        // Anchor bit update
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
            upd_addr_s2 <= 10'd0;
            new_anchor_s2 <= 1'b0;
            new_ema_s2 <= 16'd0;
            new_flags_s2 <= 3'd0;
        end else begin
            upd_addr_s2 <= upd_addr_s1;
            upd_valid_s2 <= upd_valid_s1;
            new_anchor_s2 <= new_anchor_bit;
            new_ema_s2 <= new_ema_final;
            new_flags_s2 <= old_flags_s1;
        end
    end
    
    //=========================================================================
    // Pipeline Stage 3: Write back to register file
    //=========================================================================
    always @(posedge gated_clk) begin
        if (upd_valid_s2) begin
            anchor_array[upd_addr_s2] <= new_anchor_s2;
            ema_array[upd_addr_s2] <= new_ema_s2;
            flags_array[upd_addr_s2] <= new_flags_s2;
        end
        
        // Anchor update path
        if ((anchor_set || anchor_clear) && !upd_valid && !upd_valid_s1 && !upd_valid_s2) begin
            anchor_array[anchor_chunk_id] <= anchor_set;
        end
    end
    
    //=========================================================================
    // Query Path: Single cycle read
    //=========================================================================
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            query_pht_value <= 16'd0;
        end else if (query_valid) begin
            query_pht_value <= ema_array[query_chunk_id];
        end
    end
    
    assign query_ready = 1'b1;
    
    //=========================================================================
    // Statistics Counters
    //=========================================================================
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            stat_access_count <= 32'd0;
            stat_hit_count <= 32'd0;
        end else begin
            if (query_valid)
                stat_access_count <= stat_access_count + 1;
            if (upd_valid)
                stat_hit_count <= stat_hit_count + 1;
        end
    end
    
    always @(posedge gated_clk or negedge rst_n) begin
        if (!rst_n) begin
            stat_anchor_count <= 32'd0;
        end else begin
            if (anchor_set && !anchor_clear)
                stat_anchor_count <= stat_anchor_count + 1;
            else if (anchor_clear && !anchor_set)
                stat_anchor_count <= stat_anchor_count - 1;
        end
    end

endmodule
