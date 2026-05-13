//=============================================================================
// PHT Engine Simple Testbench
//=============================================================================
`timescale 1ns/1ps

module PHT_ENGINE_TB_SIMPLE;

    reg clk = 0;
    reg rst_n;
    reg clk_en;
    
    reg [9:0] query_chunk_id;
    reg       query_valid;
    wire [15:0] query_pht_value;
    wire       query_ready;
    
    reg [9:0]  upd_chunk_id;
    reg [15:0] upd_new_value;
    reg        upd_valid;
    reg        upd_is_promoted;
    reg        upd_importance;
    
    reg [9:0]  anchor_chunk_id;
    reg        anchor_set;
    reg        anchor_clear;
    
    wire [31:0] stat_access_count;
    wire [31:0] stat_hit_count;
    wire [31:0] stat_anchor_count;
    
    PHT_ENGINE dut (
        .clk(clk),
        .rst_n(rst_n),
        .clk_en(clk_en),
        .query_chunk_id(query_chunk_id),
        .query_valid(query_valid),
        .query_pht_value(query_pht_value),
        .query_ready(query_ready),
        .upd_chunk_id(upd_chunk_id),
        .upd_new_value(upd_new_value),
        .upd_valid(upd_valid),
        .upd_is_promoted(upd_is_promoted),
        .upd_importance(upd_importance),
        .anchor_chunk_id(anchor_chunk_id),
        .anchor_set(anchor_set),
        .anchor_clear(anchor_clear),
        .stat_access_count(stat_access_count),
        .stat_hit_count(stat_hit_count),
        .stat_anchor_count(stat_anchor_count)
    );
    
    always #0.5 clk = ~clk;
    
    initial begin
        $display("================================================================");
        $display("PHT Engine Simple Verification");
        $display("================================================================");
        
        // Init
        rst_n = 0;
        clk_en = 1;
        query_valid = 0;
        upd_valid = 0;
        anchor_set = 0;
        anchor_clear = 0;
        
        // Reset
        #10;
        rst_n = 1;
        #2;
        
        $display("\nStep 1: Write to Entry 42 (promote, importance=1)");
        @(posedge clk);
        upd_chunk_id = 10'd42;
        upd_valid = 1;
        upd_is_promoted = 1'b1;
        upd_importance = 1'b1;
        @(posedge clk);
        upd_valid = 0;
        
        $display("  Update sent at time %0t", $time);
        
        // Wait for pipeline
        #20;
        
        $display("\nStep 2: Read from Entry 42");
        @(posedge clk);
        query_chunk_id = 10'd42;
        query_valid = 1;
        @(posedge clk);
        query_valid = 0;
        #1;
        $display("  Value read: 0x%04X at time %0t", query_pht_value, $time);
        
        $display("\nStep 3: Write to Entry 43 (promote, importance=0)");
        @(posedge clk);
        upd_chunk_id = 10'd43;
        upd_valid = 1;
        upd_is_promoted = 1'b1;
        upd_importance = 1'b0;
        @(posedge clk);
        upd_valid = 0;
        
        #20;
        
        $display("\nStep 4: Read from Entry 43");
        @(posedge clk);
        query_chunk_id = 10'd43;
        query_valid = 1;
        @(posedge clk);
        query_valid = 0;
        #1;
        $display("  Value read: 0x%04X at time %0t", query_pht_value, $time);
        
        $display("\nStep 5: Decay Entry 42 (not promoted)");
        @(posedge clk);
        upd_chunk_id = 10'd42;
        upd_valid = 1;
        upd_is_promoted = 1'b0;
        upd_importance = 1'b0;
        @(posedge clk);
        upd_valid = 0;
        
        #20;
        
        $display("\nStep 6: Read Entry 42 after decay");
        @(posedge clk);
        query_chunk_id = 10'd42;
        query_valid = 1;
        @(posedge clk);
        query_valid = 0;
        #1;
        $display("  Value read: 0x%04X at time %0t", query_pht_value, $time);
        
        $display("\n================================================================");
        $display("Done");
        $display("================================================================");
        
        $finish;
    end

endmodule
