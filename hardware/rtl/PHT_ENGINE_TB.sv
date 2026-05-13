//=============================================================================
// PHT Engine Testbench
// Validates single-cycle query and 3-cycle update pipeline
//=============================================================================
`timescale 1ns/1ps

module PHT_ENGINE_TB;

    //=========================================================================
    // Signals
    //=========================================================================
    logic clk = 0;
    logic rst_n;
    logic clk_en;
    
    // Query interface
    logic [9:0] query_chunk_id;
    logic       query_valid;
    logic [15:0] query_pht_value;
    logic       query_ready;
    
    // Update interface
    logic [9:0]  upd_chunk_id;
    logic [15:0] upd_new_value;
    logic        upd_valid;
    logic        upd_is_promoted;
    logic        upd_importance;
    
    // Anchor control
    logic [9:0]  anchor_chunk_id;
    logic        anchor_set;
    logic        anchor_clear;
    
    // Statistics
    logic [31:0] stat_access_count;
    logic [31:0] stat_hit_count;
    logic [31:0] stat_anchor_count;
    
    //=========================================================================
    // DUT
    //=========================================================================
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
    
    //=========================================================================
    // Clock Generation (1 GHz = 1ns period)
    //=========================================================================
    always #0.5 clk = ~clk;
    
    //=========================================================================
    // Test Stimulus
    //=========================================================================
    initial begin
        $display("================================================================");
        $display("PHT Engine Testbench");
        $display("================================================================");
        
        // Initialize
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
        
        // Test 1: Basic write then read
        $display("\n[Test 1] Basic Write/Read");
        write_pht(10'd42, 1'b1, 1'b1);  // Promote with importance=1
        #1;
        read_pht(10'd42);
        check_result(16'hA666, "EMA after promote+importance");  // ~0.65
        
        // Test 2: EMA decay (not promoted)
        $display("\n[Test 2] EMA Decay");
        repeat(3) begin
            write_pht(10'd42, 1'b0, 1'b0);  // Not promoted
            #3;
        end
        read_pht(10'd42);
        // After 3 decays, value should be lower
        $display("  Decayed EMA: 0x%04X", query_pht_value);
        
        // Test 3: Anchor functionality
        $display("\n[Test 3] Anchor Lock");
        set_anchor(10'd100);
        #1;
        write_pht(10'd100, 1'b0, 1'b0);  // Try to decay
        #3;
        read_pht(10'd100);
        check_result(16'h826C, "Anchor floor value");  // 0.51
        
        // Test 4: Multiple entries
        $display("\n[Test 4] Multiple Entries");
        for (int i = 0; i < 10; i++) begin
            write_pht(i[9:0], 1'b1, i[0]);  // Alternate importance
            #1;
        end
        
        // Read back all
        for (int i = 0; i < 10; i++) begin
            read_pht(i[9:0]);
            #1;
            $display("  Entry %0d: 0x%04X", i, query_pht_value);
        end
        
        // Test 5: Clock gating
        $display("\n[Test 5] Clock Gating");
        clk_en = 0;
        #10;
        clk_en = 1;
        #2;
        read_pht(10'd42);
        $display("  Read after clock gating: 0x%04X", query_pht_value);
        
        // Final statistics
        #10;
        $display("\n================================================================");
        $display("Final Statistics:");
        $display("  Access Count: %0d", stat_access_count);
        $display("  Update Count: %0d", stat_hit_count);
        $display("  Anchor Count: %0d", stat_anchor_count);
        $display("================================================================");
        $display("All tests completed!");
        $display("================================================================");
        
        $finish;
    end
    
    //=========================================================================
    // Tasks
    //=========================================================================
    task write_pht(input [9:0] cid, input promoted, input importance);
        begin
            @(posedge clk);
            upd_chunk_id = cid;
            upd_valid = 1;
            upd_is_promoted = promoted;
            upd_importance = importance;
            @(posedge clk);
            upd_valid = 0;
        end
    endtask
    
    task read_pht(input [9:0] cid);
        begin
            @(posedge clk);
            query_chunk_id = cid;
            query_valid = 1;
            @(posedge clk);
            query_valid = 0;
        end
    endtask
    
    task set_anchor(input [9:0] cid);
        begin
            @(posedge clk);
            anchor_chunk_id = cid;
            anchor_set = 1;
            @(posedge clk);
            anchor_set = 0;
        end
    endtask
    
    task check_result(input [15:0] expected, input string msg);
        begin
            #0.1;  // Wait for output to settle
            if (query_pht_value === expected) begin
                $display("  [PASS] %s: Expected 0x%04X, Got 0x%04X", 
                    msg, expected, query_pht_value);
            end else begin
                $display("  [FAIL] %s: Expected 0x%04X, Got 0x%04X", 
                    msg, expected, query_pht_value);
            end
        end
    endtask
    
    //=========================================================================
    // Waveform Dump
    //=========================================================================
    initial begin
        $dumpfile("pht_engine_tb.vcd");
        $dumpvars(0, PHT_ENGINE_TB);
    end

endmodule
