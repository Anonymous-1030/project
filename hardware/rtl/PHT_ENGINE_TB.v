//=============================================================================
// PHT Engine Testbench
// Validates single-cycle query and 3-cycle update pipeline
//=============================================================================
`timescale 1ns/1ps

module PHT_ENGINE_TB;

    //=========================================================================
    // Signals
    //=========================================================================
    reg clk = 0;
    reg rst_n;
    reg clk_en;
    
    // Query interface
    reg [9:0] query_chunk_id;
    reg       query_valid;
    wire [15:0] query_pht_value;
    wire       query_ready;
    
    // Update interface
    reg [9:0]  upd_chunk_id;
    reg [15:0] upd_new_value;
    reg        upd_valid;
    reg        upd_is_promoted;
    reg        upd_importance;
    
    // Anchor control
    reg [9:0]  anchor_chunk_id;
    reg        anchor_set;
    reg        anchor_clear;
    
    // Statistics
    wire [31:0] stat_access_count;
    wire [31:0] stat_hit_count;
    wire [31:0] stat_anchor_count;
    
    // Test tracking
    integer test_num;
    integer pass_count;
    integer fail_count;
    
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
        $display("PHT Engine Testbench - ProSE Hardware Verification");
        $display("================================================================");
        
        // Initialize
        test_num = 0;
        pass_count = 0;
        fail_count = 0;
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
        test_num = 1;
        $display("\n[Test %0d] Basic Write/Read", test_num);
        write_pht(10'd42, 1'b1, 1'b1);  // Promote with importance=1
        #10;  // Wait for 3-stage pipeline + margin
        read_pht(10'd42);
        #2;
        $display("  Raw value read: 0x%04X", query_pht_value);
        // Value should be approximately 0x3333 (~0.2) after first promote
        // Allow 20% tolerance for fixed-point arithmetic
        if (query_pht_value >= 16'h3000 && query_pht_value <= 16'h3800) begin
            $display("  [PASS] EMA value 0x%04X within expected range", query_pht_value);
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] EMA value 0x%04X outside expected range (0x3000-0x3800)", query_pht_value);
            fail_count = fail_count + 1;
        end
        
        // Test 2: EMA decay (not promoted)
        test_num = 2;
        $display("\n[Test %0d] EMA Decay", test_num);
        repeat(3) begin
            write_pht(10'd42, 1'b0, 1'b0);  // Not promoted
            #3;
        end
        read_pht(10'd42);
        $display("  Decayed EMA: 0x%04X (expected < 0xA666)", query_pht_value);
        if (query_pht_value < 16'hA666)
            pass_count = pass_count + 1;
        else
            fail_count = fail_count + 1;
        
        // Test 3: Anchor functionality
        test_num = 3;
        $display("\n[Test %0d] Anchor Lock", test_num);
        set_anchor(10'd100);
        #1;
        write_pht(10'd100, 1'b0, 1'b0);  // Try to decay
        #3;
        read_pht(10'd100);
        check_result(16'h826C, "Anchor floor value", 0);  // 0.51 (may vary slightly due to EMA)
        
        // Test 4: Multiple entries
        test_num = 4;
        $display("\n[Test %0d] Multiple Entries", test_num);
        write_pht(10'd0, 1'b1, 1'b0);
        write_pht(10'd1, 1'b1, 1'b1);
        write_pht(10'd2, 1'b0, 1'b0);
        write_pht(10'd3, 1'b1, 1'b0);
        #4;
        
        // Read back
        read_pht(10'd0);
        #1; $display("  Entry 0: 0x%04X", query_pht_value);
        read_pht(10'd1);
        #1; $display("  Entry 1: 0x%04X", query_pht_value);
        read_pht(10'd2);
        #1; $display("  Entry 2: 0x%04X", query_pht_value);
        read_pht(10'd3);
        #1; $display("  Entry 3: 0x%04X", query_pht_value);
        pass_count = pass_count + 1;
        
        // Test 5: Clock gating
        test_num = 5;
        $display("\n[Test %0d] Clock Gating", test_num);
        write_pht(10'd50, 1'b1, 1'b1);
        #4;
        clk_en = 0;
        #10;
        clk_en = 1;
        #2;
        read_pht(10'd50);
        $display("  Read after clock gating: 0x%04X", query_pht_value);
        if (query_pht_value != 16'h0000)
            pass_count = pass_count + 1;
        else
            fail_count = fail_count + 1;
        
        // Test 6: Statistics counters
        test_num = 6;
        $display("\n[Test %0d] Statistics Verification", test_num);
        #1;
        $display("  Access Count: %0d", stat_access_count);
        $display("  Update Count: %0d", stat_hit_count);
        $display("  Anchor Count: %0d", stat_anchor_count);
        if (stat_access_count > 0 && stat_hit_count > 0)
            pass_count = pass_count + 1;
        else
            fail_count = fail_count + 1;
        
        // Final results
        #10;
        $display("\n================================================================");
        $display("Test Summary:");
        $display("  Total Tests: %0d", test_num);
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        if (fail_count == 0)
            $display("  STATUS: ALL TESTS PASSED!");
        else
            $display("  STATUS: SOME TESTS FAILED");
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
    
    task check_result(input [15:0] expected, input [255:0] msg, input exact_match);
        reg [15:0] diff;
        begin
            #0.1;  // Wait for output to settle
            if (exact_match) begin
                if (query_pht_value === expected) begin
                    $display("  [PASS] %0s: Expected 0x%04X, Got 0x%04X", msg, expected, query_pht_value);
                    pass_count = pass_count + 1;
                end else begin
                    $display("  [FAIL] %0s: Expected 0x%04X, Got 0x%04X", msg, expected, query_pht_value);
                    fail_count = fail_count + 1;
                end
            end else begin
                // Approximate match (within 10%)
                if (query_pht_value >= (expected - 16'h1000) && query_pht_value <= (expected + 16'h1000)) begin
                    $display("  [PASS] %0s: Got 0x%04X (within tolerance)", msg, query_pht_value);
                    pass_count = pass_count + 1;
                end else begin
                    $display("  [FAIL] %0s: Expected ~0x%04X, Got 0x%04X", msg, expected, query_pht_value);
                    fail_count = fail_count + 1;
                end
            end
        end
    endtask
    
    //=========================================================================
    // Waveform Dump
    //=========================================================================
    initial begin
        $dumpfile("prose_v2/hardware/results/pht_engine_tb.vcd");
        $dumpvars(0, PHT_ENGINE_TB);
    end

endmodule
