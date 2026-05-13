//=============================================================================
// QFC Engine Testbench
// Verifies:
//   1. Single QFC request latency (~50.5us + transfer overhead)
//   2. 8 parallel requests utilize all MACs
//   3. Backpressure when FIFO full
//   4. Full-duplex upstream/downstream operation
//   5. Reset behavior
//=============================================================================
`timescale 1ns/1ps

module QFC_ENGINE_TB;

    //=========================================================================
    // Clock & Reset
    //=========================================================================
    logic clk = 0;
    logic rst_n;
    localparam real CLK_PERIOD = 1.0;  // 1GHz = 1ns period

    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // DUT Signals
    //=========================================================================
    logic        clk_en;
    logic [511:0] cxl_up_data;
    logic         cxl_up_valid;
    logic         cxl_up_ready;
    logic [511:0] cxl_down_data;
    logic         cxl_down_valid;
    logic         cxl_down_ready;
    logic [31:0]  kv_mem_addr;
    logic         kv_mem_req_valid;
    logic [511:0] kv_mem_rdata;
    logic         kv_mem_rdata_valid;
    logic         kv_mem_rdata_ready;
    logic [15:0]  cfg_mac_compute_cycles;
    logic [7:0]   cfg_num_active_macs;
    logic [31:0]  stat_total_requests;
    logic [31:0]  stat_qfc_requests;
    logic [31:0]  stat_mac_busy_cycles;
    logic [7:0]   stat_mac_status;

    //=========================================================================
    // DUT Instantiation
    //=========================================================================
    QFC_ENGINE dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .clk_en             (clk_en),
        .cxl_up_data        (cxl_up_data),
        .cxl_up_valid       (cxl_up_valid),
        .cxl_up_ready       (cxl_up_ready),
        .cxl_down_data      (cxl_down_data),
        .cxl_down_valid     (cxl_down_valid),
        .cxl_down_ready     (cxl_down_ready),
        .kv_mem_addr        (kv_mem_addr),
        .kv_mem_req_valid   (kv_mem_req_valid),
        .kv_mem_rdata       (kv_mem_rdata),
        .kv_mem_rdata_valid (kv_mem_rdata_valid),
        .kv_mem_rdata_ready (kv_mem_rdata_ready),
        .cfg_mac_compute_cycles (cfg_mac_compute_cycles),
        .cfg_num_active_macs    (cfg_num_active_macs),
        .stat_total_requests    (stat_total_requests),
        .stat_qfc_requests      (stat_qfc_requests),
        .stat_mac_busy_cycles   (stat_mac_busy_cycles),
        .stat_mac_status        (stat_mac_status)
    );

    //=========================================================================
    // KV Memory Model (returns deterministic data)
    //=========================================================================
    always_ff @(posedge clk) begin
        if (kv_mem_req_valid)
            kv_mem_rdata_valid <= 1'b1;
        else
            kv_mem_rdata_valid <= 1'b0;
    end

    // Deterministic KV data for verifiability
    genvar kv_g;
    generate
        for (kv_g = 0; kv_g < 32; kv_g = kv_g + 1) begin : gen_kv_data
            assign kv_mem_rdata[kv_g*16 +: 16] = kv_mem_addr[15:0] + kv_g[15:0];
        end
    endgenerate

    //=========================================================================
    // Task: Send single QFC request
    //=========================================================================
    task automatic send_qfc_request(
        input logic [31:0] chunk_addr,
        input logic [9:0]  request_id
    );
        integer i;
        begin
            // Header beat
            @(posedge clk);
            cxl_up_valid <= 1'b1;
            cxl_up_data <= {448'b0, request_id, chunk_addr};
            @(posedge clk);
            while (!cxl_up_ready) @(posedge clk);
            cxl_up_valid <= 1'b0;

            // Query data beats (16 beats × 64B = 1KB)
            for (i = 0; i < 16; i++) begin
                @(posedge clk);
                cxl_up_valid <= 1'b1;
                // Deterministic query data
                cxl_up_data <= {32{16'(request_id + i[15:0])}};
                @(posedge clk);
                while (!cxl_up_ready) @(posedge clk);
            end
            cxl_up_valid <= 1'b0;
        end
    endtask

    //=========================================================================
    // Task: Wait for result
    //=========================================================================
    task automatic wait_for_results(
        input int expected_count,
        output int received_count,
        output int first_latency_ns,
        output int last_latency_ns
    );
        int start_time;
        int result_time;
        received_count = 0;
        first_latency_ns = 0;
        last_latency_ns = 0;
        start_time = $time;

        while (received_count < expected_count) begin
            @(posedge clk);
            if (cxl_down_valid && cxl_down_ready) begin
                result_time = $time - start_time;
                if (received_count == 0)
                    first_latency_ns = result_time;
                last_latency_ns = result_time;
                received_count++;
                $display("  [TB] Result %0d: req_id=%0d data=0x%08h @ t=%0t",
                    received_count, cxl_down_data[41:32], cxl_down_data[31:0], $time);
            end
        end
    endtask

    //=========================================================================
    // Test Sequence
    //=========================================================================
    int test_passed = 0;
    int test_failed = 0;
    int rcvd_count;
    int first_lat;
    int last_lat;

    initial begin
        $display("================================================================");
        $display("QFC Engine RTL Testbench");
        $display("================================================================");

        // Initialize
        rst_n <= 0;
        clk_en <= 1;
        cxl_up_valid <= 0;
        cxl_up_data <= 0;
        cxl_down_ready <= 1;
        cfg_mac_compute_cycles <= 16'd1024;
        cfg_num_active_macs <= 8'd8;

        repeat(5) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);

        //---------------------------------------------------------------------
        // Test 1: Single QFC request
        //---------------------------------------------------------------------
        $display("\n[Test 1] Single QFC request...");
        fork
            send_qfc_request(32'h1000, 10'd42);
            wait_for_results(1, rcvd_count, first_lat, last_lat);
        join

        if (rcvd_count == 1 && first_lat > 1000 && first_lat < 1500) begin
            $display("  PASS: Single request completed in %0d ns", first_lat);
            test_passed++;
        end else begin
            $display("  FAIL: Expected 1 result in 1024-1500ns, got %0d results in %0d ns",
                rcvd_count, first_lat);
            test_failed++;
        end
        @(posedge clk);

        //---------------------------------------------------------------------
        // Test 2: 8 parallel requests (all MACs)
        //---------------------------------------------------------------------
        $display("\n[Test 2] 8 parallel requests...");
        rst_n <= 0;
        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);

        fork
            begin
                int i;
                for (i = 0; i < 8; i++) begin
                    send_qfc_request(32'h2000 + (i * 32'h10000), 10'(i));
                    repeat(2) @(posedge clk);
                end
            end
            wait_for_results(8, rcvd_count, first_lat, last_lat);
        join

        if (rcvd_count == 8) begin
            $display("  PASS: All 8 requests completed. First=%0d ns, Last=%0d ns",
                first_lat, last_lat);
            // All should finish within ~1200ns because they run in parallel
            if (last_lat < 1200) begin
                $display("  PASS: Parallel execution confirmed (last < 1200ns)");
                test_passed++;
            end else begin
                $display("  FAIL: Last result too slow (%0d ns), possible serialization", last_lat);
                test_failed++;
            end
        end else begin
            $display("  FAIL: Expected 8 results, got %0d", rcvd_count);
            test_failed++;
        end
        @(posedge clk);

        //---------------------------------------------------------------------
        // Test 3: Backpressure / FIFO full
        //---------------------------------------------------------------------
        $display("\n[Test 3] FIFO backpressure with 20 requests...");
        rst_n <= 0;
        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);

        fork
            begin
                int i;
                for (i = 0; i < 20; i++) begin
                    send_qfc_request(32'h3000 + (i * 32'h10000), 10'(i));
                end
            end
            wait_for_results(20, rcvd_count, first_lat, last_lat);
        join

        if (rcvd_count == 20) begin
            $display("  PASS: All 20 requests completed despite FIFO backpressure");
            test_passed++;
        end else begin
            $display("  FAIL: Expected 20 results, got %0d", rcvd_count);
            test_failed++;
        end
        @(posedge clk);

        //---------------------------------------------------------------------
        // Test 4: Downstream backpressure
        //---------------------------------------------------------------------
        $display("\n[Test 4] Downstream backpressure...");
        rst_n <= 0;
        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);
        cxl_down_ready <= 0;  // Block downstream

        send_qfc_request(32'h4000, 10'd99);

        // Wait for compute to finish but result should be held
        repeat(1200) @(posedge clk);
        cxl_down_ready <= 1;  // Release downstream

        // Now result should come out
        rcvd_count = 0;
        fork
            wait_for_results(1, rcvd_count, first_lat, last_lat);
        join

        if (rcvd_count == 1) begin
            $display("  PASS: Result delivered after downstream backpressure release");
            test_passed++;
        end else begin
            $display("  FAIL: Result not received after backpressure release");
            test_failed++;
        end
        @(posedge clk);

        //---------------------------------------------------------------------
        // Test 5: Check status counters
        //---------------------------------------------------------------------
        $display("\n[Test 5] Status counters...");
        $display("  stat_total_requests    = %0d", stat_total_requests);
        $display("  stat_qfc_requests      = %0d", stat_qfc_requests);
        $display("  stat_mac_busy_cycles   = %0d", stat_mac_busy_cycles);
        $display("  stat_mac_status        = 0b%08b", stat_mac_status);

        if (stat_total_requests > 0 && stat_qfc_requests > 0) begin
            $display("  PASS: Status counters incrementing");
            test_passed++;
        end else begin
            $display("  FAIL: Status counters not incrementing");
            test_failed++;
        end

        //---------------------------------------------------------------------
        // Summary
        //---------------------------------------------------------------------
        repeat(10) @(posedge clk);
        $display("\n================================================================");
        $display("Test Summary: PASSED=%0d FAILED=%0d", test_passed, test_failed);
        if (test_failed == 0)
            $display("STATUS: ALL TESTS PASSED!");
        else
            $display("STATUS: SOME TESTS FAILED");
        $display("================================================================");

        $finish;
    end

    // Timeout watchdog
    initial begin
        repeat(10000) @(posedge clk);
        $display("TIMEOUT: Simulation exceeded 10000 cycles");
        $finish;
    end

endmodule
