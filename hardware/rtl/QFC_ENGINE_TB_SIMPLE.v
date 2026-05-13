//=============================================================================
// QFC Engine Simple Debug Testbench
//=============================================================================
`timescale 1ns/1ps

module QFC_ENGINE_TB_SIMPLE;

    reg clk = 0;
    reg rst_n;
    always #0.5 clk = ~clk;

    reg        clk_en;
    reg [511:0] cxl_up_data;
    reg         cxl_up_valid;
    wire        cxl_up_ready;
    wire [511:0] cxl_down_data;
    wire         cxl_down_valid;
    reg          cxl_down_ready;
    wire [31:0]  kv_mem_addr;
    wire         kv_mem_req_valid;
    wire [511:0]  kv_mem_rdata;
    reg          kv_mem_rdata_valid;
    wire         kv_mem_rdata_ready;
    reg [15:0]   cfg_mac_compute_cycles;
    reg [7:0]    cfg_num_active_macs;
    wire [31:0]  stat_total_requests;
    wire [31:0]  stat_qfc_requests;
    wire [31:0]  stat_mac_busy_cycles;
    wire [7:0]   stat_mac_status;

    QFC_ENGINE dut (
        .clk, .rst_n, .clk_en,
        .cxl_up_data, .cxl_up_valid, .cxl_up_ready,
        .cxl_down_data, .cxl_down_valid, .cxl_down_ready,
        .kv_mem_addr, .kv_mem_req_valid,
        .kv_mem_rdata, .kv_mem_rdata_valid, .kv_mem_rdata_ready,
        .cfg_mac_compute_cycles, .cfg_num_active_macs,
        .stat_total_requests, .stat_qfc_requests,
        .stat_mac_busy_cycles, .stat_mac_status
    );

    always @(posedge clk) begin
        kv_mem_rdata_valid <= kv_mem_req_valid;
    end
    genvar kv_g;
    generate
        for (kv_g = 0; kv_g < 32; kv_g = kv_g + 1) begin
            assign kv_mem_rdata[kv_g*16 +: 16] = kv_mem_addr[15:0] + kv_g[15:0];
        end
    endgenerate

    integer cycle;

    task send_header;
        begin
            @(negedge clk);
            cxl_up_data = {448'b0, 10'd42, 32'h1000};
            cxl_up_valid = 1;
            @(posedge clk);
            @(negedge clk);
            cxl_up_valid = 0;
        end
    endtask

    task send_query_beat(input [3:0] beat_num);
        begin
            @(negedge clk);
            cxl_up_data = {512{1'b0}};
            cxl_up_data[15:0] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[31:16] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[47:32] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[63:48] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[79:64] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[95:80] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[111:96] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[127:112] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[143:128] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[159:144] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[175:160] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[191:176] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[207:192] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[223:208] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[239:224] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[255:240] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[271:256] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[287:272] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[303:288] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[319:304] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[335:320] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[351:336] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[367:352] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[383:368] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[399:384] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[415:400] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[431:416] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[447:432] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[463:448] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[479:464] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[495:480] = 16'd42 + {12'b0, beat_num};
            cxl_up_data[511:496] = 16'd42 + {12'b0, beat_num};
            cxl_up_valid = 1;
            @(posedge clk);
            @(negedge clk);
            cxl_up_valid = 0;
        end
    endtask

    initial begin
        rst_n = 0;
        clk_en = 1;
        cxl_up_valid = 0;
        cxl_down_ready = 1;
        cfg_mac_compute_cycles = 16'd1024;
        cfg_num_active_macs = 8'd8;
        cycle = 0;

        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("T=%0t: Starting test", $time);

        send_header;
        repeat(2) @(posedge clk);
        $display("T=%0t: After header: qip=%b qtm=%0d", $time, dut.query_in_progress, dut.query_target_mac);

        if (dut.query_in_progress) begin
            begin : send_query
                integer i;
                for (i = 0; i < 16; i = i + 1) begin
                    send_query_beat(i[3:0]);
                end
            end
            $display("T=%0t: Query beats sent", $time);
        end else begin
            $display("T=%0t: WARNING: query_in_progress not set!", $time);
        end

        repeat(2000) begin
            @(posedge clk);
            cycle = cycle + 1;
            if (cxl_down_valid) begin
                $display("T=%0t: RESULT! data=0x%08h req_id=%0d", $time, cxl_down_data[31:0], cxl_down_data[41:32]);
                $display("SUCCESS");
                $finish;
            end
            if (cycle % 50 == 0) begin
                $display("T=%0t: cycle=%0d mac0_state=%b mac0_cmdcnt=%0d mac0_qbcnt=%0d mac_busy=%b qbc_eng=%0d qvalid0=%b qready0=%b", 
                    $time, cycle, 
                    dut.gen_mac[0].mac_inst.state,
                    dut.gen_mac[0].mac_inst.cmd_fifo_cnt,
                    dut.gen_mac[0].mac_inst.query_beat_cnt,
                    dut.mac_busy_vec[0],
                    dut.query_beat_counter,
                    dut.mac_query_valid[0],
                    dut.gen_mac[0].mac_inst.query_ready);
            end
        end

        $display("TIMEOUT after %0d cycles", cycle);
        $finish;
    end

endmodule
