//=============================================================================
// QFC MAC Array Testbench
//=============================================================================
`timescale 1ns/1ps

module QFC_MAC_ARRAY_TB;

    reg clk = 0;
    reg rst_n;
    always #0.5 clk = ~clk;

    reg        clk_en;
    reg        cmd_valid;
    reg [31:0] cmd_chunk_addr;
    reg [9:0]  cmd_request_id;
    wire       cmd_ready;
    reg [511:0] query_data;
    reg         query_valid;
    reg [3:0]   query_beat;
    wire        query_ready;
    wire [31:0] kv_addr;
    wire        kv_req_valid;
    wire [511:0] kv_rdata;
    reg         kv_rdata_valid;
    wire [31:0] result_data;
    wire [9:0]  result_req_id;
    wire        result_valid;
    reg         result_ready;
    wire        mac_busy;
    wire [31:0] mac_cycle_count;

    QFC_MAC_ARRAY dut (
        .clk(clk), .rst_n(rst_n), .clk_en(clk_en),
        .cmd_valid(cmd_valid), .cmd_chunk_addr(cmd_chunk_addr),
        .cmd_request_id(cmd_request_id), .cmd_ready(cmd_ready),
        .query_data(query_data), .query_valid(query_valid),
        .query_beat(query_beat), .query_ready(query_ready),
        .kv_addr(kv_addr), .kv_req_valid(kv_req_valid),
        .kv_rdata(kv_rdata), .kv_rdata_valid(kv_rdata_valid),
        .result_data(result_data), .result_req_id(result_req_id),
        .result_valid(result_valid), .result_ready(result_ready),
        .mac_busy(mac_busy), .mac_cycle_count(mac_cycle_count)
    );

    always @(posedge clk) begin
        kv_rdata_valid <= kv_req_valid;
    end

    reg row_acc_was_x = 0;
    always @(posedge clk) begin
        if ((dut.row_accumulator ^ dut.row_accumulator) !== 32'b0 && !row_acc_was_x) begin
            row_acc_was_x <= 1;
            $display("T=%0t: row_accumulator first became x at row_idx=%0d dim_idx=%0d adder_l5=%h state=%b", $time, dut.row_idx, dut.dim_idx, dut.adder_tree_l5, dut.state);
        end
    end
    assign kv_rdata[15:0]   = kv_addr[15:0] + 16'd0;
    assign kv_rdata[31:16]  = kv_addr[15:0] + 16'd1;
    assign kv_rdata[47:32]  = kv_addr[15:0] + 16'd2;
    assign kv_rdata[63:48]  = kv_addr[15:0] + 16'd3;
    assign kv_rdata[79:64]  = kv_addr[15:0] + 16'd4;
    assign kv_rdata[95:80]  = kv_addr[15:0] + 16'd5;
    assign kv_rdata[111:96] = kv_addr[15:0] + 16'd6;
    assign kv_rdata[127:112]= kv_addr[15:0] + 16'd7;
    assign kv_rdata[143:128]= kv_addr[15:0] + 16'd8;
    assign kv_rdata[159:144]= kv_addr[15:0] + 16'd9;
    assign kv_rdata[175:160]= kv_addr[15:0] + 16'd10;
    assign kv_rdata[191:176]= kv_addr[15:0] + 16'd11;
    assign kv_rdata[207:192]= kv_addr[15:0] + 16'd12;
    assign kv_rdata[223:208]= kv_addr[15:0] + 16'd13;
    assign kv_rdata[239:224]= kv_addr[15:0] + 16'd14;
    assign kv_rdata[255:240]= kv_addr[15:0] + 16'd15;
    assign kv_rdata[271:256]= kv_addr[15:0] + 16'd16;
    assign kv_rdata[287:272]= kv_addr[15:0] + 16'd17;
    assign kv_rdata[303:288]= kv_addr[15:0] + 16'd18;
    assign kv_rdata[319:304]= kv_addr[15:0] + 16'd19;
    assign kv_rdata[335:320]= kv_addr[15:0] + 16'd20;
    assign kv_rdata[351:336]= kv_addr[15:0] + 16'd21;
    assign kv_rdata[367:352]= kv_addr[15:0] + 16'd22;
    assign kv_rdata[383:368]= kv_addr[15:0] + 16'd23;
    assign kv_rdata[399:384]= kv_addr[15:0] + 16'd24;
    assign kv_rdata[415:400]= kv_addr[15:0] + 16'd25;
    assign kv_rdata[431:416]= kv_addr[15:0] + 16'd26;
    assign kv_rdata[447:432]= kv_addr[15:0] + 16'd27;
    assign kv_rdata[463:448]= kv_addr[15:0] + 16'd28;
    assign kv_rdata[479:464]= kv_addr[15:0] + 16'd29;
    assign kv_rdata[495:480]= kv_addr[15:0] + 16'd30;
    assign kv_rdata[511:496]= kv_addr[15:0] + 16'd31;

    integer cycle;
    initial begin
        rst_n = 0;
        clk_en = 1;
        cmd_valid = 0;
        query_valid = 0;
        result_ready = 1;
        cycle = 0;

        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("T=%0t: Starting MAC test", $time);

        // Send command
        @(negedge clk);
        cmd_chunk_addr = 32'h1000;
        cmd_request_id = 10'd42;
        cmd_valid = 1;
        @(posedge clk);
        @(negedge clk);
        cmd_valid = 0;
        repeat(2) @(posedge clk);
        $display("T=%0t: After cmd: state=%b busy=%b", $time, dut.state, mac_busy);

        // Send 16 query beats
        begin : send_query
            integer i;
            for (i = 0; i < 16; i = i + 1) begin
                @(negedge clk);
                query_data = {32{16'(42 + i[15:0])}};
                query_beat = i[3:0];
                query_valid = 1;
                @(posedge clk);
                @(negedge clk);
                query_valid = 0;
            end
        end
        $display("T=%0t: Query beats sent, qb[0]=%h qb[1]=%h qb[511]=%h", $time, dut.query_buffer[0], dut.query_buffer[1], dut.query_buffer[511]);

        // Wait for query_load_done and then compute
        repeat(1500) begin
            @(posedge clk);
            cycle = cycle + 1;
            if (result_valid) begin
                $display("T=%0t: RESULT! data=0x%08h req_id=%0d", $time, result_data, result_req_id);
                $display("SUCCESS");
                $finish;
            end
        end

        $display("TIMEOUT after %0d cycles (state=%b)", cycle, dut.state);
        $finish;
    end

endmodule
