//=============================================================================
// Integrated Clock Gating Cell
// TSMC 4nm ULVT compatible implementation
// Area: ~25 um2 per instance
//=============================================================================
`timescale 1ns/1ps

module ICG (
    input  wire CK,    // Clock input
    input  wire E,     // Enable
    input  wire SE,    // Scan enable
    output wire GCK    // Gated clock output
);

    // Latch-based clock gating
    // In physical design, replace with foundry cell:
    // TSMC 4nm: CKLNQD1BWP16P90CPDULVT or similar
    
    reg latch_en;
    
    always @(*) begin
        if (!CK)
            latch_en = E || SE;
    end
    
    assign GCK = CK && latch_en;

endmodule
