//=============================================================================
// Integrated Clock Gating Cell
// TSMC 4nm ULVT compatible implementation
// Area: ~25 µm² per instance
//=============================================================================
`timescale 1ns/1ps

module ICG (
    input  logic CK,   // Clock input
    input  logic E,    // Enable
    input  logic SE,   // Scan enable
    output logic GCK   // Gated clock output
);

    // Latch-based clock gating
    // In physical design, replace with foundry cell:
    // TSMC 4nm: CKLNQD1BWP16P90CPDULVT or similar
    
    logic latch_en;
    
    always_latch begin
        if (!CK) begin
            latch_en = E || SE;
        end
    end
    
    assign GCK = CK && latch_en;

endmodule
