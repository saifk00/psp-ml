`timescale 1ns/1ps
// One 8 KB VME ring buffer: 2048 x 32-bit words (24-bit samples,
// sign-extended -- the RAM itself stores whatever is written).
//
// Four registered read ports (one per PE -- the emulator gives every PE an
// independent view of every buffer, a superset of the real port structure
// that behaves identically for well-formed contexts), one array write port,
// and one combinational host port for the Media-Engine-side accesses.
//
// The documented "contiguous mirror" after each buffer exists so streams
// can run off the end without wrap handling; its observable behaviour is
// address mod 2048, which is what the 11-bit address gives directly.
module vme_ringbuf (
    input  wire         clk,
    input  wire [43:0]  raddr,    // 4 x 11
    output reg  [127:0] rdata,    // 4 x 32, one cycle after raddr
    input  wire         we,
    input  wire [10:0]  waddr,
    input  wire [31:0]  wdata,
    input  wire         hwe,
    input  wire [10:0]  haddr,
    input  wire [31:0]  hwdata,
    output wire [31:0]  hrdata
);
    reg [31:0] mem [0:2047];
    integer i;
    initial for (i = 0; i < 2048; i = i + 1) mem[i] = 32'd0;

    always @(posedge clk) begin
        for (i = 0; i < 4; i = i + 1)
            rdata[i * 32 +: 32] <= mem[raddr[i * 11 +: 11]];
        if (hwe) mem[haddr] <= hwdata;
        if (we)  mem[waddr] <= wdata;   // array write wins over host
    end

    assign hrdata = mem[haddr];
endmodule
