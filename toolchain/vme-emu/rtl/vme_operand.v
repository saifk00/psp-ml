`timescale 1ns/1ps
// One operand path of a functional unit: the 16-way source mux of section
// 5.2 (8 buffers + 8 staging taps, selected by a 5-bit FSEL/BSEL field)
// followed by the ICN_SKEW forwarded-stream delay of section 8.6.
//
// FSEL/BSEL encoding: [4] C (0 buffer / 1 staging), buffers at [3:1]
// (0-3 TOP_0..3, 4-7 BASE_0..3), staging taps at [3:0].  Buffer read data
// arrives on an 8-lane bus indexed 0-3 BASE, 4-7 TOP (Table 2.2 order);
// this module converts between the two index spaces.
//
// The ICN_SKEW 3-bit codes are not cycle counts (Table 8.3):
//   000 -> 0, 100 -> 1, 101 -> 2, 110 -> 3, 111 -> 4 cycles.
// The top-bank code delays top-bank reads, the base-bank code base-bank
// reads; staging taps are never delayed here (their alignment is the
// producer's write/read skew ladder).
module vme_operand #(
    parameter LANE = 0   // which PE this operand path belongs to
) (
    input  wire         clk,
    input  wire [4:0]   sel,
    input  wire [255:0] buf_rdata,   // 8 x 32, internal index 0-3 BASE, 4-7 TOP
    input  wire         top_rv,
    input  wire         base_rv,
    input  wire [255:0] stg_data,    // 8 x 32 staging taps
    input  wire [7:0]   stg_valid,
    input  wire [2:0]   tcode,       // ICN_SKEW top-bank code for this PE
    input  wire [2:0]   bcode,       // ICN_SKEW base-bank code
    output wire [31:0]  data,
    output wire         valid
);
    wire       is_stg  = sel[4];
    wire [3:0] tap     = sel[3:0];
    wire [2:0] bi      = sel[3:1];               // selector index space
    wire       is_base = bi[2];
    // Base-bank reads have own-lane affinity: PE n reads BASE_n no matter
    // what the selector index says -- measured on silicon (BSEL naming
    // BASE_1 from PE0 does not deliver BASE_1; the manual's full-crossbar
    // claim is wrong for the base bank).  Top-bank reads select by index.
    wire [2:0] ibuf    = is_base ? {1'b0, LANE[1:0]} : {1'b1, bi[1:0]};

    wire [31:0] raw  = is_stg ? stg_data[tap * 32 +: 32]
                              : buf_rdata[ibuf * 32 +: 32];
    wire        rawv = is_stg ? stg_valid[tap]
                              : (is_base ? base_rv : top_rv);

    function [2:0] f_dec;
        input [2:0] c;
        case (c)
            3'b000:  f_dec = 3'd0;
            3'b100:  f_dec = 3'd1;
            3'b101:  f_dec = 3'd2;
            3'b110:  f_dec = 3'd3;
            3'b111:  f_dec = 3'd4;
            default: f_dec = 3'd0;
        endcase
    endfunction

    // Buffer reads take BUF_EXTRA more cycles than the RAM register alone:
    // silicon calibration split the 6-cycle addr-to-capture path as
    // read = 3, FU = 3.  Staging taps bypass this.
    localparam BUF_EXTRA = 3'd2;
    wire [3:0] dly = is_stg ? 4'd0
                   : {1'b0, BUF_EXTRA} + {1'b0, (is_base ? f_dec(bcode) : f_dec(tcode))};

    reg [32:0] pipe [0:6];
    integer i;
    initial for (i = 0; i < 7; i = i + 1) pipe[i] = 33'd0;
    always @(posedge clk) begin
        pipe[0] <= {rawv, raw};
        for (i = 6; i > 0; i = i - 1) pipe[i] <= pipe[i - 1];
    end

    wire [32:0] outp = (dly == 4'd0) ? {rawv, raw} : pipe[dly - 4'd1];
    assign data  = outp[31:0];
    assign valid = outp[32];
endmodule
