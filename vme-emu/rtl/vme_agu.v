`timescale 1ns/1ps
// VME address generation unit -- one of twelve (3 per PE: RTOP, RBASE, WR).
//
// Consumes the six-word register group of docs/vme-reference.html section 7:
//   +0 MODE   [31] E, [30:24] MODE, [23:16] SKEW, [15:0] OFFSET
//   +1 COUNT  [31:16] STEP, [15:0] COUNT (N-1)
//   +2 INNER0 [31:16] CFG, [15:0] SEGMENT (len-1)
//   +3 INNER1 [31:16] CFG, [15:0] STRIDE
//   +4 FMT0   [28] REV, [21]+[16] RP1/RP0, [17] RNG, [15:0] DRAIN
//   +5 FMT1   [29] BRV, [21] END, [3:0] BRVW
//
// After `trigger` the unit idles for SKEW cycles, then emits one element
// offset per cycle for COUNT+1 elements, then goes done.  Offsets are
// computed in 16-bit wrapping arithmetic (so 0x1_0000 - x is a negative
// start offset, per section 7.2) and truncated to 11 bits -- which is
// exactly the observable behaviour of the documented buffer mirror.
//
// Linear mode (MODE 0x04):     off[n] = OFFSET + t(n)*STEP
// Segmented   (MODE 0x02, or FMT0.RNG):
//                              off[n] = OFFSET + j*STEP + w*STRIDE
//   where j counts 0..seglen-1 and reloads (counter B), w counts reloads.
//   seglen = SEGMENT+1, doubled when INNER0.CFG = 0x0003 (the reading that
//   makes Table 7.5's "0 1 2 3 ..." example come out; see README).
// t(n) is the FMT transform: identity, REV (COUNT-n), replicate (0), or
// bit-reversal over BRVW bits.  Transforms apply to linear mode.
module vme_agu (
    input  wire        clk,
    input  wire        rst,
    input  wire        trigger,
    input  wire [31:0] w_mode,
    input  wire [31:0] w_count,
    input  wire [31:0] w_inner0,
    input  wire [31:0] w_inner1,
    input  wire [31:0] w_fmt0,
    input  wire [31:0] w_fmt1,
    output reg  [10:0] addr,
    output reg         addr_valid,
    output reg         done
);
    wire        en     = w_mode[31];
    wire [6:0]  mode   = w_mode[30:24];
    wire [7:0]  skew   = w_mode[23:16];
    wire [15:0] offs   = w_mode[15:0];
    wire [15:0] step   = w_count[31:16];
    wire [15:0] count  = w_count[15:0];
    wire [15:0] cfg0   = w_inner0[31:16];
    wire [15:0] segf   = w_inner0[15:0];
    wire [15:0] stride = w_inner1[15:0];
    wire        rev    = w_fmt0[28];
    wire        rep    = w_fmt0[21] & w_fmt0[16];
    wire        rng    = w_fmt0[17];
    wire        brv    = w_fmt1[29];
    wire [3:0]  brvw   = w_fmt1[3:0];

    wire        segmented = (mode == 7'h02) | rng;
    wire [16:0] seg_len   = (cfg0 == 16'h0003) ? (({1'b0, segf} + 17'd1) << 1)
                                               :  ({1'b0, segf} + 17'd1);

    localparam S_IDLE = 2'd0, S_HOLD = 2'd1, S_RUN = 2'd2, S_DONE = 2'd3;
    reg [1:0]  state;
    reg [7:0]  hold_cnt;
    reg [16:0] n;                 // element index, 0..COUNT
    reg [16:0] j, w;              // counter B position / reload count

    function [15:0] f_bitrev;
        input [15:0] x;
        input [3:0]  wdt;
        integer bi;
        begin
            f_bitrev = x;
            for (bi = 0; bi < 16; bi = bi + 1)
                if (bi < wdt)
                    f_bitrev[bi] = x[wdt - 1 - bi];
        end
    endfunction

    wire [15:0] n_eff = rep ? 16'd0 :
                        rev ? (count - n[15:0]) :
                        brv ? f_bitrev(n[15:0], brvw) : n[15:0];

    wire [15:0] off16 = segmented
        ? (offs + j[15:0] * step + w[15:0] * stride)
        : (offs + n_eff * step);

    always @(posedge clk) begin
        addr_valid <= 1'b0;
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; addr <= 11'd0;
            n <= 17'd0; j <= 17'd0; w <= 17'd0; hold_cnt <= 8'd0;
        end else if (trigger) begin
            n <= 17'd0; j <= 17'd0; w <= 17'd0; done <= 1'b0;
            if (!en) begin
                state <= S_DONE; done <= 1'b1;
            end else if (skew != 8'd0) begin
                state <= S_HOLD; hold_cnt <= skew;
            end else begin
                state <= S_RUN;
            end
        end else case (state)
            S_HOLD: begin
                hold_cnt <= hold_cnt - 8'd1;
                if (hold_cnt == 8'd1) state <= S_RUN;
            end
            S_RUN: begin
                addr       <= off16[10:0];
                addr_valid <= 1'b1;
                n <= n + 17'd1;
                if (j == seg_len - 17'd1) begin
                    j <= 17'd0; w <= w + 17'd1;
                end else begin
                    j <= j + 17'd1;
                end
                if (n == {1'b0, count}) begin
                    state <= S_DONE; done <= 1'b1;
                end
            end
            default: ; // idle / done
        endcase
    end
endmodule
