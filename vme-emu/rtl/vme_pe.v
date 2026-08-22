`timescale 1ns/1ps
// One VME processing element (chapter 4): three AGUs (RTOP, RBASE, WR),
// two functional units, four operand muxes and one write port.
//
// Pipeline of this model (each arrow is one clock):
//   AGU address issue -> buffer read data -> FU0 result register (staging
//   tap) -> FU1 result register -> write commit.
// Nothing interlocks, exactly as documented: the write port stores whatever
// the selected FU's result register holds on the cycle its WR AGU emits an
// address, so a correct context skews its write port past its reads --
// WR SKEW = read SKEW + 2 when FU0 drives the port, + 3 when FU1 does
// (see README "Timing of this model").
//
// FMT0.DRAIN on the write port (when FMT1.END is set) inserts DRAIN extra
// cycles of delay into the write data path, so the first DRAIN offsets of
// the address sequence receive pipeline junk and valid element j lands at
// sequence position j + DRAIN -- which a downstream or final stage cancels
// with a negative start offset, per the drain construction of section 7.7.
//
// The AGU configuration arrives pre-muxed (ICN_CFGMAP inheritance is
// resolved by vme_top); the read address streams likewise leave this
// module and come back post-ICN_SRCMAP as `buf_rdata`/`*_rv`.
module vme_pe #(
    parameter MAX_DRAIN = 64
) (
    input  wire         clk,
    input  wire         rst,
    input  wire         trigger,
    // context, post-CFGMAP
    input  wire [31:0]  fu0_cfg, fu0_a, fu0_b,
    input  wire [31:0]  fu1_cfg, fu1_a, fu1_b,
    input  wire         fu1_en,
    input  wire [191:0] rtop_cfg,    // 6 words, word i at [i*32 +: 32]
    input  wire [191:0] rbase_cfg,
    input  wire [191:0] wr_cfg,
    input  wire [2:0]   tskew_code,  // ICN_SKEW codes for this PE
    input  wire [2:0]   bskew_code,
    // own read address streams (pre-SRCMAP, consumed by vme_top)
    output wire [10:0]  top_addr,
    output wire         top_avalid,
    output wire [10:0]  base_addr,
    output wire         base_avalid,
    // read data at this PE's effective (post-SRCMAP) addresses
    input  wire [255:0] buf_rdata,   // internal index 0-3 BASE, 4-7 TOP
    input  wire         top_rv,      // validity of that data, read-latency aligned
    input  wire         base_rv,
    // staging bus
    input  wire [255:0] stg_data,
    input  wire [7:0]   stg_valid,
    output wire [31:0]  tap0_data,   // this PE's primary tap
    output wire         tap0_valid,
    output wire [31:0]  tap1_data,   // secondary tap
    output wire         tap1_valid,
    // write port (vme_top wires it to BASE_n -- section 4.2: PEn writes BASE_n)
    output wire         wr_en,
    output wire [10:0]  wr_addr,
    output wire [31:0]  wr_data,
    output wire [2:0]   agu_done     // {WR, RBASE, RTOP}
);
    // ------------------------------------------------------------------
    // address generation
    // ------------------------------------------------------------------
    wire wr_avalid;
    vme_agu u_rtop (
        .clk(clk), .rst(rst), .trigger(trigger),
        .w_mode  (rtop_cfg[0*32 +: 32]), .w_count (rtop_cfg[1*32 +: 32]),
        .w_inner0(rtop_cfg[2*32 +: 32]), .w_inner1(rtop_cfg[3*32 +: 32]),
        .w_fmt0  (rtop_cfg[4*32 +: 32]), .w_fmt1  (rtop_cfg[5*32 +: 32]),
        .addr(top_addr), .addr_valid(top_avalid), .done(agu_done[0])
    );
    vme_agu u_rbase (
        .clk(clk), .rst(rst), .trigger(trigger),
        .w_mode  (rbase_cfg[0*32 +: 32]), .w_count (rbase_cfg[1*32 +: 32]),
        .w_inner0(rbase_cfg[2*32 +: 32]), .w_inner1(rbase_cfg[3*32 +: 32]),
        .w_fmt0  (rbase_cfg[4*32 +: 32]), .w_fmt1  (rbase_cfg[5*32 +: 32]),
        .addr(base_addr), .addr_valid(base_avalid), .done(agu_done[1])
    );
    vme_agu u_wr (
        .clk(clk), .rst(rst), .trigger(trigger),
        .w_mode  (wr_cfg[0*32 +: 32]), .w_count (wr_cfg[1*32 +: 32]),
        .w_inner0(wr_cfg[2*32 +: 32]), .w_inner1(wr_cfg[3*32 +: 32]),
        .w_fmt0  (wr_cfg[4*32 +: 32]), .w_fmt1  (wr_cfg[5*32 +: 32]),
        .addr(wr_addr), .addr_valid(wr_avalid), .done(agu_done[2])
    );

    // ------------------------------------------------------------------
    // operand muxes (FSEL [31:27], BSEL [26:22] of each descriptor)
    // ------------------------------------------------------------------
    wire [31:0] fu0_back_d,  fu0_front_d, fu1_back_d, fu1_front_d;
    wire        fu0_back_v,  fu0_front_v, fu1_back_v, fu1_front_v;

    vme_operand u_op0b (.clk(clk), .sel(fu0_cfg[26:22]),
        .buf_rdata(buf_rdata), .top_rv(top_rv), .base_rv(base_rv),
        .stg_data(stg_data), .stg_valid(stg_valid),
        .tcode(tskew_code), .bcode(bskew_code),
        .data(fu0_back_d), .valid(fu0_back_v));
    vme_operand u_op0f (.clk(clk), .sel(fu0_cfg[31:27]),
        .buf_rdata(buf_rdata), .top_rv(top_rv), .base_rv(base_rv),
        .stg_data(stg_data), .stg_valid(stg_valid),
        .tcode(tskew_code), .bcode(bskew_code),
        .data(fu0_front_d), .valid(fu0_front_v));
    vme_operand u_op1b (.clk(clk), .sel(fu1_cfg[26:22]),
        .buf_rdata(buf_rdata), .top_rv(top_rv), .base_rv(base_rv),
        .stg_data(stg_data), .stg_valid(stg_valid),
        .tcode(tskew_code), .bcode(bskew_code),
        .data(fu1_back_d), .valid(fu1_back_v));
    vme_operand u_op1f (.clk(clk), .sel(fu1_cfg[31:27]),
        .buf_rdata(buf_rdata), .top_rv(top_rv), .base_rv(base_rv),
        .stg_data(stg_data), .stg_valid(stg_valid),
        .tcode(tskew_code), .bcode(bskew_code),
        .data(fu1_front_d), .valid(fu1_front_v));

    // ------------------------------------------------------------------
    // functional units
    // ------------------------------------------------------------------
    vme_fu u_fu0 (
        .clk(clk), .rst(rst), .trigger(trigger),
        .cfg(fu0_cfg), .const_a(fu0_a), .const_b(fu0_b),
        .back(fu0_back_d), .back_v(fu0_back_v),
        .front(fu0_front_d), .front_v(fu0_front_v),
        .result(tap0_data), .result_v(tap0_valid)
    );
    vme_fu u_fu1 (
        .clk(clk), .rst(rst), .trigger(trigger),
        .cfg(fu1_cfg), .const_a(fu1_a), .const_b(fu1_b),
        .back(fu1_back_d), .back_v(fu1_back_v),
        .front(fu1_front_d), .front_v(fu1_front_v),
        .result(tap1_data), .result_v(tap1_valid)
    );

    // ------------------------------------------------------------------
    // write port: FU1 drives it when enabled, else FU0 (section 4.5),
    // through the DRAIN delay line
    // ------------------------------------------------------------------
    wire [31:0] wsel_data = fu1_en ? tap1_data : tap0_data;
    wire [15:0] drain     = wr_cfg[4*32 +: 16];   // WR FMT0[15:0]
    wire        end_tok   = wr_cfg[5*32 + 21];    // WR FMT1[21]
    wire [6:0]  dly       = !end_tok            ? 7'd0 :
                            (drain > MAX_DRAIN) ? MAX_DRAIN[6:0] :
                                                  drain[6:0];

    reg [31:0] dpipe [0:MAX_DRAIN-1];
    integer i;
    initial for (i = 0; i < MAX_DRAIN; i = i + 1) dpipe[i] = 32'd0;
    always @(posedge clk) begin
        dpipe[0] <= wsel_data;
        for (i = MAX_DRAIN - 1; i > 0; i = i - 1) dpipe[i] <= dpipe[i - 1];
    end

    assign wr_en   = wr_avalid;
    assign wr_data = (dly == 7'd0) ? wsel_data : dpipe[dly - 7'd1];
endmodule
