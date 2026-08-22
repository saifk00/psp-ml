`timescale 1ns/1ps
// VME array top level: the 106-word context register file, four processing
// elements, eight ring buffers, the staging bus and the interconnect
// registers -- everything of docs/vme-reference.html that is the array
// itself (the local DMA controller is reduced to the two words the array
// needs: TRIGGER and DMA_STAT, since block moves are the host port here).
//
// The host port speaks the documented memory map, as byte offsets from
// 0x4400_0000 (writes take effect at the clock edge; reads are
// combinational):
//   0x00000-0x07FFF  BASE_0..3       (0x2000 bytes each)
//   0x20000-0x27FFF  TOP_0..3
//   0xF8000-0xF81A4  context words 0..105  (a live image: single-word
//                    stores rewire one node, exactly like section 11.3)
//   0xFF000          DMA_STAT: [11] VD array done, [9] TD always set
//   0xFF008          DMA_CTRL: writing 0x18 (TRIGGER) starts the array
//
// Interconnect semantics implemented here:
//   ICN_CFGMAP  PE p's whole 18-word AGU block is taken from element
//               CFGMAP[p] (out-of-range: own block).
//   ICN_SRCMAP  PE p's read address streams come from element SRCMAP[p]'s
//               AGUs; out-of-range leaves the lane unrouted (reads never
//               validate).
//   ICN_SKEW    per-PE 3-bit skew codes, applied in vme_operand.
//   FU1EN       bit 31-p enables PE p's secondary unit onto its write port.
//   ICN_INMAP / ICN_XPCR / MOD_A/B/C are stored and readable but inert --
//   the descriptor's own selectors always win over INMAP (section 8.2),
//   and no context in the manual's examples depends on the others.
module vme_top #(
    parameter MAX_DRAIN = 64,
    parameter FLUSH_CYCLES = 16
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        host_we,
    input  wire [19:0] host_addr,
    input  wire [31:0] host_wdata,
    output wire [31:0] host_rdata,
    output wire        busy,
    output wire        done        // DMA_STAT[11], VD
);
    // ------------------------------------------------------------------
    // context register file (word indices per Table 3.2)
    // ------------------------------------------------------------------
    reg [31:0] ctx [0:105];

    wire sel_buf = (host_addr[19:18] == 2'b00) && !host_addr[15];
    wire [2:0]  hbuf    = {host_addr[17], host_addr[14:13]};
    wire [10:0] hword   = host_addr[12:2];
    wire sel_ctx = (host_addr[19:12] == 8'hF8);
    wire [7:0]  ctx_idx = host_addr[9:2];
    wire sel_dma = (host_addr[19:12] == 8'hFF);

    reg  trig;      // one-cycle pulse into the array
    reg  vd;        // array-done flag
    integer ci;

    always @(posedge clk) begin
        trig <= 1'b0;
        if (rst) begin
            for (ci = 0; ci < 106; ci = ci + 1) ctx[ci] <= 32'h0;
            for (ci = 0; ci < 8; ci = ci + 1)
                ctx[ci] <= 32'h0000_4000;             // FU descriptors: MOV
            ctx[29]  <= 32'h0000_3210;                // ICN_SRCMAP identity
            ctx[30]  <= 32'h0000_3210;                // ICN_CFGMAP identity
            ctx[105] <= 32'h0000_0018;                // CTX_END
            for (ci = 0; ci < 12; ci = ci + 1) begin  // all 12 AGU groups
                ctx[33 + (ci / 3) * 18 + (ci % 3) * 6]     <= 32'h8400_0000;
                ctx[33 + (ci / 3) * 18 + (ci % 3) * 6 + 1] <= 32'h0001_0000;
            end
        end else if (host_we) begin
            if (sel_ctx && ctx_idx < 8'd106)
                ctx[ctx_idx[6:0]] <= host_wdata;
            if (sel_dma && host_addr[11:0] == 12'h008 && host_wdata == 32'h18)
                trig <= 1'b1;                         // DMA_CTRL TRIGGER
        end
    end

    // ------------------------------------------------------------------
    // per-PE context slices and interconnect muxing
    // ------------------------------------------------------------------
    wire [43:0]  top_addr_f, base_addr_f;   // own AGU streams, 4 x 11
    wire [3:0]   top_av_f, base_av_f;
    wire [43:0]  eff_top_addr_f, eff_base_addr_f;
    wire [3:0]   eff_top_av, eff_base_av;
    reg  [3:0]   top_rv_d, base_rv_d;       // aligned to RAM read latency
    wire [255:0] stg_data;
    wire [7:0]   stg_valid;
    wire [127:0] tap0_f, tap1_f;
    wire [3:0]   tap0_v, tap1_v;
    wire [3:0]   pe_wr_en;
    wire [43:0]  pe_wr_addr_f;
    wire [127:0] pe_wr_data_f;
    wire [11:0]  agu_done_f;
    wire [127:0] rb_rdata [0:7];
    wire [31:0]  rb_hrdata [0:7];

    assign stg_data  = {tap1_f, tap0_f};
    assign stg_valid = {tap1_v, tap0_v};

    genvar p, b;
    generate
    for (p = 0; p < 4; p = p + 1) begin : g_pe
        // ICN_SRCMAP: whose address streams feed this PE's reads
        wire [3:0] smap = ctx[29][p*4 +: 4];
        wire       srouted = (smap < 4'd4);
        assign eff_top_addr_f [p*11 +: 11] = srouted ? top_addr_f [smap[1:0]*11 +: 11] : 11'd0;
        assign eff_base_addr_f[p*11 +: 11] = srouted ? base_addr_f[smap[1:0]*11 +: 11] : 11'd0;
        assign eff_top_av [p] = srouted ? top_av_f [smap[1:0]] : 1'b0;
        assign eff_base_av[p] = srouted ? base_av_f[smap[1:0]] : 1'b0;

        // ICN_CFGMAP: whose AGU block this PE adopts
        wire [3:0] cmap = ctx[30][p*4 +: 4];
        wire [1:0] ce   = (cmap < 4'd4) ? cmap[1:0] : p[1:0];
        wire [6:0] wb   = 7'd33 + {1'b0, ce, 4'd0} + {4'd0, ce, 1'b0};  // 33 + 18*ce
        wire [191:0] rtop_cfg  = {ctx[wb+5],  ctx[wb+4],  ctx[wb+3],
                                  ctx[wb+2],  ctx[wb+1],  ctx[wb]};
        wire [191:0] rbase_cfg = {ctx[wb+11], ctx[wb+10], ctx[wb+9],
                                  ctx[wb+8],  ctx[wb+7],  ctx[wb+6]};
        wire [191:0] wr_cfg    = {ctx[wb+17], ctx[wb+16], ctx[wb+15],
                                  ctx[wb+14], ctx[wb+13], ctx[wb+12]};

        // read data bus: internal index 0-3 BASE, 4-7 TOP, at this PE's lane
        wire [255:0] buf_rdata_pe = {rb_rdata[7][p*32 +: 32], rb_rdata[6][p*32 +: 32],
                                     rb_rdata[5][p*32 +: 32], rb_rdata[4][p*32 +: 32],
                                     rb_rdata[3][p*32 +: 32], rb_rdata[2][p*32 +: 32],
                                     rb_rdata[1][p*32 +: 32], rb_rdata[0][p*32 +: 32]};

        vme_pe #(.MAX_DRAIN(MAX_DRAIN)) u_pe (
            .clk(clk), .rst(rst), .trigger(trig),
            .fu0_cfg(ctx[p]),    .fu0_a(ctx[8+2*p]),  .fu0_b(ctx[9+2*p]),
            .fu1_cfg(ctx[4+p]),  .fu1_a(ctx[16+2*p]), .fu1_b(ctx[17+2*p]),
            .fu1_en(ctx[27][31-p]),
            .rtop_cfg(rtop_cfg), .rbase_cfg(rbase_cfg), .wr_cfg(wr_cfg),
            .tskew_code(ctx[32][16+3*p +: 3]),
            .bskew_code(ctx[32][3*p +: 3]),
            .top_addr (top_addr_f [p*11 +: 11]), .top_avalid (top_av_f[p]),
            .base_addr(base_addr_f[p*11 +: 11]), .base_avalid(base_av_f[p]),
            .buf_rdata(buf_rdata_pe),
            .top_rv(top_rv_d[p]), .base_rv(base_rv_d[p]),
            .stg_data(stg_data), .stg_valid(stg_valid),
            .tap0_data(tap0_f[p*32 +: 32]), .tap0_valid(tap0_v[p]),
            .tap1_data(tap1_f[p*32 +: 32]), .tap1_valid(tap1_v[p]),
            .wr_en(pe_wr_en[p]),
            .wr_addr(pe_wr_addr_f[p*11 +: 11]),
            .wr_data(pe_wr_data_f[p*32 +: 32]),
            .agu_done(agu_done_f[p*3 +: 3])
        );
    end

    // ------------------------------------------------------------------
    // ring buffers: 0-3 BASE (array-writable, PEn -> BASE_n), 4-7 TOP
    // ------------------------------------------------------------------
    for (b = 0; b < 8; b = b + 1) begin : g_rb
        wire is_top = (b >= 4);
        vme_ringbuf u_rb (
            .clk(clk),
            .raddr(is_top ? eff_top_addr_f : eff_base_addr_f),
            .rdata(rb_rdata[b]),
            .we   (is_top ? 1'b0  : pe_wr_en[b % 4]),
            .waddr(is_top ? 11'd0 : pe_wr_addr_f[(b % 4)*11 +: 11]),
            .wdata(is_top ? 32'd0 : pe_wr_data_f[(b % 4)*32 +: 32]),
            .hwe(host_we && sel_buf && (hbuf == b[2:0])),
            .haddr(hword),
            .hwdata(host_wdata),
            .hrdata(rb_hrdata[b])
        );
    end
    endgenerate

    always @(posedge clk) begin
        top_rv_d  <= rst ? 4'd0 : eff_top_av;
        base_rv_d <= rst ? 4'd0 : eff_base_av;
    end

    // ------------------------------------------------------------------
    // completion: all twelve AGUs done, then a fixed pipeline flush
    // ------------------------------------------------------------------
    reg       running;
    reg [7:0] flush;
    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0; vd <= 1'b0; flush <= 8'd0;
        end else if (trig) begin
            running <= 1'b1; vd <= 1'b0; flush <= 8'd0;
        end else if (running) begin
            if (&agu_done_f) begin
                flush <= flush + 8'd1;
                if (flush == FLUSH_CYCLES[7:0]) begin
                    running <= 1'b0; vd <= 1'b1;
                end
            end else
                flush <= 8'd0;
        end
    end
    assign busy = running;
    assign done = vd;

    // ------------------------------------------------------------------
    // host reads (combinational)
    // ------------------------------------------------------------------
    assign host_rdata =
        sel_ctx ? ((ctx_idx < 8'd106) ? ctx[ctx_idx[6:0]] : 32'd0) :
        sel_dma ? ((host_addr[11:0] == 12'h000)
                       ? {20'd0, vd, 1'b0, 1'b1, 9'd0}   // DMA_STAT: VD, TD
                       : 32'd0) :
        sel_buf ? rb_hrdata[hbuf] : 32'd0;
endmodule
