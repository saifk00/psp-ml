// Self-checking testbench for the VME emulator.
//
// Every test builds a complete 106-word context image, stores it word by
// word through the memory-mapped context window (the section 11.3 "fine
// path"), stages input data in buffer memory, writes TRIGGER to DMA_CTRL,
// polls DMA_STAT[11], and reads results back out of BASE_0 -- i.e. it
// drives the emulator exactly the way me-core-lib drives the real array.
//
// Tests:
//   A  VMUL with rounding      dual-stream multiply, R and K descriptor bits
//   B  MACI                    running multiply-accumulate (the accumulator)
//   C  ADD -> CLAMP via FU1    staging bus + secondary unit + FU1EN
//   D  bit-reversed addressing FMT1.BRV / BRVW
//   E  drain + negative offset the section 7.7 drain construction
//   F  segment replay          counter B: coefficients against a data stream
`timescale 1ns/1ps
module tb_vme;
    reg clk = 0;
    reg rst = 1;
    reg host_we = 0;
    reg [19:0] host_addr = 0;
    reg [31:0] host_wdata = 0;
    wire [31:0] host_rdata;
    wire busy, done;

    vme_top dut (
        .clk(clk), .rst(rst),
        .host_we(host_we), .host_addr(host_addr), .host_wdata(host_wdata),
        .host_rdata(host_rdata), .busy(busy), .done(done)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    integer checks = 0;

    // ------------------------------------------------------------------
    // host bus helpers
    // ------------------------------------------------------------------
    task hw(input [19:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            host_we = 1; host_addr = addr; host_wdata = data;
            @(negedge clk);
            host_we = 0;
        end
    endtask

    task hr(input [19:0] addr, output [31:0] data);
        begin
            @(negedge clk);
            host_addr = addr;
            #1 data = host_rdata;
        end
    endtask

    function [19:0] buf_addr(input [2:0] b, input [10:0] w);
        buf_addr = (b < 3'd4) ? ({b[1:0], 13'd0} + {w, 2'b00})
                              : (20'h20000 + {b[1:0], 13'd0} + {w, 2'b00});
    endfunction

    function [19:0] ctx_addr(input [6:0] i);
        ctx_addr = 20'hF8000 + {i, 2'b00};
    endfunction

    // ------------------------------------------------------------------
    // context image
    // ------------------------------------------------------------------
    reg [31:0] c [0:105];
    integer i;

    // word indices of PE0's AGU groups
    localparam RTOP = 33, RBASE = 39, WR = 45;

    task ctx_default;   // everything parked: FUs at MOV, every AGU disabled
        begin
            for (i = 0; i < 106; i = i + 1) c[i] = 32'h0;
            for (i = 0; i < 8; i = i + 1) c[i] = 32'h0000_4000;
            c[29] = 32'h0000_3210;
            c[30] = 32'h0000_3210;
            c[105] = 32'h0000_0018;
            for (i = 0; i < 12; i = i + 1)
                c[33 + (i / 3) * 18 + (i % 3) * 6 + 1] = 32'h0001_0000;
        end
    endtask

    task ctx_flush;
        begin
            for (i = 0; i < 106; i = i + 1)
                hw(ctx_addr(i[6:0]), c[i]);
        end
    endtask

    task run_and_wait;
        integer t;
        reg [31:0] st;
        begin
            hw(20'hFF008, 32'h18);          // DMA_CTRL = TRIGGER
            st = 0;
            for (t = 0; t < 5000 && !st[11]; t = t + 1)
                hr(20'hFF000, st);          // poll DMA_STAT.VD
            if (!st[11]) begin
                $display("FAIL: array never reported done");
                errors = errors + 1;
            end
        end
    endtask

    task check(input [255:0] name, input [10:0] w, input [31:0] exp);
        reg [31:0] got;
        begin
            hr(buf_addr(3'd0, w), got);     // results land in BASE_0
            checks = checks + 1;
            if (got !== exp) begin
                errors = errors + 1;
                $display("FAIL %0s: BASE_0[%0d] = %h, expected %h", name, w, got, exp);
            end
        end
    endtask

    function [31:0] sext24(input [63:0] x);
        sext24 = {{8{x[23]}}, x[23:0]};
    endfunction

    // descriptor field constants (Table 5.1 / Appendix C)
    localparam FSEL_BASE1 = 32'h5000_0000;
    localparam BSEL_TOP0  = 32'h0000_0000;
    localparam BSEL_STG0  = 32'h0400_0000;
    localparam OP_VMUL    = 32'h0022_0000;
    localparam OP_MACI    = 32'h0024_0000;
    localparam OP_ADD     = 32'h0001_0000;
    localparam OP_CLAMP   = 32'h0007_C000;
    localparam OP_MOV_I   = 32'h0000_4000;
    localparam RBIT       = 32'h0000_0040;
    localparam MODE_LIN   = 32'h8400_0000;

    reg signed [63:0] va, vb, prod, acc64;
    reg [31:0] tmp;
    integer j;

    initial begin
        repeat (4) @(negedge clk);
        rst = 0;
        repeat (2) @(negedge clk);

        // ==========================================================
        // Test A: VMUL, K=4, R=1 -- BASE_0[i] = round((TOP_0*BASE_1)>>4)
        // ==========================================================
        for (i = 0; i < 16; i = i + 1) begin
            hw(buf_addr(3'd4, i[10:0]), i + 1);          // TOP_0
            hw(buf_addr(3'd1, i[10:0]), 3 * i - 20);     // BASE_1
        end
        ctx_default;
        c[0]        = FSEL_BASE1 | BSEL_TOP0 | OP_VMUL | RBIT | 32'd4;
        c[RTOP]     = MODE_LIN;          c[RTOP + 1]  = 32'h0001_000F;
        c[RBASE]    = MODE_LIN;          c[RBASE + 1] = 32'h0001_000F;
        c[WR]       = MODE_LIN | 32'h0002_0000;  // write skew 2 (FU0 latency)
        c[WR + 1]   = 32'h0001_000F;
        ctx_flush;
        run_and_wait;
        for (i = 0; i < 16; i = i + 1) begin
            va = i + 1; vb = 3 * i - 20;
            prod = ((va * vb) + 64'sd8) >>> 4;
            check("A/vmul", i[10:0], sext24(prod));
        end

        // ==========================================================
        // Test B: MACI dot product -- BASE_0[i] = sum(TOP_0[0..i]*BASE_1[0..i])
        // ==========================================================
        ctx_default;
        c[0]      = FSEL_BASE1 | BSEL_TOP0 | OP_MACI;    // K=0, b=0
        c[RTOP]   = MODE_LIN;           c[RTOP + 1]  = 32'h0001_000F;
        c[RBASE]  = MODE_LIN;           c[RBASE + 1] = 32'h0001_000F;
        c[WR]     = MODE_LIN | 32'h0002_0000;
        c[WR + 1] = 32'h0001_000F;
        ctx_flush;
        run_and_wait;
        acc64 = 0;
        for (i = 0; i < 16; i = i + 1) begin
            va = i + 1; vb = 3 * i - 20;
            acc64 = acc64 + va * vb;
            check("B/maci", i[10:0], sext24(acc64));
        end

        // ==========================================================
        // Test C: FU0 ADD, FU1 CLAMP on STG_0, FU1 drives the write port
        // ==========================================================
        for (i = 0; i < 16; i = i + 1) begin
            hw(buf_addr(3'd4, i[10:0]), 7 * i - 40);     // TOP_0
            hw(buf_addr(3'd1, i[10:0]), 5 * i);          // BASE_1
        end
        ctx_default;
        c[0]      = FSEL_BASE1 | BSEL_TOP0 | OP_ADD;     // sum, K=0
        c[4]      = BSEL_STG0 | OP_CLAMP;                // clamp(a=ceil, b=floor)
        c[16]     = 32'd100;                             // FU1 a: ceiling
        c[17]     = -32'd30;                             // FU1 b: floor
        c[27]     = 32'h8000_0000;                       // FU1EN: PE0
        c[RTOP]   = MODE_LIN;           c[RTOP + 1]  = 32'h0001_000F;
        c[RBASE]  = MODE_LIN;           c[RBASE + 1] = 32'h0001_000F;
        c[WR]     = MODE_LIN | 32'h0003_0000;    // write skew 3 (FU1 latency)
        c[WR + 1] = 32'h0001_000F;
        ctx_flush;
        run_and_wait;
        for (i = 0; i < 16; i = i + 1) begin
            va = 12 * i - 40;
            if (va > 100) va = 100;
            if (va < -30) va = -30;
            check("C/clamp", i[10:0], sext24(va));
        end

        // ==========================================================
        // Test D: bit-reversed read -- BASE_0[m] = TOP_0[bitrev4(m)]
        // ==========================================================
        for (i = 0; i < 16; i = i + 1)
            hw(buf_addr(3'd4, i[10:0]), 32'h111 * i);
        ctx_default;
        c[0]        = OP_MOV_I;                          // MOV back (TOP_0)
        c[RTOP]     = MODE_LIN;         c[RTOP + 1] = 32'h0001_000F;
        c[RTOP + 5] = 32'hA400_0004;                     // FMT1: BITREV(4)
        c[WR]       = MODE_LIN | 32'h0002_0000;
        c[WR + 1]   = 32'h0001_000F;
        ctx_flush;
        run_and_wait;
        for (i = 0; i < 16; i = i + 1) begin
            j = {i[0], i[1], i[2], i[3]};                // bitrev4
            check("D/bitrev", i[10:0], 32'h111 * j);
        end

        // ==========================================================
        // Test E: drain construction -- DRAIN=16 displaces valid data by 16,
        // which a following stage would cancel with a -16 start offset
        // ==========================================================
        for (i = 0; i < 32; i = i + 1)
            hw(buf_addr(3'd4, i[10:0]), 32'h0BEE0 + i);
        ctx_default;
        c[0]        = OP_MOV_I;
        c[RTOP]     = MODE_LIN;         c[RTOP + 1] = 32'h0001_001F;
        c[WR]       = MODE_LIN | 32'h0002_0000;
        c[WR + 1]   = 32'h0001_001F;                     // 32 elements
        c[WR + 4]   = 32'h0000_0010;                     // FMT0: DRAIN = 16
        c[WR + 5]   = 32'h0020_0000;                     // FMT1: END token
        ctx_flush;
        run_and_wait;
        for (i = 0; i < 16; i = i + 1)
            check("E/drain", i[10:0] + 11'd16, 32'h0BEE0 + i);

        // ==========================================================
        // Test F: segment replay -- BASE_0[i] = TOP_0[i] * BASE_1[i mod 4]
        // ==========================================================
        for (i = 0; i < 16; i = i + 1)
            hw(buf_addr(3'd4, i[10:0]), i + 2);          // TOP_0 data stream
        for (i = 0; i < 4; i = i + 1)
            hw(buf_addr(3'd1, i[10:0]), 10 * i - 15);    // BASE_1 coefficients
        ctx_default;
        c[0]         = FSEL_BASE1 | BSEL_TOP0 | OP_VMUL; // K=0
        c[RTOP]      = MODE_LIN;        c[RTOP + 1]  = 32'h0001_000F;
        c[RBASE]     = MODE_LIN;        c[RBASE + 1] = 32'h0001_000F;
        c[RBASE + 2] = 32'h0001_0003;                    // INNER0: seg len 4
        c[RBASE + 4] = 32'h0002_0000;                    // FMT0: RING
        c[WR]        = MODE_LIN | 32'h0002_0000;
        c[WR + 1]    = 32'h0001_000F;
        ctx_flush;
        run_and_wait;
        for (i = 0; i < 16; i = i + 1) begin
            va = i + 2; vb = 10 * (i % 4) - 15;
            check("F/replay", i[10:0], sext24(va * vb));
        end

        // ==========================================================
        // Test G: cross-PE staging pipeline -- PE0 multiplies, PE1 reads
        // the product off staging tap 0, adds BASE_2, and writes BASE_1.
        // A two-stage skew ladder: PE1's base read +1, PE1's write +3.
        // ==========================================================
        for (i = 0; i < 16; i = i + 1) begin
            hw(buf_addr(3'd4, i[10:0]), i + 1);          // TOP_0
            hw(buf_addr(3'd5, i[10:0]), 2 * i - 9);      // TOP_1
            hw(buf_addr(3'd2, i[10:0]), 100 - 7 * i);    // BASE_2
        end
        ctx_default;
        c[0]      = 32'h1000_0000 | BSEL_TOP0 | OP_VMUL; // PE0: TOP_0 * TOP_1
        c[RTOP]   = MODE_LIN;           c[RTOP + 1] = 32'h0001_000F;
        c[1]      = 32'h6000_0000 | BSEL_STG0 | OP_ADD;  // PE1: STG_0 + BASE_2
        c[57]     = MODE_LIN | 32'h0001_0000;            // PE1 RBASE skew 1
        c[58]     = 32'h0001_000F;
        c[63]     = MODE_LIN | 32'h0003_0000;            // PE1 WR skew 3
        c[64]     = 32'h0001_000F;
        ctx_flush;
        run_and_wait;
        for (i = 0; i < 16; i = i + 1) begin
            va = (i + 1) * (2 * i - 9) + (100 - 7 * i);
            hr(buf_addr(3'd1, i[10:0]), tmp);            // PE1 writes BASE_1
            checks = checks + 1;
            if (tmp !== sext24(va)) begin
                errors = errors + 1;
                $display("FAIL G/staging: BASE_1[%0d] = %h, expected %h",
                         i, tmp, sext24(va));
            end
        end

        // ==========================================================
        if (errors == 0)
            $display("ALL %0d CHECKS PASSED", checks);
        else begin
            $display("%0d/%0d CHECKS FAILED", errors, checks);
            $fatal(1);
        end
        $finish;
    end

    initial begin
        #2_000_000;
        $display("FAIL: global timeout");
        $fatal(1);
    end
endmodule
