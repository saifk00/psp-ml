`timescale 1ns/1ps
// VME functional unit -- one of eight (FU0/FU1 per PE).
//
// Decodes the 32-bit descriptor of docs/vme-reference.html chapter 5:
//   [31:27] FSEL  (consumed by the PE's operand mux, not here)
//   [26:22] BSEL  (likewise)
//   [21:12] OP    = CLASS[9:8] FN[7:4] OPM[3:2] ACC[1:0]
//   [11:7]  SAT   saturation width in bits (0 = off)
//   [6]     R     round the post-op shift
//   [5:0]   K     post-op shift amount
//
// The operation set is the complete Appendix C table (Table C.1),
// implemented per-equation in 64-bit signed arithmetic.  OPM bit 0
// substitutes constant b for the front stream; OPM bit 1 negates the back
// operand for classes 1 and 2 (for class 0 the per-mode tables are
// authoritative and each {FN,OPM} cell is implemented verbatim).
//
// The registered result is truncated to the architectural sample format:
// 24-bit two's complement sign-extended into 32 bits (section 2.4).  The
// unit paces on the validity of its *back* source ("back is the primary
// operand"); the front stream is sampled un-interlocked, exactly like the
// real array -- aligning it is the programmer's skew problem.
//
// Stream state (accumulator, out[n-1], back[n-1], parity, n) is cleared by
// `trigger`, matching "reset before every new context" semantics.  ACC:
//   00 NONE  result = op output
//   01 HOLD  acc += result, result = accumulated value
//   10 LOAD  acc preloaded from constant a at trigger
//   11 ZERO  acc zeroed at trigger
// (MACI/SAD/ROR64 own the accumulator internally and ignore HOLD.)
module vme_fu #(
    // Extra result-pipeline stages after the compute register.  Calibrated
    // against silicon (2026-08-27 skew probes): a buffer-read FU0 result is
    // captured by a write port skewed 6 cycles after address issue =
    // 3 (buffer read path) + 1 (compute register) + 2 (this), and each
    // staging hop adds the FU half only (3 cycles).
    parameter RESULT_PIPE = 2
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        trigger,
    input  wire [31:0] cfg,
    input  wire [31:0] const_a,
    input  wire [31:0] const_b,
    input  wire [31:0] back,
    input  wire        back_v,
    input  wire [31:0] front,
    input  wire        front_v,   // sampled, not interlocked
    output wire [31:0] result,
    output wire        result_v
);
    reg [31:0] result_c;
    reg        result_v_c;
    reg [32:0] rpipe [0:RESULT_PIPE-1];
    integer pi;
    initial for (pi = 0; pi < RESULT_PIPE; pi = pi + 1) rpipe[pi] = 33'd0;
    always @(posedge clk) begin
        rpipe[0] <= {result_v_c, result_c};
        for (pi = RESULT_PIPE - 1; pi > 0; pi = pi - 1) rpipe[pi] <= rpipe[pi - 1];
    end
    assign result   = (RESULT_PIPE == 0) ? result_c   : rpipe[RESULT_PIPE-1][31:0];
    assign result_v = (RESULT_PIPE == 0) ? result_v_c : rpipe[RESULT_PIPE-1][32];
    wire [1:0] klass = cfg[21:20];
    wire [3:0] fn    = cfg[19:16];
    wire [1:0] opm   = cfg[15:14];
    wire [1:0] accm  = cfg[13:12];
    wire [4:0] sat   = cfg[11:7];
    wire       rnd   = cfg[6];
    wire [5:0] k     = cfg[5:0];

    wire en = back_v;

    // ------------------------------------------------------------------
    // stream state
    // ------------------------------------------------------------------
    reg signed [63:0] acc;
    reg signed [31:0] prev_out;     // out[n-1]
    reg signed [31:0] prev_back;    // back[n-1] (MULD: init = b)
    reg signed [31:0] prev_back2;   // back[n-2]
    reg signed [31:0] prev_front;   // front[n-1]
    reg signed [31:0] prev_front2;  // front[n-2]
    reg signed [31:0] runmax;       // PCLAMP running max of back[0..n-1]
    reg        [31:0] par;          // PARITY running xor
    reg        [16:0] n;

    // ------------------------------------------------------------------
    // operands
    // ------------------------------------------------------------------
    wire signed [63:0] sB  = {{32{back[31]}},  back};
    wire signed [63:0] sF  = {{32{front[31]}}, front};
    wire signed [63:0] sA  = {{32{const_a[31]}}, const_a};
    wire signed [63:0] sCb = {{32{const_b[31]}}, const_b};
    wire signed [63:0] Fi  = opm[0] ? sCb : sF;      // front per OPM bit 2
    wire signed [63:0] Bn  = opm[1] ? -sB : sB;      // classes 1/2 only
    wire        [5:0]  bsh = const_b[5:0];           // shift amounts from b
    wire        [5:0]  fsh = Fi[5:0];
    wire        [5:0]  Bsh = back[5:0];

    function signed [63:0] f_shr;   // the >>k of the tables, with rounding
        input signed [63:0] x;
        begin
            if (k == 6'd0)      f_shr = x;
            else if (rnd)       f_shr = (x + (64'sd1 <<< (k - 6'd1))) >>> k;
            else                f_shr = x >>> k;
        end
    endfunction

    function signed [63:0] f_abs;
        input signed [63:0] x;
        f_abs = (x < 0) ? -x : x;
    endfunction

    function signed [63:0] f_min;
        input signed [63:0] x, y;
        f_min = (x < y) ? x : y;
    endfunction

    function signed [63:0] f_max;
        input signed [63:0] x, y;
        f_max = (x > y) ? x : y;
    endfunction

    wire signed [63:0] spo  = {{32{prev_out[31]}},   prev_out};
    wire signed [63:0] spb  = {{32{prev_back[31]}},  prev_back};
    wire signed [63:0] spb2 = {{32{prev_back2[31]}}, prev_back2};
    wire signed [63:0] spf2 = {{32{prev_front2[31]}},prev_front2};
    wire signed [63:0] sn   = {47'd0, n};

    wire mulw_live = (Fi >= -64'sd2) && (Fi <= 64'sd2);   // 1[-2,2](front)
    wire [63:0] acc_ror = ({acc} >> fsh) | ({acc} << (7'd64 - {1'b0, fsh}));

    // ------------------------------------------------------------------
    // the operation (raw, before ACC/SAT/truncation)
    // ------------------------------------------------------------------
    reg signed [63:0] res_op;
    reg               acc_own;      // op manages the accumulator itself
    reg signed [63:0] acc_op;       // its next accumulator value

    always @* begin
        res_op  = 64'sd0;
        acc_own = 1'b0;
        acc_op  = acc;
        case (klass)
        // ---------------- class 0: ALU, logic, shift, select ----------
        2'b00: case ({fn, opm})
            {4'b0000, 2'b00}: res_op = 64'sd0;                       // reserved
            {4'b0000, 2'b01}: res_op = sB;                           // MOV
            {4'b0000, 2'b10}: res_op = sB;                           // MOV
            {4'b0000, 2'b11}: res_op = sCb;                          // MOVI
            {4'b0001, 2'b00}: res_op = f_shr(sB + Fi);               // ADD
            {4'b0001, 2'b01}: res_op = f_shr(sB);                    // ASR
            {4'b0001, 2'b10}: res_op = f_shr(Fi - sB);               // RSB
            {4'b0001, 2'b11}: res_op = f_shr(sB);                    // ASR
            {4'b0010, 2'b00}: res_op = f_shr(sB + Fi);               // ADD alias
            {4'b0010, 2'b01}: res_op = f_shr(sB + sCb);              // ADDI
            {4'b0010, 2'b10}: res_op = f_shr(Fi - sB);               // RSB alias
            {4'b0010, 2'b11}: res_op = -f_shr(sB) + sCb;             // NASRI
            {4'b0011, 2'b00}: res_op = sB + Fi + sA;                 // ADDA
            {4'b0011, 2'b01}: res_op = (sB - Fi) + sA;               // SUBA
            {4'b0011, 2'b10}: res_op = Fi + sB;                      // ADDU
            {4'b0011, 2'b11}: res_op = Fi - sB;                      // RSBU
            {4'b0100, 2'b00}: res_op = sB + (Fi >>> bsh);            // ADDSF
            {4'b0100, 2'b01}: res_op = (sB >>> bsh) + sA;            // ASRA
            {4'b0100, 2'b10}: res_op = (Fi - sB) + sCb;              // RSBI
            {4'b0100, 2'b11}: res_op = -(sB >>> bsh) + sA;           // NASRA
            {4'b0101, 2'b00}: res_op = sB - (Fi >>> bsh);            // SUBSF
            {4'b0101, 2'b01}: res_op = (sB >>> bsh) - sA;            // ASRS
            {4'b0101, 2'b10}: res_op = (((Fi & 64'shFF00) + (sB & 64'shFF00)) >>> 8) <<< k; // ADDP1
            {4'b0101, 2'b11}: res_op = ((sB & 64'shFF) + (Fi & 64'shFF)) <<< k;             // ADDP0
            {4'b0110, 2'b00}: res_op = ((Fi & sA) != 0) ? -sB : sB;  // NEGF
            {4'b0110, 2'b01}: res_op = ((sB & sA) != 0) ? sB : -sB;  // NEGB
            {4'b0110, 2'b10}: res_op = ((Fi & sA) != 0) ? sCb : sB;  // SEL
            {4'b0110, 2'b11}: res_op = (((Fi & sA) != 0) ? sB : 64'sd0) + sCb; // SELZ
            {4'b0111, 2'b00}: res_op = (sB <<< k) + sCb;             // LSLB
            {4'b0111, 2'b01}: res_op = (sB - sCb) <<< k;             // SUBIL
            {4'b0111, 2'b10}: res_op = f_min(sB, Fi);                // MIN
            {4'b0111, 2'b11}: res_op = f_max(sCb, f_min(sA, sB));    // CLAMP
            {4'b1000, 2'b00}: res_op = mulw_live ?  (sB * Fi) : 64'sd0; // MULW
            {4'b1000, 2'b01}: res_op = mulw_live ? -(sB * Fi) : 64'sd0; // MULWN
            {4'b1000, 2'b10}: res_op = sCb;                          // MOVI
            {4'b1000, 2'b11}: res_op = ((sB & sA) != 0) ? sCb : 64'sd0; // TSTI
            {4'b1001, 2'b00}: res_op = sCb <<< Bsh;                  // LSLKB
            {4'b1001, 2'b01}: res_op = sCb >>> Bsh;                  // ASRKB
            {4'b1001, 2'b10}: res_op = sCb <<< Bsh;                  // LSLKB
            {4'b1001, 2'b11}: res_op = sCb >>> Bsh;                  // ASRKB
            {4'b1010, 2'b00}: res_op = sB <<< fsh;                   // LSLF
            {4'b1010, 2'b01}: res_op = sB <<< bsh;                   // LSLI
            {4'b1010, 2'b10}: begin                                  // ROR64
                res_op  = $signed(acc_ror);
                acc_own = 1'b1;
                acc_op  = $signed(acc_ror);
            end
            {4'b1010, 2'b11}: res_op = sB >>> bsh;                   // ASRI
            {4'b1011, 2'b00}: res_op = sB <<< fsh;                   // LSLF alias
            {4'b1011, 2'b01}: res_op = sB <<< bsh;                   // LSLI alias
            {4'b1011, 2'b10}: res_op = sB >>> fsh;                   // ASRF
            {4'b1011, 2'b11}: res_op = sB >>> bsh;                   // ASRI alias
            {4'b1100, 2'b00}: res_op = sB & Fi;                      // AND
            {4'b1100, 2'b01}: res_op = sB & sCb;                     // ANDI
            {4'b1100, 2'b10}: res_op = ~(Fi & sB);                   // NAND
            {4'b1100, 2'b11}: res_op = ~(sB & sCb);                  // NANDI
            {4'b1101, 2'b00}: res_op = sB | Fi;                      // ORR
            {4'b1101, 2'b01}: res_op = sB | sCb;                     // ORRI
            {4'b1101, 2'b10}: res_op = ~(Fi | sB);                   // NOR
            {4'b1101, 2'b11}: res_op = (~sB) & (~sCb);               // NORI
            {4'b1110, 2'b00}: res_op = sB ^ Fi;                      // EOR
            {4'b1110, 2'b01}: res_op = sB ^ sCb;                     // EORI
            {4'b1110, 2'b10}: res_op = ~f_abs(Fi - sB);              // NABSD
            {4'b1110, 2'b11}: res_op = (~sB) ^ sCb;                  // XNORI
            {4'b1111, 2'b00}: res_op = ~sB;                          // NOT
            {4'b1111, 2'b01}: res_op = (sB != 0) ? 64'sd1 : 64'sd0;  // TST
            {4'b1111, 2'b10}: res_op = {32'd0, par ^ back};          // PARITY
            {4'b1111, 2'b11}: res_op = ~sB;                          // NOT
            default:          res_op = 64'sd0;
        endcase
        // ---------------- class 1: extrema and absolute ---------------
        2'b01: case (fn)
            4'b0000: res_op = f_max(spo, Bn);                        // RMAX
            4'b0001: res_op = f_min(spo, Bn - Fi);                   // RMIND
            4'b0010: res_op = (n == 0) ? Bn                          // PCLAMP
                            : f_min(Bn, {{32{runmax[31]}}, runmax});
            4'b0011: res_op = f_max(spo, Bn - Fi);                   // RMAXD
            4'b0100: res_op = f_max(Bn, Fi);                         // MAX
            4'b0111: res_op = f_abs(Bn - sCb);                       // ABSI
            4'b1011: res_op = sCb;                                   // MOVI
            4'b1100: res_op = f_abs(Bn + Fi);                        // ABSS
            4'b1110: res_op = f_abs(Bn - Fi) + sA;                   // ABSDA
            4'b1111: res_op = f_abs(Fi - Bn) + sCb;                  // ABSDI
            default: res_op = Bn;                                    // MOV + aliases
        endcase
        // ---------------- class 2: multiply, MAC, IIR, SAD ------------
        2'b10: case (fn)
            4'b0000: res_op = f_shr(Bn * Fi);                        // MUL/I/N/NI
            4'b0001: begin                                           // MULD
                res_op = f_shr(Bn * spb);
            end
            4'b0010: res_op = f_shr(Bn * Fi);                        // VMUL
            4'b0011: res_op = -f_shr(Bn * Fi);                       // VMULN
            4'b0100: begin                                           // MACI
                res_op  = f_shr(acc + Bn * Fi + sCb);
                acc_own = 1'b1;
                acc_op  = acc + Bn * Fi;
            end
            4'b0101: res_op = (n == 0) ? f_shr(sCb)                  // ACCI
                            : f_shr(spo + Bn + sCb);
            4'b0110: res_op = f_shr(sA * sn + sCb);                  // RAMP
            4'b0111: res_op = f_shr(Bn * Fi + sCb);                  // VMULI
            4'b1000, 4'b1010:                                        // IIR
                     res_op = (n < 2) ? 64'sd0
                            : f_shr(spo + spf2 + spb2) + sA;
            4'b1001: begin                                           // SAD
                res_op  = acc + sCb;
                acc_own = 1'b1;
                acc_op  = acc + f_abs(Bn - Fi);
            end
            default: res_op = 64'sd0;                                // reserved
        endcase
        default: res_op = 64'sd0;                                    // class 3
        endcase
    end

    // ------------------------------------------------------------------
    // ACC mechanism, saturation, sample truncation
    // ------------------------------------------------------------------
    wire signed [63:0] res_acc = (accm == 2'b01 && !acc_own) ? (acc + res_op)
                                                             : res_op;
    wire signed [63:0] sat_hi  = (64'sd1 <<< (sat - 5'd1)) - 64'sd1;
    wire signed [63:0] sat_lo  = -(64'sd1 <<< (sat - 5'd1));
    wire signed [63:0] res_sat = (sat == 5'd0) ? res_acc :
                                 (res_acc > sat_hi) ? sat_hi :
                                 (res_acc < sat_lo) ? sat_lo : res_acc;
    wire [31:0] res_smp = {{8{res_sat[23]}}, res_sat[23:0]};

    always @(posedge clk) begin
        if (rst) begin
            result_c <= 32'd0; result_v_c <= 1'b0;
            acc <= 64'd0; prev_out <= 32'd0;
            prev_back <= 32'd0; prev_back2 <= 32'd0;
            prev_front <= 32'd0; prev_front2 <= 32'd0;
            runmax <= 32'd0; par <= 32'd0; n <= 17'd0;
        end else if (trigger) begin
            result_v_c  <= 1'b0;
            n           <= 17'd0;
            prev_out    <= 32'd0;
            prev_back   <= const_b;     // MULD: back[-1] = b
            prev_back2  <= 32'd0;
            prev_front  <= 32'd0;
            prev_front2 <= 32'd0;
            runmax      <= 32'd0;
            par         <= 32'd0;
            acc         <= (accm == 2'b10) ? sA : 64'd0;  // LOAD a / zero
        end else begin
            result_v_c <= en;
            if (en) begin
                result_c    <= res_smp;
                prev_out    <= res_smp;
                prev_back2  <= prev_back;
                prev_back   <= back;
                prev_front2 <= prev_front;
                prev_front  <= Fi[31:0];
                par         <= par ^ back;
                runmax      <= (n == 0) ? Bn[31:0]
                             : ($signed(Bn[31:0]) > $signed(runmax) ? Bn[31:0] : runmax);
                n           <= n + 17'd1;
                if (acc_own)              acc <= acc_op;
                else if (accm == 2'b01)   acc <= res_acc;   // HOLD
            end
        end
    end
endmodule
