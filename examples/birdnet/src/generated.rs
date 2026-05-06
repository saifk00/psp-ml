//! Generated inference module
#[allow(unused_imports)]
use psp_ml::kernels::naive::*;
#[allow(unused_imports)]
use psp_ml::kernels::*;
static mut ARENA: Aligned16<2525064usize> = Aligned16([0.0f32; 2525064usize]);
pub const OUTPUT_SIZE: usize = 6522usize;
pub fn forward(input: &[f32; 144000usize], output: &mut [f32; 6522usize]) {
    let t_169 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1usize,
        )
    };
    let t_170 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(4usize),
            144000usize,
        )
    };
    let t_171 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1usize,
        )
    };
    let t_172 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144004usize),
            1usize,
        )
    };
    let t_173 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144008usize),
            144000usize,
        )
    };
    let t_174 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_175 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144000usize),
            144000usize,
        )
    };
    let t_187 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_188 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288008usize),
            144000usize,
        )
    };
    let t_200 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(432008usize),
            1046528usize,
        )
    };
    let t_201 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1478536usize),
            1046528usize,
        )
    };
    let t_202 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            1046528usize,
        )
    };
    let t_206 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1334528usize),
            523775usize,
        )
    };
    let t_214 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            523775usize,
        )
    };
    let t_215 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_216 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    let t_217 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_218 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    let t_219 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_220 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    static mut T_221_BUF: Aligned16<49056usize> = Aligned16([0.0f32; 49056usize]);
    let t_221 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_221_BUF) as *mut f32,
            49056usize,
        )
    };
    let t_228 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_229 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144000usize),
            144000usize,
        )
    };
    let t_241 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            523264usize,
        )
    };
    let t_242 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(811264usize),
            523264usize,
        )
    };
    let t_243 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            523264usize,
        )
    };
    static mut T_247_BUF: Aligned16<262143usize> = Aligned16([0.0f32; 262143usize]);
    let t_247 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_247_BUF) as *mut f32,
            262143usize,
        )
    };
    static mut T_546_BUF: Aligned16<6522usize> = Aligned16([0.0f32; 6522usize]);
    let t_546 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_546_BUF) as *mut f32,
            6522usize,
        )
    };
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_7 = &tensor_data[T_7_OFFSET..T_7_OFFSET + T_7_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    let t_10 = &tensor_data[T_10_OFFSET..T_10_OFFSET + T_10_LEN];
    let t_11 = &tensor_data[T_11_OFFSET..T_11_OFFSET + T_11_LEN];
    let t_12 = &tensor_data[T_12_OFFSET..T_12_OFFSET + T_12_LEN];
    let t_13 = &tensor_data[T_13_OFFSET..T_13_OFFSET + T_13_LEN];
    let t_14 = &tensor_data[T_14_OFFSET..T_14_OFFSET + T_14_LEN];
    let t_15 = &tensor_data[T_15_OFFSET..T_15_OFFSET + T_15_LEN];
    let t_16 = &tensor_data[T_16_OFFSET..T_16_OFFSET + T_16_LEN];
    let t_17 = &tensor_data[T_17_OFFSET..T_17_OFFSET + T_17_LEN];
    let t_18 = &tensor_data[T_18_OFFSET..T_18_OFFSET + T_18_LEN];
    let t_19 = &tensor_data[T_19_OFFSET..T_19_OFFSET + T_19_LEN];
    let t_20 = &tensor_data[T_20_OFFSET..T_20_OFFSET + T_20_LEN];
    let t_21 = &tensor_data[T_21_OFFSET..T_21_OFFSET + T_21_LEN];
    let t_22 = &tensor_data[T_22_OFFSET..T_22_OFFSET + T_22_LEN];
    let t_23 = &tensor_data[T_23_OFFSET..T_23_OFFSET + T_23_LEN];
    let t_24 = &tensor_data[T_24_OFFSET..T_24_OFFSET + T_24_LEN];
    let t_25 = &tensor_data[T_25_OFFSET..T_25_OFFSET + T_25_LEN];
    let t_26 = &tensor_data[T_26_OFFSET..T_26_OFFSET + T_26_LEN];
    let t_27 = &tensor_data[T_27_OFFSET..T_27_OFFSET + T_27_LEN];
    let t_28 = &tensor_data[T_28_OFFSET..T_28_OFFSET + T_28_LEN];
    let t_29 = &tensor_data[T_29_OFFSET..T_29_OFFSET + T_29_LEN];
    let t_30 = &tensor_data[T_30_OFFSET..T_30_OFFSET + T_30_LEN];
    let t_31 = &tensor_data[T_31_OFFSET..T_31_OFFSET + T_31_LEN];
    let t_32 = &tensor_data[T_32_OFFSET..T_32_OFFSET + T_32_LEN];
    let t_33 = &tensor_data[T_33_OFFSET..T_33_OFFSET + T_33_LEN];
    let t_34 = &tensor_data[T_34_OFFSET..T_34_OFFSET + T_34_LEN];
    let t_35 = &tensor_data[T_35_OFFSET..T_35_OFFSET + T_35_LEN];
    let t_36 = &tensor_data[T_36_OFFSET..T_36_OFFSET + T_36_LEN];
    let t_37 = &tensor_data[T_37_OFFSET..T_37_OFFSET + T_37_LEN];
    let t_38 = &tensor_data[T_38_OFFSET..T_38_OFFSET + T_38_LEN];
    let t_39 = &tensor_data[T_39_OFFSET..T_39_OFFSET + T_39_LEN];
    let t_40 = &tensor_data[T_40_OFFSET..T_40_OFFSET + T_40_LEN];
    let t_41 = &tensor_data[T_41_OFFSET..T_41_OFFSET + T_41_LEN];
    let t_42 = &tensor_data[T_42_OFFSET..T_42_OFFSET + T_42_LEN];
    let t_43 = &tensor_data[T_43_OFFSET..T_43_OFFSET + T_43_LEN];
    let t_44 = &tensor_data[T_44_OFFSET..T_44_OFFSET + T_44_LEN];
    let t_45 = &tensor_data[T_45_OFFSET..T_45_OFFSET + T_45_LEN];
    let t_46 = &tensor_data[T_46_OFFSET..T_46_OFFSET + T_46_LEN];
    let t_47 = &tensor_data[T_47_OFFSET..T_47_OFFSET + T_47_LEN];
    let t_48 = &tensor_data[T_48_OFFSET..T_48_OFFSET + T_48_LEN];
    let t_49 = &tensor_data[T_49_OFFSET..T_49_OFFSET + T_49_LEN];
    let t_50 = &tensor_data[T_50_OFFSET..T_50_OFFSET + T_50_LEN];
    let t_51 = &tensor_data[T_51_OFFSET..T_51_OFFSET + T_51_LEN];
    let t_52 = &tensor_data[T_52_OFFSET..T_52_OFFSET + T_52_LEN];
    let t_53 = &tensor_data[T_53_OFFSET..T_53_OFFSET + T_53_LEN];
    let t_54 = &tensor_data[T_54_OFFSET..T_54_OFFSET + T_54_LEN];
    let t_55 = &tensor_data[T_55_OFFSET..T_55_OFFSET + T_55_LEN];
    let t_56 = &tensor_data[T_56_OFFSET..T_56_OFFSET + T_56_LEN];
    let t_57 = &tensor_data[T_57_OFFSET..T_57_OFFSET + T_57_LEN];
    let t_58 = &tensor_data[T_58_OFFSET..T_58_OFFSET + T_58_LEN];
    let t_59 = &tensor_data[T_59_OFFSET..T_59_OFFSET + T_59_LEN];
    let t_60 = &tensor_data[T_60_OFFSET..T_60_OFFSET + T_60_LEN];
    let t_61 = &tensor_data[T_61_OFFSET..T_61_OFFSET + T_61_LEN];
    let t_62 = &tensor_data[T_62_OFFSET..T_62_OFFSET + T_62_LEN];
    let t_63 = &tensor_data[T_63_OFFSET..T_63_OFFSET + T_63_LEN];
    let t_64 = &tensor_data[T_64_OFFSET..T_64_OFFSET + T_64_LEN];
    let t_65 = &tensor_data[T_65_OFFSET..T_65_OFFSET + T_65_LEN];
    let t_66 = &tensor_data[T_66_OFFSET..T_66_OFFSET + T_66_LEN];
    let t_67 = &tensor_data[T_67_OFFSET..T_67_OFFSET + T_67_LEN];
    let t_68 = &tensor_data[T_68_OFFSET..T_68_OFFSET + T_68_LEN];
    let t_69 = &tensor_data[T_69_OFFSET..T_69_OFFSET + T_69_LEN];
    let t_70 = &tensor_data[T_70_OFFSET..T_70_OFFSET + T_70_LEN];
    let t_71 = &tensor_data[T_71_OFFSET..T_71_OFFSET + T_71_LEN];
    let t_72 = &tensor_data[T_72_OFFSET..T_72_OFFSET + T_72_LEN];
    let t_73 = &tensor_data[T_73_OFFSET..T_73_OFFSET + T_73_LEN];
    let t_74 = &tensor_data[T_74_OFFSET..T_74_OFFSET + T_74_LEN];
    let t_75 = &tensor_data[T_75_OFFSET..T_75_OFFSET + T_75_LEN];
    let t_76 = &tensor_data[T_76_OFFSET..T_76_OFFSET + T_76_LEN];
    let t_77 = &tensor_data[T_77_OFFSET..T_77_OFFSET + T_77_LEN];
    let t_78 = &tensor_data[T_78_OFFSET..T_78_OFFSET + T_78_LEN];
    let t_79 = &tensor_data[T_79_OFFSET..T_79_OFFSET + T_79_LEN];
    let t_80 = &tensor_data[T_80_OFFSET..T_80_OFFSET + T_80_LEN];
    let t_81 = &tensor_data[T_81_OFFSET..T_81_OFFSET + T_81_LEN];
    let t_82 = &tensor_data[T_82_OFFSET..T_82_OFFSET + T_82_LEN];
    let t_83 = &tensor_data[T_83_OFFSET..T_83_OFFSET + T_83_LEN];
    let t_84 = &tensor_data[T_84_OFFSET..T_84_OFFSET + T_84_LEN];
    let t_85 = &tensor_data[T_85_OFFSET..T_85_OFFSET + T_85_LEN];
    let t_86 = &tensor_data[T_86_OFFSET..T_86_OFFSET + T_86_LEN];
    let t_87 = &tensor_data[T_87_OFFSET..T_87_OFFSET + T_87_LEN];
    let t_88 = &tensor_data[T_88_OFFSET..T_88_OFFSET + T_88_LEN];
    let t_89 = &tensor_data[T_89_OFFSET..T_89_OFFSET + T_89_LEN];
    let t_90 = &tensor_data[T_90_OFFSET..T_90_OFFSET + T_90_LEN];
    let t_91 = &tensor_data[T_91_OFFSET..T_91_OFFSET + T_91_LEN];
    let t_92 = &tensor_data[T_92_OFFSET..T_92_OFFSET + T_92_LEN];
    let t_93 = &tensor_data[T_93_OFFSET..T_93_OFFSET + T_93_LEN];
    let t_94 = &tensor_data[T_94_OFFSET..T_94_OFFSET + T_94_LEN];
    let t_95 = &tensor_data[T_95_OFFSET..T_95_OFFSET + T_95_LEN];
    let t_96 = &tensor_data[T_96_OFFSET..T_96_OFFSET + T_96_LEN];
    let t_97 = &tensor_data[T_97_OFFSET..T_97_OFFSET + T_97_LEN];
    let t_98 = &tensor_data[T_98_OFFSET..T_98_OFFSET + T_98_LEN];
    let t_99 = &tensor_data[T_99_OFFSET..T_99_OFFSET + T_99_LEN];
    let t_100 = &tensor_data[T_100_OFFSET..T_100_OFFSET + T_100_LEN];
    let t_101 = &tensor_data[T_101_OFFSET..T_101_OFFSET + T_101_LEN];
    let t_102 = &tensor_data[T_102_OFFSET..T_102_OFFSET + T_102_LEN];
    let t_103 = &tensor_data[T_103_OFFSET..T_103_OFFSET + T_103_LEN];
    let t_104 = &tensor_data[T_104_OFFSET..T_104_OFFSET + T_104_LEN];
    let t_105 = &tensor_data[T_105_OFFSET..T_105_OFFSET + T_105_LEN];
    let t_106 = &tensor_data[T_106_OFFSET..T_106_OFFSET + T_106_LEN];
    let t_107 = &tensor_data[T_107_OFFSET..T_107_OFFSET + T_107_LEN];
    let t_108 = &tensor_data[T_108_OFFSET..T_108_OFFSET + T_108_LEN];
    let t_109 = &tensor_data[T_109_OFFSET..T_109_OFFSET + T_109_LEN];
    let t_110 = &tensor_data[T_110_OFFSET..T_110_OFFSET + T_110_LEN];
    let t_111 = &tensor_data[T_111_OFFSET..T_111_OFFSET + T_111_LEN];
    let t_112 = &tensor_data[T_112_OFFSET..T_112_OFFSET + T_112_LEN];
    let t_113 = &tensor_data[T_113_OFFSET..T_113_OFFSET + T_113_LEN];
    let t_114 = &tensor_data[T_114_OFFSET..T_114_OFFSET + T_114_LEN];
    let t_115 = &tensor_data[T_115_OFFSET..T_115_OFFSET + T_115_LEN];
    let t_116 = &tensor_data[T_116_OFFSET..T_116_OFFSET + T_116_LEN];
    let t_117 = &tensor_data[T_117_OFFSET..T_117_OFFSET + T_117_LEN];
    let t_118 = &tensor_data[T_118_OFFSET..T_118_OFFSET + T_118_LEN];
    let t_119 = &tensor_data[T_119_OFFSET..T_119_OFFSET + T_119_LEN];
    let t_120 = &tensor_data[T_120_OFFSET..T_120_OFFSET + T_120_LEN];
    let t_121 = &tensor_data[T_121_OFFSET..T_121_OFFSET + T_121_LEN];
    let t_122 = &tensor_data[T_122_OFFSET..T_122_OFFSET + T_122_LEN];
    let t_126 = &tensor_data[T_126_OFFSET..T_126_OFFSET + T_126_LEN];
    let t_129 = &tensor_data[T_129_OFFSET..T_129_OFFSET + T_129_LEN];
    let t_133 = &tensor_data[T_133_OFFSET..T_133_OFFSET + T_133_LEN];
    let t_134 = &tensor_data[T_134_OFFSET..T_134_OFFSET + T_134_LEN];
    let t_135 = &tensor_data[T_135_OFFSET..T_135_OFFSET + T_135_LEN];
    let t_136 = &tensor_data[T_136_OFFSET..T_136_OFFSET + T_136_LEN];
    let t_142 = &tensor_data[T_142_OFFSET..T_142_OFFSET + T_142_LEN];
    let t_144 = &tensor_data[T_144_OFFSET..T_144_OFFSET + T_144_LEN];
    let t_145 = &tensor_data[T_145_OFFSET..T_145_OFFSET + T_145_LEN];
    let t_146 = &tensor_data[T_146_OFFSET..T_146_OFFSET + T_146_LEN];
    let t_150 = &tensor_data[T_150_OFFSET..T_150_OFFSET + T_150_LEN];
    let t_153 = &tensor_data[T_153_OFFSET..T_153_OFFSET + T_153_LEN];
    let t_156 = &tensor_data[T_156_OFFSET..T_156_OFFSET + T_156_LEN];
    let t_160 = &tensor_data[T_160_OFFSET..T_160_OFFSET + T_160_LEN];
    let t_161 = &tensor_data[T_161_OFFSET..T_161_OFFSET + T_161_LEN];
    let t_162 = &tensor_data[T_162_OFFSET..T_162_OFFSET + T_162_LEN];
    let t_163 = &tensor_data[T_163_OFFSET..T_163_OFFSET + T_163_LEN];
    let t_164 = &tensor_data[T_164_OFFSET..T_164_OFFSET + T_164_LEN];
    let t_165 = &tensor_data[T_165_OFFSET..T_165_OFFSET + T_165_LEN];
    let t_166 = &tensor_data[T_166_OFFSET..T_166_OFFSET + T_166_LEN];
    let t_167 = &tensor_data[T_167_OFFSET..T_167_OFFSET + T_167_LEN];
    let t_168 = &tensor_data[T_168_OFFSET..T_168_OFFSET + T_168_LEN];
    let t_185 = &tensor_data[T_185_OFFSET..T_185_OFFSET + T_185_LEN];
    let t_199 = &tensor_data[T_199_OFFSET..T_199_OFFSET + T_199_LEN];
    let t_226 = &tensor_data[T_226_OFFSET..T_226_OFFSET + T_226_LEN];
    let t_240 = &tensor_data[T_240_OFFSET..T_240_OFFSET + T_240_LEN];
    let t_549 = &tensor_data[T_549_OFFSET..T_549_OFFSET + T_549_LEN];
    let t_550 = &tensor_data[T_550_OFFSET..T_550_OFFSET + T_550_LEN];
    let t_551 = &tensor_data[T_551_OFFSET..T_551_OFFSET + T_551_LEN];
    let t_552 = &tensor_data[T_552_OFFSET..T_552_OFFSET + T_552_LEN];
    let t_553 = &tensor_data[T_553_OFFSET..T_553_OFFSET + T_553_LEN];
    let t_554 = &tensor_data[T_554_OFFSET..T_554_OFFSET + T_554_LEN];
    let t_555 = &tensor_data[T_555_OFFSET..T_555_OFFSET + T_555_LEN];
    let t_556 = &tensor_data[T_556_OFFSET..T_556_OFFSET + T_556_LEN];
    let t_557 = &tensor_data[T_557_OFFSET..T_557_OFFSET + T_557_LEN];
    let t_558 = &tensor_data[T_558_OFFSET..T_558_OFFSET + T_558_LEN];
    let t_559 = &tensor_data[T_559_OFFSET..T_559_OFFSET + T_559_LEN];
    let t_560 = &tensor_data[T_560_OFFSET..T_560_OFFSET + T_560_LEN];
    let t_561 = &tensor_data[T_561_OFFSET..T_561_OFFSET + T_561_LEN];
    let t_562 = &tensor_data[T_562_OFFSET..T_562_OFFSET + T_562_LEN];
    let t_563 = &tensor_data[T_563_OFFSET..T_563_OFFSET + T_563_LEN];
    let t_564 = &tensor_data[T_564_OFFSET..T_564_OFFSET + T_564_LEN];
    let t_565 = &tensor_data[T_565_OFFSET..T_565_OFFSET + T_565_LEN];
    let t_566 = &tensor_data[T_566_OFFSET..T_566_OFFSET + T_566_LEN];
    let t_567 = &tensor_data[T_567_OFFSET..T_567_OFFSET + T_567_LEN];
    let t_568 = &tensor_data[T_568_OFFSET..T_568_OFFSET + T_568_LEN];
    let t_569 = &tensor_data[T_569_OFFSET..T_569_OFFSET + T_569_LEN];
    reduce_min(input, t_169);
    binary_sub(input, t_169, t_170, 1usize);
    reduce_max(t_170, t_171);
    binary_add(t_171, t_150, t_172, 1usize);
    binary_div(t_170, t_172, t_173, 1usize);
    binary_sub(t_173, t_146, t_174, 1usize);
    binary_mul(t_174, t_144, t_175, 1usize);
    strided_slice(
        t_175,
        &[1usize, 144000usize],
        t_187,
        &[0i32, 0i32],
        &[1i32, 144000i32],
        &[1i32, 1i32],
        0i32,
        0i32,
        0i32,
    );
    reshape(t_187, t_188);
    gather(
        t_188,
        &[1usize, 72000usize, 2usize],
        unsafe { core::slice::from_raw_parts(t_199.as_ptr() as *const i32, 1024usize) },
        t_200,
        &[1usize, 511usize, 1024usize, 2usize],
        1usize,
    );
    reshape(t_200, t_201);
    binary_mul(t_201, t_129, t_202, 2048usize);
    let scratch_12_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            2048usize,
        )
    };
    rfft_pack(t_202, scratch_12_0, 2048usize);
    fft_butterfly_stage(scratch_12_0, t_549, 1024usize, 1usize);
    fft_butterfly_stage(scratch_12_0, t_550, 1024usize, 2usize);
    fft_butterfly_stage(scratch_12_0, t_551, 1024usize, 4usize);
    fft_butterfly_stage(scratch_12_0, t_552, 1024usize, 8usize);
    fft_butterfly_stage(scratch_12_0, t_553, 1024usize, 16usize);
    fft_butterfly_stage(scratch_12_0, t_554, 1024usize, 32usize);
    fft_butterfly_stage(scratch_12_0, t_555, 1024usize, 64usize);
    fft_butterfly_stage(scratch_12_0, t_556, 1024usize, 128usize);
    fft_butterfly_stage(scratch_12_0, t_557, 1024usize, 256usize);
    fft_butterfly_stage(scratch_12_0, t_558, 1024usize, 512usize);
    rfft_unpack(scratch_12_0, t_559, t_206, 2048usize);
    reshape(t_206, t_214);
    for _batch_idx in 0..511usize {
        let _in_off = _batch_idx * 1025usize;
        let _out_off = _batch_idx * 96usize;
        fully_connected(
            &t_214[_in_off.._in_off + 1025usize],
            1025usize,
            t_166,
            None,
            &mut t_215[_out_off.._out_off + 96usize],
            96usize,
        );
    }
    reshape(t_215, t_216);
    binary_mul(t_216, t_216, t_217, 49056usize);
    binary_pow(t_217, t_165, t_218, 1usize);
    reverse_v2(t_218, &[511usize, 1usize, 96usize], t_219, 2usize);
    transpose(
        t_219,
        &[511usize, 1usize, 96usize],
        t_220,
        &[511usize, 96usize, 1usize],
        &[0usize, 2usize, 1usize],
    );
    reshape(t_220, t_221);
    strided_slice(
        t_175,
        &[1usize, 144000usize],
        t_228,
        &[0i32, 0i32],
        &[1i32, 144000i32],
        &[1i32, 1i32],
        0i32,
        0i32,
        0i32,
    );
    reshape(t_228, t_229);
    gather(
        t_229,
        &[1usize, 18000usize, 8usize],
        unsafe { core::slice::from_raw_parts(t_240.as_ptr() as *const i32, 511usize) },
        t_241,
        &[1usize, 511usize, 128usize, 8usize],
        1usize,
    );
    reshape(t_241, t_242);
    binary_mul(t_242, t_126, t_243, 1024usize);
    let scratch_26_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(523264usize),
            1024usize,
        )
    };
    rfft_pack(t_243, scratch_26_0, 1024usize);
    fft_butterfly_stage(scratch_26_0, t_560, 512usize, 1usize);
    fft_butterfly_stage(scratch_26_0, t_561, 512usize, 2usize);
    fft_butterfly_stage(scratch_26_0, t_562, 512usize, 4usize);
    fft_butterfly_stage(scratch_26_0, t_563, 512usize, 8usize);
    fft_butterfly_stage(scratch_26_0, t_564, 512usize, 16usize);
    fft_butterfly_stage(scratch_26_0, t_565, 512usize, 32usize);
    fft_butterfly_stage(scratch_26_0, t_566, 512usize, 64usize);
    fft_butterfly_stage(scratch_26_0, t_567, 512usize, 128usize);
    fft_butterfly_stage(scratch_26_0, t_568, 512usize, 256usize);
    rfft_unpack(scratch_26_0, t_569, t_247, 1024usize);
    for _frame_idx in 0..511usize {
        let t_255 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(661880usize),
                513usize,
            )
        };
        let t_256 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667580usize),
                96usize,
            )
        };
        let t_257 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667676usize),
                96usize,
            )
        };
        let t_258 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667772usize),
                96usize,
            )
        };
        let t_259 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667868usize),
                96usize,
            )
        };
        let t_260 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667964usize),
                96usize,
            )
        };
        let t_261 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668060usize),
                96usize,
            )
        };
        let t_262 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668156usize),
                96usize,
            )
        };
        let t_263 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667004usize),
                192usize,
            )
        };
        let t_264 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667196usize),
                192usize,
            )
        };
        let t_265 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667388usize),
                192usize,
            )
        };
        let t_266 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(616560usize),
                1152usize,
            )
        };
        let t_267 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(617712usize),
                1152usize,
            )
        };
        let t_268 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(618864usize),
                1152usize,
            )
        };
        let t_269 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(563760usize),
                2304usize,
            )
        };
        let t_270 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(620016usize),
                1152usize,
            )
        };
        let t_271 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(466992usize),
                3456usize,
            )
        };
        let t_272 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(470448usize),
                3456usize,
            )
        };
        let t_273 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(473904usize),
                3456usize,
            )
        };
        let t_274 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(95616usize),
                10800usize,
            )
        };
        let t_275 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(566064usize),
                1728usize,
            )
        };
        let t_276 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(567792usize),
                1728usize,
            )
        };
        let t_277 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(569520usize),
                1728usize,
            )
        };
        let t_278 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(623216usize),
                864usize,
            )
        };
        let t_279 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(571248usize),
                1728usize,
            )
        };
        let t_280 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(572976usize),
                1728usize,
            )
        };
        let t_281 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(574704usize),
                1728usize,
            )
        };
        let t_282 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(576432usize),
                1728usize,
            )
        };
        let t_283 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(578160usize),
                1728usize,
            )
        };
        let t_284 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(579888usize),
                1728usize,
            )
        };
        let t_285 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(624080usize),
                864usize,
            )
        };
        let t_286 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(624944usize),
                864usize,
            )
        };
        let t_287 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(581616usize),
                1728usize,
            )
        };
        let t_288 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(583344usize),
                1728usize,
            )
        };
        let t_289 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(585072usize),
                1728usize,
            )
        };
        let t_290 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(586800usize),
                1728usize,
            )
        };
        let t_291 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(588528usize),
                1728usize,
            )
        };
        let t_292 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(590256usize),
                1728usize,
            )
        };
        let t_293 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(625808usize),
                864usize,
            )
        };
        let t_294 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(626672usize),
                864usize,
            )
        };
        let t_295 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(165168usize),
                6912usize,
            )
        };
        let t_296 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(172080usize),
                6912usize,
            )
        };
        let t_297 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(178992usize),
                6912usize,
            )
        };
        let t_298 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(73152usize),
                22464usize,
            )
        };
        let t_299 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(477360usize),
                3456usize,
            )
        };
        let t_300 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(480816usize),
                3456usize,
            )
        };
        let t_301 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(484272usize),
                3456usize,
            )
        };
        let t_302 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662396usize),
                288usize,
            )
        };
        let t_306 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662684usize),
                288usize,
            )
        };
        let t_307 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669248usize),
                18usize,
            )
        };
        let t_308 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669268usize),
                18usize,
            )
        };
        let t_309 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669288usize),
                18usize,
            )
        };
        let t_310 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662972usize),
                288usize,
            )
        };
        let t_311 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663260usize),
                288usize,
            )
        };
        let t_312 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(487728usize),
                3456usize,
            )
        };
        let t_313 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(627536usize),
                864usize,
            )
        };
        let t_314 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(491184usize),
                3456usize,
            )
        };
        let t_315 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(494640usize),
                3456usize,
            )
        };
        let t_316 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(498096usize),
                3456usize,
            )
        };
        let t_317 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(501552usize),
                3456usize,
            )
        };
        let t_318 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(505008usize),
                3456usize,
            )
        };
        let t_319 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(508464usize),
                3456usize,
            )
        };
        let t_320 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663548usize),
                288usize,
            )
        };
        let t_324 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663836usize),
                288usize,
            )
        };
        let t_325 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669308usize),
                18usize,
            )
        };
        let t_326 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669328usize),
                18usize,
            )
        };
        let t_327 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669348usize),
                18usize,
            )
        };
        let t_328 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664124usize),
                288usize,
            )
        };
        let t_329 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664412usize),
                288usize,
            )
        };
        let t_330 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(511920usize),
                3456usize,
            )
        };
        let t_331 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(628400usize),
                864usize,
            )
        };
        let t_332 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(629264usize),
                864usize,
            )
        };
        let t_333 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(515376usize),
                3456usize,
            )
        };
        let t_334 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(518832usize),
                3456usize,
            )
        };
        let t_335 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(522288usize),
                3456usize,
            )
        };
        let t_336 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(525744usize),
                3456usize,
            )
        };
        let t_337 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(529200usize),
                3456usize,
            )
        };
        let t_338 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(532656usize),
                3456usize,
            )
        };
        let t_339 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664700usize),
                288usize,
            )
        };
        let t_343 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664988usize),
                288usize,
            )
        };
        let t_344 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669368usize),
                18usize,
            )
        };
        let t_345 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669388usize),
                18usize,
            )
        };
        let t_346 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669408usize),
                18usize,
            )
        };
        let t_347 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665276usize),
                288usize,
            )
        };
        let t_348 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665564usize),
                288usize,
            )
        };
        let t_349 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(536112usize),
                3456usize,
            )
        };
        let t_350 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(630128usize),
                864usize,
            )
        };
        let t_351 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(630992usize),
                864usize,
            )
        };
        let t_352 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(539568usize),
                3456usize,
            )
        };
        let t_353 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(543024usize),
                3456usize,
            )
        };
        let t_354 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(546480usize),
                3456usize,
            )
        };
        let t_355 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(549936usize),
                3456usize,
            )
        };
        let t_356 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(553392usize),
                3456usize,
            )
        };
        let t_357 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(556848usize),
                3456usize,
            )
        };
        let t_358 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665852usize),
                288usize,
            )
        };
        let t_362 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666140usize),
                288usize,
            )
        };
        let t_363 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669428usize),
                18usize,
            )
        };
        let t_364 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669448usize),
                18usize,
            )
        };
        let t_365 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669468usize),
                18usize,
            )
        };
        let t_366 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666428usize),
                288usize,
            )
        };
        let t_367 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666716usize),
                288usize,
            )
        };
        let t_368 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(560304usize),
                3456usize,
            )
        };
        let t_369 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(631856usize),
                864usize,
            )
        };
        let t_370 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(632720usize),
                864usize,
            )
        };
        let t_371 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(106416usize),
                10368usize,
            )
        };
        let t_372 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(116784usize),
                10368usize,
            )
        };
        let t_373 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(127152usize),
                10368usize,
            )
        };
        let t_374 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(36864usize),
                36288usize,
            )
        };
        let t_375 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(185904usize),
                5184usize,
            )
        };
        let t_376 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(191088usize),
                5184usize,
            )
        };
        let t_377 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(196272usize),
                5184usize,
            )
        };
        let t_378 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(633584usize),
                864usize,
            )
        };
        let t_382 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(634448usize),
                864usize,
            )
        };
        let t_383 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668828usize),
                27usize,
            )
        };
        let t_384 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668856usize),
                27usize,
            )
        };
        let t_385 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668884usize),
                27usize,
            )
        };
        let t_386 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(635312usize),
                864usize,
            )
        };
        let t_387 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(636176usize),
                864usize,
            )
        };
        let t_388 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(201456usize),
                5184usize,
            )
        };
        let t_389 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(650864usize),
                648usize,
            )
        };
        let t_390 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(206640usize),
                5184usize,
            )
        };
        let t_391 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(211824usize),
                5184usize,
            )
        };
        let t_392 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(217008usize),
                5184usize,
            )
        };
        let t_393 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(222192usize),
                5184usize,
            )
        };
        let t_394 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(227376usize),
                5184usize,
            )
        };
        let t_395 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(232560usize),
                5184usize,
            )
        };
        let t_396 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(637040usize),
                864usize,
            )
        };
        let t_400 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(637904usize),
                864usize,
            )
        };
        let t_401 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668912usize),
                27usize,
            )
        };
        let t_402 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668940usize),
                27usize,
            )
        };
        let t_403 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668968usize),
                27usize,
            )
        };
        let t_404 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(638768usize),
                864usize,
            )
        };
        let t_405 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(639632usize),
                864usize,
            )
        };
        let t_406 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(237744usize),
                5184usize,
            )
        };
        let t_407 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(651512usize),
                648usize,
            )
        };
        let t_408 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(652160usize),
                648usize,
            )
        };
        let t_409 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(242928usize),
                5184usize,
            )
        };
        let t_410 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(248112usize),
                5184usize,
            )
        };
        let t_411 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(253296usize),
                5184usize,
            )
        };
        let t_412 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(258480usize),
                5184usize,
            )
        };
        let t_413 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(263664usize),
                5184usize,
            )
        };
        let t_414 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(268848usize),
                5184usize,
            )
        };
        let t_415 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(640496usize),
                864usize,
            )
        };
        let t_419 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(641360usize),
                864usize,
            )
        };
        let t_420 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668996usize),
                27usize,
            )
        };
        let t_421 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669024usize),
                27usize,
            )
        };
        let t_422 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669052usize),
                27usize,
            )
        };
        let t_423 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(642224usize),
                864usize,
            )
        };
        let t_424 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(643088usize),
                864usize,
            )
        };
        let t_425 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(274032usize),
                5184usize,
            )
        };
        let t_426 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(652808usize),
                648usize,
            )
        };
        let t_427 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(653456usize),
                648usize,
            )
        };
        let t_428 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(279216usize),
                5184usize,
            )
        };
        let t_429 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(284400usize),
                5184usize,
            )
        };
        let t_430 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(289584usize),
                5184usize,
            )
        };
        let t_431 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(294768usize),
                5184usize,
            )
        };
        let t_432 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(299952usize),
                5184usize,
            )
        };
        let t_433 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(305136usize),
                5184usize,
            )
        };
        let t_434 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(643952usize),
                864usize,
            )
        };
        let t_438 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(644816usize),
                864usize,
            )
        };
        let t_439 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669080usize),
                27usize,
            )
        };
        let t_440 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669108usize),
                27usize,
            )
        };
        let t_441 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669136usize),
                27usize,
            )
        };
        let t_442 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(645680usize),
                864usize,
            )
        };
        let t_443 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(646544usize),
                864usize,
            )
        };
        let t_444 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(310320usize),
                5184usize,
            )
        };
        let t_445 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(654104usize),
                648usize,
            )
        };
        let t_446 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(654752usize),
                648usize,
            )
        };
        let t_447 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(315504usize),
                5184usize,
            )
        };
        let t_448 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(320688usize),
                5184usize,
            )
        };
        let t_449 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(325872usize),
                5184usize,
            )
        };
        let t_450 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(331056usize),
                5184usize,
            )
        };
        let t_451 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(336240usize),
                5184usize,
            )
        };
        let t_452 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(341424usize),
                5184usize,
            )
        };
        let t_453 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(647408usize),
                864usize,
            )
        };
        let t_457 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(648272usize),
                864usize,
            )
        };
        let t_458 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669164usize),
                27usize,
            )
        };
        let t_459 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669192usize),
                27usize,
            )
        };
        let t_460 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669220usize),
                27usize,
            )
        };
        let t_461 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(649136usize),
                864usize,
            )
        };
        let t_462 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(650000usize),
                864usize,
            )
        };
        let t_463 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(346608usize),
                5184usize,
            )
        };
        let t_464 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(655400usize),
                648usize,
            )
        };
        let t_465 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(656048usize),
                648usize,
            )
        };
        let t_466 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(137520usize),
                9216usize,
            )
        };
        let t_467 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(146736usize),
                9216usize,
            )
        };
        let t_468 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(155952usize),
                9216usize,
            )
        };
        let t_469 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
                36864usize,
            )
        };
        let t_470 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(351792usize),
                4608usize,
            )
        };
        let t_471 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(356400usize),
                4608usize,
            )
        };
        let t_472 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(361008usize),
                4608usize,
            )
        };
        let t_473 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(591984usize),
                1536usize,
            )
        };
        let t_477 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(593520usize),
                1536usize,
            )
        };
        let t_478 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668252usize),
                48usize,
            )
        };
        let t_479 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668300usize),
                48usize,
            )
        };
        let t_480 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668348usize),
                48usize,
            )
        };
        let t_481 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(595056usize),
                1536usize,
            )
        };
        let t_482 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(596592usize),
                1536usize,
            )
        };
        let t_483 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(365616usize),
                4608usize,
            )
        };
        let t_484 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(656696usize),
                576usize,
            )
        };
        let t_485 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(370224usize),
                4608usize,
            )
        };
        let t_486 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(374832usize),
                4608usize,
            )
        };
        let t_487 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(379440usize),
                4608usize,
            )
        };
        let t_488 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(384048usize),
                4608usize,
            )
        };
        let t_489 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(388656usize),
                4608usize,
            )
        };
        let t_490 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(393264usize),
                4608usize,
            )
        };
        let t_491 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(598128usize),
                1536usize,
            )
        };
        let t_495 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(599664usize),
                1536usize,
            )
        };
        let t_496 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668396usize),
                48usize,
            )
        };
        let t_497 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668444usize),
                48usize,
            )
        };
        let t_498 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668492usize),
                48usize,
            )
        };
        let t_499 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(601200usize),
                1536usize,
            )
        };
        let t_500 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(602736usize),
                1536usize,
            )
        };
        let t_501 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(397872usize),
                4608usize,
            )
        };
        let t_502 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(657272usize),
                576usize,
            )
        };
        let t_503 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(657848usize),
                576usize,
            )
        };
        let t_504 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(402480usize),
                4608usize,
            )
        };
        let t_505 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(407088usize),
                4608usize,
            )
        };
        let t_506 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(411696usize),
                4608usize,
            )
        };
        let t_507 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(416304usize),
                4608usize,
            )
        };
        let t_508 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(420912usize),
                4608usize,
            )
        };
        let t_509 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(425520usize),
                4608usize,
            )
        };
        let t_510 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(604272usize),
                1536usize,
            )
        };
        let t_514 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(605808usize),
                1536usize,
            )
        };
        let t_515 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668540usize),
                48usize,
            )
        };
        let t_516 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668588usize),
                48usize,
            )
        };
        let t_517 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668636usize),
                48usize,
            )
        };
        let t_518 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(607344usize),
                1536usize,
            )
        };
        let t_519 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(608880usize),
                1536usize,
            )
        };
        let t_520 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(430128usize),
                4608usize,
            )
        };
        let t_521 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(658424usize),
                576usize,
            )
        };
        let t_522 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(659000usize),
                576usize,
            )
        };
        let t_523 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(434736usize),
                4608usize,
            )
        };
        let t_524 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(439344usize),
                4608usize,
            )
        };
        let t_525 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(443952usize),
                4608usize,
            )
        };
        let t_526 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(448560usize),
                4608usize,
            )
        };
        let t_527 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(453168usize),
                4608usize,
            )
        };
        let t_528 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(457776usize),
                4608usize,
            )
        };
        let t_529 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(610416usize),
                1536usize,
            )
        };
        let t_533 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(611952usize),
                1536usize,
            )
        };
        let t_534 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668684usize),
                48usize,
            )
        };
        let t_535 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668732usize),
                48usize,
            )
        };
        let t_536 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668780usize),
                48usize,
            )
        };
        let t_537 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(613488usize),
                1536usize,
            )
        };
        let t_538 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(615024usize),
                1536usize,
            )
        };
        let t_539 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(462384usize),
                4608usize,
            )
        };
        let t_540 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(659576usize),
                576usize,
            )
        };
        let t_541 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(660152usize),
                576usize,
            )
        };
        let t_542 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(660728usize),
                576usize,
            )
        };
        let t_543 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(661304usize),
                576usize,
            )
        };
        let t_544 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(621168usize),
                1024usize,
            )
        };
        let t_545 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(622192usize),
                1024usize,
            )
        };
        let t_547 = &t_247[_frame_idx * 513usize..(_frame_idx + 1) * 513usize];
        let t_548 = &t_221[_frame_idx * 96usize..(_frame_idx + 1) * 96usize];
        reshape(t_547, t_255);
        fully_connected(t_255, 513usize, t_168, None, t_256, 96usize);
        reshape(t_256, t_257);
        binary_mul(t_257, t_257, t_258, 96usize);
        binary_pow(t_258, t_167, t_259, 1usize);
        reverse_v2(t_259, &[1usize, 1usize, 96usize], t_260, 2usize);
        transpose(
            t_260,
            &[1usize, 1usize, 96usize],
            t_261,
            &[1usize, 96usize, 1usize],
            &[0usize, 2usize, 1usize],
        );
        reshape(t_261, t_262);
        {
            let src = t_548;
            for p in 0..96usize {
                for a in 0..1usize {
                    let src_off = p * (1usize * 1usize) + a * 1usize;
                    let dst_off = p * (2usize * 1usize) + (0usize + a) * 1usize;
                    t_263[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        {
            let src = t_262;
            for p in 0..96usize {
                for a in 0..1usize {
                    let src_off = p * (1usize * 1usize) + a * 1usize;
                    let dst_off = p * (2usize * 1usize) + (1usize + a) * 1usize;
                    t_263[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        binary_mul(t_263, t_163, t_264, 2usize);
        binary_add(t_264, t_162, t_265, 2usize);
        let scratch_38_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                3072usize,
            )
        };
        let scratch_38_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672560usize),
                1536usize,
            )
        };
        scratch_38_1.copy_from_slice(t_122);
        im2col_padded(
            t_265,
            [1usize, 96usize, 1usize, 2usize],
            [4usize, 8usize],
            [2usize, 2usize],
            [1usize, 1usize, 3usize, 4usize],
            [48usize, 1usize],
            scratch_38_0,
        );
        matmul_bt_tiled(scratch_38_0, scratch_38_1, t_266, 12usize, 16usize, 6usize);
        bias_add(t_266, t_51, 48usize, 24usize);
        relu(t_266);
        average_pool2d(
            t_266,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 2usize],
            [1usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_267,
            [1usize, 48usize, 1usize, 24usize],
        );
        max_pool2d(
            t_266,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 2usize],
            [1usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_268,
            [1usize, 48usize, 1usize, 24usize],
        );
        {
            let src = t_268;
            for p in 0..48usize {
                for a in 0..24usize {
                    let src_off = p * (24usize * 1usize) + a * 1usize;
                    let dst_off = p * (48usize * 1usize) + (0usize + a) * 1usize;
                    t_269[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        {
            let src = t_267;
            for p in 0..48usize {
                for a in 0..24usize {
                    let src_off = p * (24usize * 1usize) + a * 1usize;
                    let dst_off = p * (48usize * 1usize) + (24usize + a) * 1usize;
                    t_269[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        let scratch_42_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2304usize,
            )
        };
        let scratch_42_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(671792usize),
                1152usize,
            )
        };
        scratch_42_1.copy_from_slice(t_121);
        im2col_padded(
            t_269,
            [1usize, 48usize, 1usize, 48usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [48usize, 1usize],
            scratch_42_0,
        );
        matmul_bt_tiled(scratch_42_0, scratch_42_1, t_270, 12usize, 12usize, 6usize);
        bias_add(t_270, t_50, 48usize, 24usize);
        let scratch_43_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(671216usize),
                1152usize,
            )
        };
        let scratch_43_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                1728usize,
            )
        };
        scratch_43_1.copy_from_slice(t_120);
        im2col_padded(
            t_270,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [48usize, 1usize],
            scratch_43_0,
        );
        matmul_bt_tiled(scratch_43_0, scratch_43_1, t_271, 12usize, 6usize, 18usize);
        bias_add(t_271, t_49, 48usize, 72usize);
        unary_logistic(t_271, t_272);
        binary_mul(t_271, t_272, t_273, 3456usize);
        pad(
            t_273,
            [1usize, 48usize, 1usize, 72usize],
            t_274,
            [1usize, 50usize, 3usize, 72usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        depthwise_conv2d(
            t_274,
            [1usize, 50usize, 3usize, 72usize],
            t_48,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_47),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_275,
            [1usize, 24usize, 1usize, 72usize],
        );
        unary_logistic(t_275, t_276);
        binary_mul(t_275, t_276, t_277, 1728usize);
        let scratch_50_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_50_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_50_1.copy_from_slice(t_118);
        im2col_padded(
            t_277,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_50_0,
        );
        matmul_bt_tiled(scratch_50_0, scratch_50_1, t_278, 6usize, 18usize, 9usize);
        bias_add(t_278, t_117, 24usize, 36usize);
        let scratch_51_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                864usize,
            )
        };
        let scratch_51_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_51_1.copy_from_slice(t_116);
        im2col_padded(
            t_278,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_51_0,
        );
        matmul_bt_tiled(scratch_51_0, scratch_51_1, t_279, 6usize, 9usize, 18usize);
        bias_add(t_279, t_46, 24usize, 72usize);
        unary_logistic(t_279, t_280);
        binary_mul(t_279, t_280, t_281, 1728usize);
        depthwise_conv2d(
            t_281,
            [1usize, 24usize, 1usize, 72usize],
            t_45,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_44),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_282,
            [1usize, 24usize, 1usize, 72usize],
        );
        unary_logistic(t_282, t_283);
        binary_mul(t_282, t_283, t_284, 1728usize);
        let scratch_57_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_57_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_57_1.copy_from_slice(t_115);
        im2col_padded(
            t_284,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_57_0,
        );
        matmul_bt_tiled(scratch_57_0, scratch_57_1, t_285, 6usize, 18usize, 9usize);
        bias_add(t_285, t_117, 24usize, 36usize);
        binary_add(t_285, t_278, t_286, 864usize);
        let scratch_59_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                864usize,
            )
        };
        let scratch_59_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_59_1.copy_from_slice(t_114);
        im2col_padded(
            t_286,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_59_0,
        );
        matmul_bt_tiled(scratch_59_0, scratch_59_1, t_287, 6usize, 9usize, 18usize);
        bias_add(t_287, t_43, 24usize, 72usize);
        unary_logistic(t_287, t_288);
        binary_mul(t_287, t_288, t_289, 1728usize);
        depthwise_conv2d(
            t_289,
            [1usize, 24usize, 1usize, 72usize],
            t_42,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_41),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_290,
            [1usize, 24usize, 1usize, 72usize],
        );
        unary_logistic(t_290, t_291);
        binary_mul(t_290, t_291, t_292, 1728usize);
        let scratch_65_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_65_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_65_1.copy_from_slice(t_113);
        im2col_padded(
            t_292,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_65_0,
        );
        matmul_bt_tiled(scratch_65_0, scratch_65_1, t_293, 6usize, 18usize, 9usize);
        bias_add(t_293, t_117, 24usize, 36usize);
        binary_add(t_293, t_286, t_294, 864usize);
        let scratch_67_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(679856usize),
                864usize,
            )
        };
        let scratch_67_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                10368usize,
            )
        };
        scratch_67_1.copy_from_slice(t_112);
        im2col_padded(
            t_294,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_67_0,
        );
        matmul_bt_tiled(scratch_67_0, scratch_67_1, t_295, 6usize, 9usize, 72usize);
        bias_add(t_295, t_40, 24usize, 288usize);
        unary_logistic(t_295, t_296);
        binary_mul(t_295, t_296, t_297, 6912usize);
        pad(
            t_297,
            [1usize, 24usize, 1usize, 288usize],
            t_298,
            [1usize, 26usize, 3usize, 288usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        depthwise_conv2d(
            t_298,
            [1usize, 26usize, 3usize, 288usize],
            t_39,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_38),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_299,
            [1usize, 12usize, 1usize, 288usize],
        );
        unary_logistic(t_299, t_300);
        binary_mul(t_299, t_300, t_301, 3456usize);
        reduce_mean_hw(t_301, t_302);
        reshape(t_302, t_306);
        conv2d(
            t_306,
            [1usize, 1usize, 1usize, 288usize],
            t_110,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_307,
            [1usize, 1usize, 1usize, 18usize],
        );
        unary_logistic(t_307, t_308);
        binary_mul(t_307, t_308, t_309, 18usize);
        conv2d(
            t_309,
            [1usize, 1usize, 1usize, 18usize],
            t_108,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_310,
            [1usize, 1usize, 1usize, 288usize],
        );
        unary_logistic(t_310, t_311);
        binary_mul(t_301, t_311, t_312, 288usize);
        let scratch_82_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_82_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_82_1.copy_from_slice(t_107);
        im2col_padded(
            t_312,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_82_0,
        );
        matmul_bt_tiled(scratch_82_0, scratch_82_1, t_313, 3usize, 72usize, 18usize);
        bias_add(t_313, t_119, 12usize, 72usize);
        let scratch_83_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_83_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_83_1.copy_from_slice(t_106);
        im2col_padded(
            t_313,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_83_0,
        );
        matmul_bt_tiled(scratch_83_0, scratch_83_1, t_314, 3usize, 18usize, 72usize);
        bias_add(t_314, t_37, 12usize, 288usize);
        unary_logistic(t_314, t_315);
        binary_mul(t_314, t_315, t_316, 3456usize);
        depthwise_conv2d(
            t_316,
            [1usize, 12usize, 1usize, 288usize],
            t_36,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_35),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_317,
            [1usize, 12usize, 1usize, 288usize],
        );
        unary_logistic(t_317, t_318);
        binary_mul(t_317, t_318, t_319, 3456usize);
        reduce_mean_hw(t_319, t_320);
        reshape(t_320, t_324);
        conv2d(
            t_324,
            [1usize, 1usize, 1usize, 288usize],
            t_105,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_325,
            [1usize, 1usize, 1usize, 18usize],
        );
        unary_logistic(t_325, t_326);
        binary_mul(t_325, t_326, t_327, 18usize);
        conv2d(
            t_327,
            [1usize, 1usize, 1usize, 18usize],
            t_104,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_328,
            [1usize, 1usize, 1usize, 288usize],
        );
        unary_logistic(t_328, t_329);
        binary_mul(t_319, t_329, t_330, 288usize);
        let scratch_97_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_97_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_97_1.copy_from_slice(t_103);
        im2col_padded(
            t_330,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_97_0,
        );
        matmul_bt_tiled(scratch_97_0, scratch_97_1, t_331, 3usize, 72usize, 18usize);
        bias_add(t_331, t_119, 12usize, 72usize);
        binary_add(t_331, t_313, t_332, 864usize);
        let scratch_99_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_99_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_99_1.copy_from_slice(t_102);
        im2col_padded(
            t_332,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_99_0,
        );
        matmul_bt_tiled(scratch_99_0, scratch_99_1, t_333, 3usize, 18usize, 72usize);
        bias_add(t_333, t_34, 12usize, 288usize);
        unary_logistic(t_333, t_334);
        binary_mul(t_333, t_334, t_335, 3456usize);
        depthwise_conv2d(
            t_335,
            [1usize, 12usize, 1usize, 288usize],
            t_33,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_32),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_336,
            [1usize, 12usize, 1usize, 288usize],
        );
        unary_logistic(t_336, t_337);
        binary_mul(t_336, t_337, t_338, 3456usize);
        reduce_mean_hw(t_338, t_339);
        reshape(t_339, t_343);
        conv2d(
            t_343,
            [1usize, 1usize, 1usize, 288usize],
            t_101,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_344,
            [1usize, 1usize, 1usize, 18usize],
        );
        unary_logistic(t_344, t_345);
        binary_mul(t_344, t_345, t_346, 18usize);
        conv2d(
            t_346,
            [1usize, 1usize, 1usize, 18usize],
            t_100,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_347,
            [1usize, 1usize, 1usize, 288usize],
        );
        unary_logistic(t_347, t_348);
        binary_mul(t_338, t_348, t_349, 288usize);
        let scratch_113_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_113_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_113_1.copy_from_slice(t_99);
        im2col_padded(
            t_349,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_113_0,
        );
        matmul_bt_tiled(scratch_113_0, scratch_113_1, t_350, 3usize, 72usize, 18usize);
        bias_add(t_350, t_119, 12usize, 72usize);
        binary_add(t_350, t_332, t_351, 864usize);
        let scratch_115_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_115_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_115_1.copy_from_slice(t_98);
        im2col_padded(
            t_351,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_115_0,
        );
        matmul_bt_tiled(scratch_115_0, scratch_115_1, t_352, 3usize, 18usize, 72usize);
        bias_add(t_352, t_31, 12usize, 288usize);
        unary_logistic(t_352, t_353);
        binary_mul(t_352, t_353, t_354, 3456usize);
        depthwise_conv2d(
            t_354,
            [1usize, 12usize, 1usize, 288usize],
            t_30,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_29),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_355,
            [1usize, 12usize, 1usize, 288usize],
        );
        unary_logistic(t_355, t_356);
        binary_mul(t_355, t_356, t_357, 3456usize);
        reduce_mean_hw(t_357, t_358);
        reshape(t_358, t_362);
        conv2d(
            t_362,
            [1usize, 1usize, 1usize, 288usize],
            t_97,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_363,
            [1usize, 1usize, 1usize, 18usize],
        );
        unary_logistic(t_363, t_364);
        binary_mul(t_363, t_364, t_365, 18usize);
        conv2d(
            t_365,
            [1usize, 1usize, 1usize, 18usize],
            t_96,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_366,
            [1usize, 1usize, 1usize, 288usize],
        );
        unary_logistic(t_366, t_367);
        binary_mul(t_357, t_367, t_368, 288usize);
        let scratch_129_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_129_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_129_1.copy_from_slice(t_95);
        im2col_padded(
            t_368,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_129_0,
        );
        matmul_bt_tiled(scratch_129_0, scratch_129_1, t_369, 3usize, 72usize, 18usize);
        bias_add(t_369, t_119, 12usize, 72usize);
        binary_add(t_369, t_351, t_370, 864usize);
        let scratch_131_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(731696usize),
                864usize,
            )
        };
        let scratch_131_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                62208usize,
            )
        };
        scratch_131_1.copy_from_slice(t_94);
        im2col_padded(
            t_370,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_131_0,
        );
        matmul_bt_tiled(scratch_131_0, scratch_131_1, t_371, 3usize, 18usize, 216usize);
        bias_add(t_371, t_28, 12usize, 864usize);
        unary_logistic(t_371, t_372);
        binary_mul(t_371, t_372, t_373, 10368usize);
        pad(
            t_373,
            [1usize, 12usize, 1usize, 864usize],
            t_374,
            [1usize, 14usize, 3usize, 864usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        depthwise_conv2d(
            t_374,
            [1usize, 14usize, 3usize, 864usize],
            t_27,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_26),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_375,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_375, t_376);
        binary_mul(t_375, t_376, t_377, 5184usize);
        reduce_mean_hw(t_377, t_378);
        reshape(t_378, t_382);
        conv2d(
            t_382,
            [1usize, 1usize, 1usize, 864usize],
            t_92,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_383,
            [1usize, 1usize, 1usize, 27usize],
        );
        unary_logistic(t_383, t_384);
        binary_mul(t_383, t_384, t_385, 27usize);
        conv2d(
            t_385,
            [1usize, 1usize, 1usize, 27usize],
            t_90,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_386,
            [1usize, 1usize, 1usize, 864usize],
        );
        unary_logistic(t_386, t_387);
        binary_mul(t_377, t_387, t_388, 864usize);
        conv2d(
            t_388,
            [1usize, 6usize, 1usize, 864usize],
            t_89,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_389,
            [1usize, 6usize, 1usize, 108usize],
        );
        conv2d(
            t_389,
            [1usize, 6usize, 1usize, 108usize],
            t_87,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_25),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_390,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_390, t_391);
        binary_mul(t_390, t_391, t_392, 5184usize);
        depthwise_conv2d(
            t_392,
            [1usize, 6usize, 1usize, 864usize],
            t_24,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_23),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_393,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_393, t_394);
        binary_mul(t_393, t_394, t_395, 5184usize);
        reduce_mean_hw(t_395, t_396);
        reshape(t_396, t_400);
        conv2d(
            t_400,
            [1usize, 1usize, 1usize, 864usize],
            t_86,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_401,
            [1usize, 1usize, 1usize, 27usize],
        );
        unary_logistic(t_401, t_402);
        binary_mul(t_401, t_402, t_403, 27usize);
        conv2d(
            t_403,
            [1usize, 1usize, 1usize, 27usize],
            t_85,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_404,
            [1usize, 1usize, 1usize, 864usize],
        );
        unary_logistic(t_404, t_405);
        binary_mul(t_395, t_405, t_406, 864usize);
        conv2d(
            t_406,
            [1usize, 6usize, 1usize, 864usize],
            t_84,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_407,
            [1usize, 6usize, 1usize, 108usize],
        );
        binary_add(t_407, t_389, t_408, 648usize);
        conv2d(
            t_408,
            [1usize, 6usize, 1usize, 108usize],
            t_83,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_22),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_409,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_409, t_410);
        binary_mul(t_409, t_410, t_411, 5184usize);
        depthwise_conv2d(
            t_411,
            [1usize, 6usize, 1usize, 864usize],
            t_21,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_20),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_412,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_412, t_413);
        binary_mul(t_412, t_413, t_414, 5184usize);
        reduce_mean_hw(t_414, t_415);
        reshape(t_415, t_419);
        conv2d(
            t_419,
            [1usize, 1usize, 1usize, 864usize],
            t_82,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_420,
            [1usize, 1usize, 1usize, 27usize],
        );
        unary_logistic(t_420, t_421);
        binary_mul(t_420, t_421, t_422, 27usize);
        conv2d(
            t_422,
            [1usize, 1usize, 1usize, 27usize],
            t_81,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_423,
            [1usize, 1usize, 1usize, 864usize],
        );
        unary_logistic(t_423, t_424);
        binary_mul(t_414, t_424, t_425, 864usize);
        conv2d(
            t_425,
            [1usize, 6usize, 1usize, 864usize],
            t_80,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_426,
            [1usize, 6usize, 1usize, 108usize],
        );
        binary_add(t_426, t_408, t_427, 648usize);
        conv2d(
            t_427,
            [1usize, 6usize, 1usize, 108usize],
            t_79,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_19),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_428,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_428, t_429);
        binary_mul(t_428, t_429, t_430, 5184usize);
        depthwise_conv2d(
            t_430,
            [1usize, 6usize, 1usize, 864usize],
            t_18,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_17),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_431,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_431, t_432);
        binary_mul(t_431, t_432, t_433, 5184usize);
        reduce_mean_hw(t_433, t_434);
        reshape(t_434, t_438);
        conv2d(
            t_438,
            [1usize, 1usize, 1usize, 864usize],
            t_78,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_439,
            [1usize, 1usize, 1usize, 27usize],
        );
        unary_logistic(t_439, t_440);
        binary_mul(t_439, t_440, t_441, 27usize);
        conv2d(
            t_441,
            [1usize, 1usize, 1usize, 27usize],
            t_77,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_442,
            [1usize, 1usize, 1usize, 864usize],
        );
        unary_logistic(t_442, t_443);
        binary_mul(t_433, t_443, t_444, 864usize);
        conv2d(
            t_444,
            [1usize, 6usize, 1usize, 864usize],
            t_76,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_445,
            [1usize, 6usize, 1usize, 108usize],
        );
        binary_add(t_445, t_427, t_446, 648usize);
        conv2d(
            t_446,
            [1usize, 6usize, 1usize, 108usize],
            t_75,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_16),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_447,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_447, t_448);
        binary_mul(t_447, t_448, t_449, 5184usize);
        depthwise_conv2d(
            t_449,
            [1usize, 6usize, 1usize, 864usize],
            t_15,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_14),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_450,
            [1usize, 6usize, 1usize, 864usize],
        );
        unary_logistic(t_450, t_451);
        binary_mul(t_450, t_451, t_452, 5184usize);
        reduce_mean_hw(t_452, t_453);
        reshape(t_453, t_457);
        conv2d(
            t_457,
            [1usize, 1usize, 1usize, 864usize],
            t_74,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_458,
            [1usize, 1usize, 1usize, 27usize],
        );
        unary_logistic(t_458, t_459);
        binary_mul(t_458, t_459, t_460, 27usize);
        conv2d(
            t_460,
            [1usize, 1usize, 1usize, 27usize],
            t_73,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_461,
            [1usize, 1usize, 1usize, 864usize],
        );
        unary_logistic(t_461, t_462);
        binary_mul(t_452, t_462, t_463, 864usize);
        conv2d(
            t_463,
            [1usize, 6usize, 1usize, 864usize],
            t_72,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_464,
            [1usize, 6usize, 1usize, 108usize],
        );
        binary_add(t_464, t_446, t_465, 648usize);
        conv2d(
            t_465,
            [1usize, 6usize, 1usize, 108usize],
            t_71,
            [1536usize, 1usize, 1usize, 108usize],
            Some(t_13),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_466,
            [1usize, 6usize, 1usize, 1536usize],
        );
        unary_logistic(t_466, t_467);
        binary_mul(t_466, t_467, t_468, 9216usize);
        pad(
            t_468,
            [1usize, 6usize, 1usize, 1536usize],
            t_469,
            [1usize, 8usize, 3usize, 1536usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        depthwise_conv2d(
            t_469,
            [1usize, 8usize, 3usize, 1536usize],
            t_12,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_11),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_470,
            [1usize, 3usize, 1usize, 1536usize],
        );
        unary_logistic(t_470, t_471);
        binary_mul(t_470, t_471, t_472, 4608usize);
        reduce_mean_hw(t_472, t_473);
        reshape(t_473, t_477);
        conv2d(
            t_477,
            [1usize, 1usize, 1usize, 1536usize],
            t_69,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_478,
            [1usize, 1usize, 1usize, 48usize],
        );
        unary_logistic(t_478, t_479);
        binary_mul(t_478, t_479, t_480, 48usize);
        conv2d(
            t_480,
            [1usize, 1usize, 1usize, 48usize],
            t_67,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_481,
            [1usize, 1usize, 1usize, 1536usize],
        );
        unary_logistic(t_481, t_482);
        binary_mul(t_472, t_482, t_483, 1536usize);
        conv2d(
            t_483,
            [1usize, 3usize, 1usize, 1536usize],
            t_66,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_484,
            [1usize, 3usize, 1usize, 192usize],
        );
        conv2d(
            t_484,
            [1usize, 3usize, 1usize, 192usize],
            t_64,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_10),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_485,
            [1usize, 3usize, 1usize, 1536usize],
        );
        unary_logistic(t_485, t_486);
        binary_mul(t_485, t_486, t_487, 4608usize);
        depthwise_conv2d(
            t_487,
            [1usize, 3usize, 1usize, 1536usize],
            t_9,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_8),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_488,
            [1usize, 3usize, 1usize, 1536usize],
        );
        unary_logistic(t_488, t_489);
        binary_mul(t_488, t_489, t_490, 4608usize);
        reduce_mean_hw(t_490, t_491);
        reshape(t_491, t_495);
        conv2d(
            t_495,
            [1usize, 1usize, 1usize, 1536usize],
            t_63,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_496,
            [1usize, 1usize, 1usize, 48usize],
        );
        unary_logistic(t_496, t_497);
        binary_mul(t_496, t_497, t_498, 48usize);
        conv2d(
            t_498,
            [1usize, 1usize, 1usize, 48usize],
            t_62,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_499,
            [1usize, 1usize, 1usize, 1536usize],
        );
        unary_logistic(t_499, t_500);
        binary_mul(t_490, t_500, t_501, 1536usize);
        conv2d(
            t_501,
            [1usize, 3usize, 1usize, 1536usize],
            t_61,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_502,
            [1usize, 3usize, 1usize, 192usize],
        );
        binary_add(t_502, t_484, t_503, 576usize);
        conv2d(
            t_503,
            [1usize, 3usize, 1usize, 192usize],
            t_60,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_7),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_504,
            [1usize, 3usize, 1usize, 1536usize],
        );
        unary_logistic(t_504, t_505);
        binary_mul(t_504, t_505, t_506, 4608usize);
        depthwise_conv2d(
            t_506,
            [1usize, 3usize, 1usize, 1536usize],
            t_6,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_5),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_507,
            [1usize, 3usize, 1usize, 1536usize],
        );
        unary_logistic(t_507, t_508);
        binary_mul(t_507, t_508, t_509, 4608usize);
        reduce_mean_hw(t_509, t_510);
        reshape(t_510, t_514);
        conv2d(
            t_514,
            [1usize, 1usize, 1usize, 1536usize],
            t_59,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_515,
            [1usize, 1usize, 1usize, 48usize],
        );
        unary_logistic(t_515, t_516);
        binary_mul(t_515, t_516, t_517, 48usize);
        conv2d(
            t_517,
            [1usize, 1usize, 1usize, 48usize],
            t_58,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_518,
            [1usize, 1usize, 1usize, 1536usize],
        );
        unary_logistic(t_518, t_519);
        binary_mul(t_509, t_519, t_520, 1536usize);
        conv2d(
            t_520,
            [1usize, 3usize, 1usize, 1536usize],
            t_57,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_521,
            [1usize, 3usize, 1usize, 192usize],
        );
        binary_add(t_521, t_503, t_522, 576usize);
        conv2d(
            t_522,
            [1usize, 3usize, 1usize, 192usize],
            t_56,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_4),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_523,
            [1usize, 3usize, 1usize, 1536usize],
        );
        unary_logistic(t_523, t_524);
        binary_mul(t_523, t_524, t_525, 4608usize);
        depthwise_conv2d(
            t_525,
            [1usize, 3usize, 1usize, 1536usize],
            t_3,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_2),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_526,
            [1usize, 3usize, 1usize, 1536usize],
        );
        unary_logistic(t_526, t_527);
        binary_mul(t_526, t_527, t_528, 4608usize);
        reduce_mean_hw(t_528, t_529);
        reshape(t_529, t_533);
        conv2d(
            t_533,
            [1usize, 1usize, 1usize, 1536usize],
            t_55,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_534,
            [1usize, 1usize, 1usize, 48usize],
        );
        unary_logistic(t_534, t_535);
        binary_mul(t_534, t_535, t_536, 48usize);
        conv2d(
            t_536,
            [1usize, 1usize, 1usize, 48usize],
            t_54,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_537,
            [1usize, 1usize, 1usize, 1536usize],
        );
        unary_logistic(t_537, t_538);
        binary_mul(t_528, t_538, t_539, 1536usize);
        conv2d(
            t_539,
            [1usize, 3usize, 1usize, 1536usize],
            t_53,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_540,
            [1usize, 3usize, 1usize, 192usize],
        );
        binary_add(t_540, t_522, t_541, 576usize);
        binary_mul(t_541, t_161, t_542, 192usize);
        binary_add(t_542, t_160, t_543, 192usize);
        conv2d_relu(
            t_543,
            [1usize, 3usize, 1usize, 192usize],
            t_52,
            [1024usize, 3usize, 3usize, 192usize],
            Some(t_1),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_544,
            [1usize, 1usize, 1usize, 1024usize],
        );
        reduce_mean_hw(t_544, t_545);
        fully_connected(t_545, 1024usize, t_164, Some(t_142), t_546, 6522usize);
    }
    output.copy_from_slice(&t_546);
}
/// Instrumented inference: accumulates per-op tick deltas into `op_ticks`.
pub fn forward_timed(
    input: &[f32; 144000usize],
    output: &mut [f32; 6522usize],
    op_ticks: &mut [u64; NUM_OPS],
    get_tick: fn() -> u64,
) {
    let t_169 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1usize,
        )
    };
    let t_170 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(4usize),
            144000usize,
        )
    };
    let t_171 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1usize,
        )
    };
    let t_172 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144004usize),
            1usize,
        )
    };
    let t_173 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144008usize),
            144000usize,
        )
    };
    let t_174 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_175 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144000usize),
            144000usize,
        )
    };
    let t_187 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_188 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288008usize),
            144000usize,
        )
    };
    let t_200 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(432008usize),
            1046528usize,
        )
    };
    let t_201 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1478536usize),
            1046528usize,
        )
    };
    let t_202 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            1046528usize,
        )
    };
    let t_206 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1334528usize),
            523775usize,
        )
    };
    let t_214 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            523775usize,
        )
    };
    let t_215 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_216 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    let t_217 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_218 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    let t_219 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_220 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    static mut T_221_BUF: Aligned16<49056usize> = Aligned16([0.0f32; 49056usize]);
    let t_221 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_221_BUF) as *mut f32,
            49056usize,
        )
    };
    let t_228 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_229 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144000usize),
            144000usize,
        )
    };
    let t_241 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            523264usize,
        )
    };
    let t_242 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(811264usize),
            523264usize,
        )
    };
    let t_243 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            523264usize,
        )
    };
    static mut T_247_BUF: Aligned16<262143usize> = Aligned16([0.0f32; 262143usize]);
    let t_247 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_247_BUF) as *mut f32,
            262143usize,
        )
    };
    static mut T_546_BUF: Aligned16<6522usize> = Aligned16([0.0f32; 6522usize]);
    let t_546 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_546_BUF) as *mut f32,
            6522usize,
        )
    };
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_7 = &tensor_data[T_7_OFFSET..T_7_OFFSET + T_7_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    let t_10 = &tensor_data[T_10_OFFSET..T_10_OFFSET + T_10_LEN];
    let t_11 = &tensor_data[T_11_OFFSET..T_11_OFFSET + T_11_LEN];
    let t_12 = &tensor_data[T_12_OFFSET..T_12_OFFSET + T_12_LEN];
    let t_13 = &tensor_data[T_13_OFFSET..T_13_OFFSET + T_13_LEN];
    let t_14 = &tensor_data[T_14_OFFSET..T_14_OFFSET + T_14_LEN];
    let t_15 = &tensor_data[T_15_OFFSET..T_15_OFFSET + T_15_LEN];
    let t_16 = &tensor_data[T_16_OFFSET..T_16_OFFSET + T_16_LEN];
    let t_17 = &tensor_data[T_17_OFFSET..T_17_OFFSET + T_17_LEN];
    let t_18 = &tensor_data[T_18_OFFSET..T_18_OFFSET + T_18_LEN];
    let t_19 = &tensor_data[T_19_OFFSET..T_19_OFFSET + T_19_LEN];
    let t_20 = &tensor_data[T_20_OFFSET..T_20_OFFSET + T_20_LEN];
    let t_21 = &tensor_data[T_21_OFFSET..T_21_OFFSET + T_21_LEN];
    let t_22 = &tensor_data[T_22_OFFSET..T_22_OFFSET + T_22_LEN];
    let t_23 = &tensor_data[T_23_OFFSET..T_23_OFFSET + T_23_LEN];
    let t_24 = &tensor_data[T_24_OFFSET..T_24_OFFSET + T_24_LEN];
    let t_25 = &tensor_data[T_25_OFFSET..T_25_OFFSET + T_25_LEN];
    let t_26 = &tensor_data[T_26_OFFSET..T_26_OFFSET + T_26_LEN];
    let t_27 = &tensor_data[T_27_OFFSET..T_27_OFFSET + T_27_LEN];
    let t_28 = &tensor_data[T_28_OFFSET..T_28_OFFSET + T_28_LEN];
    let t_29 = &tensor_data[T_29_OFFSET..T_29_OFFSET + T_29_LEN];
    let t_30 = &tensor_data[T_30_OFFSET..T_30_OFFSET + T_30_LEN];
    let t_31 = &tensor_data[T_31_OFFSET..T_31_OFFSET + T_31_LEN];
    let t_32 = &tensor_data[T_32_OFFSET..T_32_OFFSET + T_32_LEN];
    let t_33 = &tensor_data[T_33_OFFSET..T_33_OFFSET + T_33_LEN];
    let t_34 = &tensor_data[T_34_OFFSET..T_34_OFFSET + T_34_LEN];
    let t_35 = &tensor_data[T_35_OFFSET..T_35_OFFSET + T_35_LEN];
    let t_36 = &tensor_data[T_36_OFFSET..T_36_OFFSET + T_36_LEN];
    let t_37 = &tensor_data[T_37_OFFSET..T_37_OFFSET + T_37_LEN];
    let t_38 = &tensor_data[T_38_OFFSET..T_38_OFFSET + T_38_LEN];
    let t_39 = &tensor_data[T_39_OFFSET..T_39_OFFSET + T_39_LEN];
    let t_40 = &tensor_data[T_40_OFFSET..T_40_OFFSET + T_40_LEN];
    let t_41 = &tensor_data[T_41_OFFSET..T_41_OFFSET + T_41_LEN];
    let t_42 = &tensor_data[T_42_OFFSET..T_42_OFFSET + T_42_LEN];
    let t_43 = &tensor_data[T_43_OFFSET..T_43_OFFSET + T_43_LEN];
    let t_44 = &tensor_data[T_44_OFFSET..T_44_OFFSET + T_44_LEN];
    let t_45 = &tensor_data[T_45_OFFSET..T_45_OFFSET + T_45_LEN];
    let t_46 = &tensor_data[T_46_OFFSET..T_46_OFFSET + T_46_LEN];
    let t_47 = &tensor_data[T_47_OFFSET..T_47_OFFSET + T_47_LEN];
    let t_48 = &tensor_data[T_48_OFFSET..T_48_OFFSET + T_48_LEN];
    let t_49 = &tensor_data[T_49_OFFSET..T_49_OFFSET + T_49_LEN];
    let t_50 = &tensor_data[T_50_OFFSET..T_50_OFFSET + T_50_LEN];
    let t_51 = &tensor_data[T_51_OFFSET..T_51_OFFSET + T_51_LEN];
    let t_52 = &tensor_data[T_52_OFFSET..T_52_OFFSET + T_52_LEN];
    let t_53 = &tensor_data[T_53_OFFSET..T_53_OFFSET + T_53_LEN];
    let t_54 = &tensor_data[T_54_OFFSET..T_54_OFFSET + T_54_LEN];
    let t_55 = &tensor_data[T_55_OFFSET..T_55_OFFSET + T_55_LEN];
    let t_56 = &tensor_data[T_56_OFFSET..T_56_OFFSET + T_56_LEN];
    let t_57 = &tensor_data[T_57_OFFSET..T_57_OFFSET + T_57_LEN];
    let t_58 = &tensor_data[T_58_OFFSET..T_58_OFFSET + T_58_LEN];
    let t_59 = &tensor_data[T_59_OFFSET..T_59_OFFSET + T_59_LEN];
    let t_60 = &tensor_data[T_60_OFFSET..T_60_OFFSET + T_60_LEN];
    let t_61 = &tensor_data[T_61_OFFSET..T_61_OFFSET + T_61_LEN];
    let t_62 = &tensor_data[T_62_OFFSET..T_62_OFFSET + T_62_LEN];
    let t_63 = &tensor_data[T_63_OFFSET..T_63_OFFSET + T_63_LEN];
    let t_64 = &tensor_data[T_64_OFFSET..T_64_OFFSET + T_64_LEN];
    let t_65 = &tensor_data[T_65_OFFSET..T_65_OFFSET + T_65_LEN];
    let t_66 = &tensor_data[T_66_OFFSET..T_66_OFFSET + T_66_LEN];
    let t_67 = &tensor_data[T_67_OFFSET..T_67_OFFSET + T_67_LEN];
    let t_68 = &tensor_data[T_68_OFFSET..T_68_OFFSET + T_68_LEN];
    let t_69 = &tensor_data[T_69_OFFSET..T_69_OFFSET + T_69_LEN];
    let t_70 = &tensor_data[T_70_OFFSET..T_70_OFFSET + T_70_LEN];
    let t_71 = &tensor_data[T_71_OFFSET..T_71_OFFSET + T_71_LEN];
    let t_72 = &tensor_data[T_72_OFFSET..T_72_OFFSET + T_72_LEN];
    let t_73 = &tensor_data[T_73_OFFSET..T_73_OFFSET + T_73_LEN];
    let t_74 = &tensor_data[T_74_OFFSET..T_74_OFFSET + T_74_LEN];
    let t_75 = &tensor_data[T_75_OFFSET..T_75_OFFSET + T_75_LEN];
    let t_76 = &tensor_data[T_76_OFFSET..T_76_OFFSET + T_76_LEN];
    let t_77 = &tensor_data[T_77_OFFSET..T_77_OFFSET + T_77_LEN];
    let t_78 = &tensor_data[T_78_OFFSET..T_78_OFFSET + T_78_LEN];
    let t_79 = &tensor_data[T_79_OFFSET..T_79_OFFSET + T_79_LEN];
    let t_80 = &tensor_data[T_80_OFFSET..T_80_OFFSET + T_80_LEN];
    let t_81 = &tensor_data[T_81_OFFSET..T_81_OFFSET + T_81_LEN];
    let t_82 = &tensor_data[T_82_OFFSET..T_82_OFFSET + T_82_LEN];
    let t_83 = &tensor_data[T_83_OFFSET..T_83_OFFSET + T_83_LEN];
    let t_84 = &tensor_data[T_84_OFFSET..T_84_OFFSET + T_84_LEN];
    let t_85 = &tensor_data[T_85_OFFSET..T_85_OFFSET + T_85_LEN];
    let t_86 = &tensor_data[T_86_OFFSET..T_86_OFFSET + T_86_LEN];
    let t_87 = &tensor_data[T_87_OFFSET..T_87_OFFSET + T_87_LEN];
    let t_88 = &tensor_data[T_88_OFFSET..T_88_OFFSET + T_88_LEN];
    let t_89 = &tensor_data[T_89_OFFSET..T_89_OFFSET + T_89_LEN];
    let t_90 = &tensor_data[T_90_OFFSET..T_90_OFFSET + T_90_LEN];
    let t_91 = &tensor_data[T_91_OFFSET..T_91_OFFSET + T_91_LEN];
    let t_92 = &tensor_data[T_92_OFFSET..T_92_OFFSET + T_92_LEN];
    let t_93 = &tensor_data[T_93_OFFSET..T_93_OFFSET + T_93_LEN];
    let t_94 = &tensor_data[T_94_OFFSET..T_94_OFFSET + T_94_LEN];
    let t_95 = &tensor_data[T_95_OFFSET..T_95_OFFSET + T_95_LEN];
    let t_96 = &tensor_data[T_96_OFFSET..T_96_OFFSET + T_96_LEN];
    let t_97 = &tensor_data[T_97_OFFSET..T_97_OFFSET + T_97_LEN];
    let t_98 = &tensor_data[T_98_OFFSET..T_98_OFFSET + T_98_LEN];
    let t_99 = &tensor_data[T_99_OFFSET..T_99_OFFSET + T_99_LEN];
    let t_100 = &tensor_data[T_100_OFFSET..T_100_OFFSET + T_100_LEN];
    let t_101 = &tensor_data[T_101_OFFSET..T_101_OFFSET + T_101_LEN];
    let t_102 = &tensor_data[T_102_OFFSET..T_102_OFFSET + T_102_LEN];
    let t_103 = &tensor_data[T_103_OFFSET..T_103_OFFSET + T_103_LEN];
    let t_104 = &tensor_data[T_104_OFFSET..T_104_OFFSET + T_104_LEN];
    let t_105 = &tensor_data[T_105_OFFSET..T_105_OFFSET + T_105_LEN];
    let t_106 = &tensor_data[T_106_OFFSET..T_106_OFFSET + T_106_LEN];
    let t_107 = &tensor_data[T_107_OFFSET..T_107_OFFSET + T_107_LEN];
    let t_108 = &tensor_data[T_108_OFFSET..T_108_OFFSET + T_108_LEN];
    let t_109 = &tensor_data[T_109_OFFSET..T_109_OFFSET + T_109_LEN];
    let t_110 = &tensor_data[T_110_OFFSET..T_110_OFFSET + T_110_LEN];
    let t_111 = &tensor_data[T_111_OFFSET..T_111_OFFSET + T_111_LEN];
    let t_112 = &tensor_data[T_112_OFFSET..T_112_OFFSET + T_112_LEN];
    let t_113 = &tensor_data[T_113_OFFSET..T_113_OFFSET + T_113_LEN];
    let t_114 = &tensor_data[T_114_OFFSET..T_114_OFFSET + T_114_LEN];
    let t_115 = &tensor_data[T_115_OFFSET..T_115_OFFSET + T_115_LEN];
    let t_116 = &tensor_data[T_116_OFFSET..T_116_OFFSET + T_116_LEN];
    let t_117 = &tensor_data[T_117_OFFSET..T_117_OFFSET + T_117_LEN];
    let t_118 = &tensor_data[T_118_OFFSET..T_118_OFFSET + T_118_LEN];
    let t_119 = &tensor_data[T_119_OFFSET..T_119_OFFSET + T_119_LEN];
    let t_120 = &tensor_data[T_120_OFFSET..T_120_OFFSET + T_120_LEN];
    let t_121 = &tensor_data[T_121_OFFSET..T_121_OFFSET + T_121_LEN];
    let t_122 = &tensor_data[T_122_OFFSET..T_122_OFFSET + T_122_LEN];
    let t_126 = &tensor_data[T_126_OFFSET..T_126_OFFSET + T_126_LEN];
    let t_129 = &tensor_data[T_129_OFFSET..T_129_OFFSET + T_129_LEN];
    let t_133 = &tensor_data[T_133_OFFSET..T_133_OFFSET + T_133_LEN];
    let t_134 = &tensor_data[T_134_OFFSET..T_134_OFFSET + T_134_LEN];
    let t_135 = &tensor_data[T_135_OFFSET..T_135_OFFSET + T_135_LEN];
    let t_136 = &tensor_data[T_136_OFFSET..T_136_OFFSET + T_136_LEN];
    let t_142 = &tensor_data[T_142_OFFSET..T_142_OFFSET + T_142_LEN];
    let t_144 = &tensor_data[T_144_OFFSET..T_144_OFFSET + T_144_LEN];
    let t_145 = &tensor_data[T_145_OFFSET..T_145_OFFSET + T_145_LEN];
    let t_146 = &tensor_data[T_146_OFFSET..T_146_OFFSET + T_146_LEN];
    let t_150 = &tensor_data[T_150_OFFSET..T_150_OFFSET + T_150_LEN];
    let t_153 = &tensor_data[T_153_OFFSET..T_153_OFFSET + T_153_LEN];
    let t_156 = &tensor_data[T_156_OFFSET..T_156_OFFSET + T_156_LEN];
    let t_160 = &tensor_data[T_160_OFFSET..T_160_OFFSET + T_160_LEN];
    let t_161 = &tensor_data[T_161_OFFSET..T_161_OFFSET + T_161_LEN];
    let t_162 = &tensor_data[T_162_OFFSET..T_162_OFFSET + T_162_LEN];
    let t_163 = &tensor_data[T_163_OFFSET..T_163_OFFSET + T_163_LEN];
    let t_164 = &tensor_data[T_164_OFFSET..T_164_OFFSET + T_164_LEN];
    let t_165 = &tensor_data[T_165_OFFSET..T_165_OFFSET + T_165_LEN];
    let t_166 = &tensor_data[T_166_OFFSET..T_166_OFFSET + T_166_LEN];
    let t_167 = &tensor_data[T_167_OFFSET..T_167_OFFSET + T_167_LEN];
    let t_168 = &tensor_data[T_168_OFFSET..T_168_OFFSET + T_168_LEN];
    let t_185 = &tensor_data[T_185_OFFSET..T_185_OFFSET + T_185_LEN];
    let t_199 = &tensor_data[T_199_OFFSET..T_199_OFFSET + T_199_LEN];
    let t_226 = &tensor_data[T_226_OFFSET..T_226_OFFSET + T_226_LEN];
    let t_240 = &tensor_data[T_240_OFFSET..T_240_OFFSET + T_240_LEN];
    let t_549 = &tensor_data[T_549_OFFSET..T_549_OFFSET + T_549_LEN];
    let t_550 = &tensor_data[T_550_OFFSET..T_550_OFFSET + T_550_LEN];
    let t_551 = &tensor_data[T_551_OFFSET..T_551_OFFSET + T_551_LEN];
    let t_552 = &tensor_data[T_552_OFFSET..T_552_OFFSET + T_552_LEN];
    let t_553 = &tensor_data[T_553_OFFSET..T_553_OFFSET + T_553_LEN];
    let t_554 = &tensor_data[T_554_OFFSET..T_554_OFFSET + T_554_LEN];
    let t_555 = &tensor_data[T_555_OFFSET..T_555_OFFSET + T_555_LEN];
    let t_556 = &tensor_data[T_556_OFFSET..T_556_OFFSET + T_556_LEN];
    let t_557 = &tensor_data[T_557_OFFSET..T_557_OFFSET + T_557_LEN];
    let t_558 = &tensor_data[T_558_OFFSET..T_558_OFFSET + T_558_LEN];
    let t_559 = &tensor_data[T_559_OFFSET..T_559_OFFSET + T_559_LEN];
    let t_560 = &tensor_data[T_560_OFFSET..T_560_OFFSET + T_560_LEN];
    let t_561 = &tensor_data[T_561_OFFSET..T_561_OFFSET + T_561_LEN];
    let t_562 = &tensor_data[T_562_OFFSET..T_562_OFFSET + T_562_LEN];
    let t_563 = &tensor_data[T_563_OFFSET..T_563_OFFSET + T_563_LEN];
    let t_564 = &tensor_data[T_564_OFFSET..T_564_OFFSET + T_564_LEN];
    let t_565 = &tensor_data[T_565_OFFSET..T_565_OFFSET + T_565_LEN];
    let t_566 = &tensor_data[T_566_OFFSET..T_566_OFFSET + T_566_LEN];
    let t_567 = &tensor_data[T_567_OFFSET..T_567_OFFSET + T_567_LEN];
    let t_568 = &tensor_data[T_568_OFFSET..T_568_OFFSET + T_568_LEN];
    let t_569 = &tensor_data[T_569_OFFSET..T_569_OFFSET + T_569_LEN];
    let __t0 = get_tick();
    reduce_min(input, t_169);
    op_ticks[0usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_sub(input, t_169, t_170, 1usize);
    op_ticks[1usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_max(t_170, t_171);
    op_ticks[2usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_171, t_150, t_172, 1usize);
    op_ticks[3usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_div(t_170, t_172, t_173, 1usize);
    op_ticks[4usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_sub(t_173, t_146, t_174, 1usize);
    op_ticks[5usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_174, t_144, t_175, 1usize);
    op_ticks[6usize] += get_tick() - __t0;
    let __t0 = get_tick();
    strided_slice(
        t_175,
        &[1usize, 144000usize],
        t_187,
        &[0i32, 0i32],
        &[1i32, 144000i32],
        &[1i32, 1i32],
        0i32,
        0i32,
        0i32,
    );
    op_ticks[7usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_187, t_188);
    op_ticks[8usize] += get_tick() - __t0;
    let __t0 = get_tick();
    gather(
        t_188,
        &[1usize, 72000usize, 2usize],
        unsafe { core::slice::from_raw_parts(t_199.as_ptr() as *const i32, 1024usize) },
        t_200,
        &[1usize, 511usize, 1024usize, 2usize],
        1usize,
    );
    op_ticks[9usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_200, t_201);
    op_ticks[10usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_201, t_129, t_202, 2048usize);
    op_ticks[11usize] += get_tick() - __t0;
    let scratch_12_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            2048usize,
        )
    };
    let __t0 = get_tick();
    rfft_pack(t_202, scratch_12_0, 2048usize);
    op_ticks[12usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_549, 1024usize, 1usize);
    op_ticks[13usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_550, 1024usize, 2usize);
    op_ticks[14usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_551, 1024usize, 4usize);
    op_ticks[15usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_552, 1024usize, 8usize);
    op_ticks[16usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_553, 1024usize, 16usize);
    op_ticks[17usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_554, 1024usize, 32usize);
    op_ticks[18usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_555, 1024usize, 64usize);
    op_ticks[19usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_556, 1024usize, 128usize);
    op_ticks[20usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_557, 1024usize, 256usize);
    op_ticks[21usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_558, 1024usize, 512usize);
    op_ticks[22usize] += get_tick() - __t0;
    let __t0 = get_tick();
    rfft_unpack(scratch_12_0, t_559, t_206, 2048usize);
    op_ticks[23usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_206, t_214);
    op_ticks[24usize] += get_tick() - __t0;
    let __t0 = get_tick();
    for _batch_idx in 0..511usize {
        let _in_off = _batch_idx * 1025usize;
        let _out_off = _batch_idx * 96usize;
        fully_connected(
            &t_214[_in_off.._in_off + 1025usize],
            1025usize,
            t_166,
            None,
            &mut t_215[_out_off.._out_off + 96usize],
            96usize,
        );
    }
    op_ticks[25usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_215, t_216);
    op_ticks[26usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_216, t_216, t_217, 49056usize);
    op_ticks[27usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_pow(t_217, t_165, t_218, 1usize);
    op_ticks[28usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reverse_v2(t_218, &[511usize, 1usize, 96usize], t_219, 2usize);
    op_ticks[29usize] += get_tick() - __t0;
    let __t0 = get_tick();
    transpose(
        t_219,
        &[511usize, 1usize, 96usize],
        t_220,
        &[511usize, 96usize, 1usize],
        &[0usize, 2usize, 1usize],
    );
    op_ticks[30usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_220, t_221);
    op_ticks[31usize] += get_tick() - __t0;
    let __t0 = get_tick();
    strided_slice(
        t_175,
        &[1usize, 144000usize],
        t_228,
        &[0i32, 0i32],
        &[1i32, 144000i32],
        &[1i32, 1i32],
        0i32,
        0i32,
        0i32,
    );
    op_ticks[32usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_228, t_229);
    op_ticks[33usize] += get_tick() - __t0;
    let __t0 = get_tick();
    gather(
        t_229,
        &[1usize, 18000usize, 8usize],
        unsafe { core::slice::from_raw_parts(t_240.as_ptr() as *const i32, 511usize) },
        t_241,
        &[1usize, 511usize, 128usize, 8usize],
        1usize,
    );
    op_ticks[34usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_241, t_242);
    op_ticks[35usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_242, t_126, t_243, 1024usize);
    op_ticks[36usize] += get_tick() - __t0;
    let scratch_26_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(523264usize),
            1024usize,
        )
    };
    let __t0 = get_tick();
    rfft_pack(t_243, scratch_26_0, 1024usize);
    op_ticks[37usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_560, 512usize, 1usize);
    op_ticks[38usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_561, 512usize, 2usize);
    op_ticks[39usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_562, 512usize, 4usize);
    op_ticks[40usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_563, 512usize, 8usize);
    op_ticks[41usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_564, 512usize, 16usize);
    op_ticks[42usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_565, 512usize, 32usize);
    op_ticks[43usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_566, 512usize, 64usize);
    op_ticks[44usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_567, 512usize, 128usize);
    op_ticks[45usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_568, 512usize, 256usize);
    op_ticks[46usize] += get_tick() - __t0;
    let __t0 = get_tick();
    rfft_unpack(scratch_26_0, t_569, t_247, 1024usize);
    op_ticks[47usize] += get_tick() - __t0;
    for _frame_idx in 0..511usize {
        let t_255 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(661880usize),
                513usize,
            )
        };
        let t_256 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667580usize),
                96usize,
            )
        };
        let t_257 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667676usize),
                96usize,
            )
        };
        let t_258 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667772usize),
                96usize,
            )
        };
        let t_259 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667868usize),
                96usize,
            )
        };
        let t_260 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667964usize),
                96usize,
            )
        };
        let t_261 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668060usize),
                96usize,
            )
        };
        let t_262 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668156usize),
                96usize,
            )
        };
        let t_263 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667004usize),
                192usize,
            )
        };
        let t_264 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667196usize),
                192usize,
            )
        };
        let t_265 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667388usize),
                192usize,
            )
        };
        let t_266 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(616560usize),
                1152usize,
            )
        };
        let t_267 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(617712usize),
                1152usize,
            )
        };
        let t_268 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(618864usize),
                1152usize,
            )
        };
        let t_269 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(563760usize),
                2304usize,
            )
        };
        let t_270 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(620016usize),
                1152usize,
            )
        };
        let t_271 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(466992usize),
                3456usize,
            )
        };
        let t_272 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(470448usize),
                3456usize,
            )
        };
        let t_273 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(473904usize),
                3456usize,
            )
        };
        let t_274 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(95616usize),
                10800usize,
            )
        };
        let t_275 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(566064usize),
                1728usize,
            )
        };
        let t_276 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(567792usize),
                1728usize,
            )
        };
        let t_277 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(569520usize),
                1728usize,
            )
        };
        let t_278 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(623216usize),
                864usize,
            )
        };
        let t_279 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(571248usize),
                1728usize,
            )
        };
        let t_280 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(572976usize),
                1728usize,
            )
        };
        let t_281 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(574704usize),
                1728usize,
            )
        };
        let t_282 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(576432usize),
                1728usize,
            )
        };
        let t_283 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(578160usize),
                1728usize,
            )
        };
        let t_284 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(579888usize),
                1728usize,
            )
        };
        let t_285 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(624080usize),
                864usize,
            )
        };
        let t_286 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(624944usize),
                864usize,
            )
        };
        let t_287 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(581616usize),
                1728usize,
            )
        };
        let t_288 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(583344usize),
                1728usize,
            )
        };
        let t_289 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(585072usize),
                1728usize,
            )
        };
        let t_290 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(586800usize),
                1728usize,
            )
        };
        let t_291 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(588528usize),
                1728usize,
            )
        };
        let t_292 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(590256usize),
                1728usize,
            )
        };
        let t_293 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(625808usize),
                864usize,
            )
        };
        let t_294 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(626672usize),
                864usize,
            )
        };
        let t_295 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(165168usize),
                6912usize,
            )
        };
        let t_296 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(172080usize),
                6912usize,
            )
        };
        let t_297 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(178992usize),
                6912usize,
            )
        };
        let t_298 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(73152usize),
                22464usize,
            )
        };
        let t_299 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(477360usize),
                3456usize,
            )
        };
        let t_300 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(480816usize),
                3456usize,
            )
        };
        let t_301 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(484272usize),
                3456usize,
            )
        };
        let t_302 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662396usize),
                288usize,
            )
        };
        let t_306 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662684usize),
                288usize,
            )
        };
        let t_307 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669248usize),
                18usize,
            )
        };
        let t_308 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669268usize),
                18usize,
            )
        };
        let t_309 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669288usize),
                18usize,
            )
        };
        let t_310 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662972usize),
                288usize,
            )
        };
        let t_311 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663260usize),
                288usize,
            )
        };
        let t_312 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(487728usize),
                3456usize,
            )
        };
        let t_313 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(627536usize),
                864usize,
            )
        };
        let t_314 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(491184usize),
                3456usize,
            )
        };
        let t_315 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(494640usize),
                3456usize,
            )
        };
        let t_316 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(498096usize),
                3456usize,
            )
        };
        let t_317 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(501552usize),
                3456usize,
            )
        };
        let t_318 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(505008usize),
                3456usize,
            )
        };
        let t_319 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(508464usize),
                3456usize,
            )
        };
        let t_320 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663548usize),
                288usize,
            )
        };
        let t_324 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663836usize),
                288usize,
            )
        };
        let t_325 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669308usize),
                18usize,
            )
        };
        let t_326 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669328usize),
                18usize,
            )
        };
        let t_327 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669348usize),
                18usize,
            )
        };
        let t_328 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664124usize),
                288usize,
            )
        };
        let t_329 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664412usize),
                288usize,
            )
        };
        let t_330 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(511920usize),
                3456usize,
            )
        };
        let t_331 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(628400usize),
                864usize,
            )
        };
        let t_332 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(629264usize),
                864usize,
            )
        };
        let t_333 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(515376usize),
                3456usize,
            )
        };
        let t_334 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(518832usize),
                3456usize,
            )
        };
        let t_335 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(522288usize),
                3456usize,
            )
        };
        let t_336 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(525744usize),
                3456usize,
            )
        };
        let t_337 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(529200usize),
                3456usize,
            )
        };
        let t_338 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(532656usize),
                3456usize,
            )
        };
        let t_339 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664700usize),
                288usize,
            )
        };
        let t_343 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664988usize),
                288usize,
            )
        };
        let t_344 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669368usize),
                18usize,
            )
        };
        let t_345 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669388usize),
                18usize,
            )
        };
        let t_346 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669408usize),
                18usize,
            )
        };
        let t_347 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665276usize),
                288usize,
            )
        };
        let t_348 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665564usize),
                288usize,
            )
        };
        let t_349 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(536112usize),
                3456usize,
            )
        };
        let t_350 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(630128usize),
                864usize,
            )
        };
        let t_351 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(630992usize),
                864usize,
            )
        };
        let t_352 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(539568usize),
                3456usize,
            )
        };
        let t_353 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(543024usize),
                3456usize,
            )
        };
        let t_354 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(546480usize),
                3456usize,
            )
        };
        let t_355 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(549936usize),
                3456usize,
            )
        };
        let t_356 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(553392usize),
                3456usize,
            )
        };
        let t_357 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(556848usize),
                3456usize,
            )
        };
        let t_358 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665852usize),
                288usize,
            )
        };
        let t_362 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666140usize),
                288usize,
            )
        };
        let t_363 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669428usize),
                18usize,
            )
        };
        let t_364 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669448usize),
                18usize,
            )
        };
        let t_365 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669468usize),
                18usize,
            )
        };
        let t_366 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666428usize),
                288usize,
            )
        };
        let t_367 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666716usize),
                288usize,
            )
        };
        let t_368 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(560304usize),
                3456usize,
            )
        };
        let t_369 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(631856usize),
                864usize,
            )
        };
        let t_370 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(632720usize),
                864usize,
            )
        };
        let t_371 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(106416usize),
                10368usize,
            )
        };
        let t_372 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(116784usize),
                10368usize,
            )
        };
        let t_373 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(127152usize),
                10368usize,
            )
        };
        let t_374 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(36864usize),
                36288usize,
            )
        };
        let t_375 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(185904usize),
                5184usize,
            )
        };
        let t_376 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(191088usize),
                5184usize,
            )
        };
        let t_377 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(196272usize),
                5184usize,
            )
        };
        let t_378 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(633584usize),
                864usize,
            )
        };
        let t_382 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(634448usize),
                864usize,
            )
        };
        let t_383 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668828usize),
                27usize,
            )
        };
        let t_384 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668856usize),
                27usize,
            )
        };
        let t_385 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668884usize),
                27usize,
            )
        };
        let t_386 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(635312usize),
                864usize,
            )
        };
        let t_387 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(636176usize),
                864usize,
            )
        };
        let t_388 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(201456usize),
                5184usize,
            )
        };
        let t_389 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(650864usize),
                648usize,
            )
        };
        let t_390 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(206640usize),
                5184usize,
            )
        };
        let t_391 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(211824usize),
                5184usize,
            )
        };
        let t_392 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(217008usize),
                5184usize,
            )
        };
        let t_393 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(222192usize),
                5184usize,
            )
        };
        let t_394 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(227376usize),
                5184usize,
            )
        };
        let t_395 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(232560usize),
                5184usize,
            )
        };
        let t_396 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(637040usize),
                864usize,
            )
        };
        let t_400 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(637904usize),
                864usize,
            )
        };
        let t_401 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668912usize),
                27usize,
            )
        };
        let t_402 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668940usize),
                27usize,
            )
        };
        let t_403 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668968usize),
                27usize,
            )
        };
        let t_404 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(638768usize),
                864usize,
            )
        };
        let t_405 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(639632usize),
                864usize,
            )
        };
        let t_406 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(237744usize),
                5184usize,
            )
        };
        let t_407 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(651512usize),
                648usize,
            )
        };
        let t_408 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(652160usize),
                648usize,
            )
        };
        let t_409 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(242928usize),
                5184usize,
            )
        };
        let t_410 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(248112usize),
                5184usize,
            )
        };
        let t_411 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(253296usize),
                5184usize,
            )
        };
        let t_412 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(258480usize),
                5184usize,
            )
        };
        let t_413 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(263664usize),
                5184usize,
            )
        };
        let t_414 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(268848usize),
                5184usize,
            )
        };
        let t_415 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(640496usize),
                864usize,
            )
        };
        let t_419 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(641360usize),
                864usize,
            )
        };
        let t_420 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668996usize),
                27usize,
            )
        };
        let t_421 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669024usize),
                27usize,
            )
        };
        let t_422 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669052usize),
                27usize,
            )
        };
        let t_423 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(642224usize),
                864usize,
            )
        };
        let t_424 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(643088usize),
                864usize,
            )
        };
        let t_425 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(274032usize),
                5184usize,
            )
        };
        let t_426 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(652808usize),
                648usize,
            )
        };
        let t_427 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(653456usize),
                648usize,
            )
        };
        let t_428 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(279216usize),
                5184usize,
            )
        };
        let t_429 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(284400usize),
                5184usize,
            )
        };
        let t_430 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(289584usize),
                5184usize,
            )
        };
        let t_431 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(294768usize),
                5184usize,
            )
        };
        let t_432 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(299952usize),
                5184usize,
            )
        };
        let t_433 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(305136usize),
                5184usize,
            )
        };
        let t_434 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(643952usize),
                864usize,
            )
        };
        let t_438 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(644816usize),
                864usize,
            )
        };
        let t_439 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669080usize),
                27usize,
            )
        };
        let t_440 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669108usize),
                27usize,
            )
        };
        let t_441 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669136usize),
                27usize,
            )
        };
        let t_442 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(645680usize),
                864usize,
            )
        };
        let t_443 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(646544usize),
                864usize,
            )
        };
        let t_444 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(310320usize),
                5184usize,
            )
        };
        let t_445 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(654104usize),
                648usize,
            )
        };
        let t_446 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(654752usize),
                648usize,
            )
        };
        let t_447 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(315504usize),
                5184usize,
            )
        };
        let t_448 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(320688usize),
                5184usize,
            )
        };
        let t_449 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(325872usize),
                5184usize,
            )
        };
        let t_450 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(331056usize),
                5184usize,
            )
        };
        let t_451 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(336240usize),
                5184usize,
            )
        };
        let t_452 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(341424usize),
                5184usize,
            )
        };
        let t_453 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(647408usize),
                864usize,
            )
        };
        let t_457 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(648272usize),
                864usize,
            )
        };
        let t_458 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669164usize),
                27usize,
            )
        };
        let t_459 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669192usize),
                27usize,
            )
        };
        let t_460 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669220usize),
                27usize,
            )
        };
        let t_461 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(649136usize),
                864usize,
            )
        };
        let t_462 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(650000usize),
                864usize,
            )
        };
        let t_463 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(346608usize),
                5184usize,
            )
        };
        let t_464 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(655400usize),
                648usize,
            )
        };
        let t_465 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(656048usize),
                648usize,
            )
        };
        let t_466 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(137520usize),
                9216usize,
            )
        };
        let t_467 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(146736usize),
                9216usize,
            )
        };
        let t_468 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(155952usize),
                9216usize,
            )
        };
        let t_469 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
                36864usize,
            )
        };
        let t_470 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(351792usize),
                4608usize,
            )
        };
        let t_471 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(356400usize),
                4608usize,
            )
        };
        let t_472 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(361008usize),
                4608usize,
            )
        };
        let t_473 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(591984usize),
                1536usize,
            )
        };
        let t_477 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(593520usize),
                1536usize,
            )
        };
        let t_478 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668252usize),
                48usize,
            )
        };
        let t_479 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668300usize),
                48usize,
            )
        };
        let t_480 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668348usize),
                48usize,
            )
        };
        let t_481 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(595056usize),
                1536usize,
            )
        };
        let t_482 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(596592usize),
                1536usize,
            )
        };
        let t_483 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(365616usize),
                4608usize,
            )
        };
        let t_484 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(656696usize),
                576usize,
            )
        };
        let t_485 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(370224usize),
                4608usize,
            )
        };
        let t_486 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(374832usize),
                4608usize,
            )
        };
        let t_487 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(379440usize),
                4608usize,
            )
        };
        let t_488 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(384048usize),
                4608usize,
            )
        };
        let t_489 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(388656usize),
                4608usize,
            )
        };
        let t_490 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(393264usize),
                4608usize,
            )
        };
        let t_491 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(598128usize),
                1536usize,
            )
        };
        let t_495 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(599664usize),
                1536usize,
            )
        };
        let t_496 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668396usize),
                48usize,
            )
        };
        let t_497 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668444usize),
                48usize,
            )
        };
        let t_498 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668492usize),
                48usize,
            )
        };
        let t_499 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(601200usize),
                1536usize,
            )
        };
        let t_500 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(602736usize),
                1536usize,
            )
        };
        let t_501 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(397872usize),
                4608usize,
            )
        };
        let t_502 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(657272usize),
                576usize,
            )
        };
        let t_503 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(657848usize),
                576usize,
            )
        };
        let t_504 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(402480usize),
                4608usize,
            )
        };
        let t_505 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(407088usize),
                4608usize,
            )
        };
        let t_506 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(411696usize),
                4608usize,
            )
        };
        let t_507 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(416304usize),
                4608usize,
            )
        };
        let t_508 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(420912usize),
                4608usize,
            )
        };
        let t_509 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(425520usize),
                4608usize,
            )
        };
        let t_510 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(604272usize),
                1536usize,
            )
        };
        let t_514 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(605808usize),
                1536usize,
            )
        };
        let t_515 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668540usize),
                48usize,
            )
        };
        let t_516 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668588usize),
                48usize,
            )
        };
        let t_517 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668636usize),
                48usize,
            )
        };
        let t_518 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(607344usize),
                1536usize,
            )
        };
        let t_519 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(608880usize),
                1536usize,
            )
        };
        let t_520 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(430128usize),
                4608usize,
            )
        };
        let t_521 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(658424usize),
                576usize,
            )
        };
        let t_522 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(659000usize),
                576usize,
            )
        };
        let t_523 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(434736usize),
                4608usize,
            )
        };
        let t_524 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(439344usize),
                4608usize,
            )
        };
        let t_525 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(443952usize),
                4608usize,
            )
        };
        let t_526 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(448560usize),
                4608usize,
            )
        };
        let t_527 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(453168usize),
                4608usize,
            )
        };
        let t_528 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(457776usize),
                4608usize,
            )
        };
        let t_529 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(610416usize),
                1536usize,
            )
        };
        let t_533 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(611952usize),
                1536usize,
            )
        };
        let t_534 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668684usize),
                48usize,
            )
        };
        let t_535 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668732usize),
                48usize,
            )
        };
        let t_536 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668780usize),
                48usize,
            )
        };
        let t_537 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(613488usize),
                1536usize,
            )
        };
        let t_538 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(615024usize),
                1536usize,
            )
        };
        let t_539 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(462384usize),
                4608usize,
            )
        };
        let t_540 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(659576usize),
                576usize,
            )
        };
        let t_541 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(660152usize),
                576usize,
            )
        };
        let t_542 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(660728usize),
                576usize,
            )
        };
        let t_543 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(661304usize),
                576usize,
            )
        };
        let t_544 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(621168usize),
                1024usize,
            )
        };
        let t_545 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(622192usize),
                1024usize,
            )
        };
        let t_547 = &t_247[_frame_idx * 513usize..(_frame_idx + 1) * 513usize];
        let t_548 = &t_221[_frame_idx * 96usize..(_frame_idx + 1) * 96usize];
        let __t0 = get_tick();
        reshape(t_547, t_255);
        op_ticks[48usize] += get_tick() - __t0;
        let __t0 = get_tick();
        fully_connected(t_255, 513usize, t_168, None, t_256, 96usize);
        op_ticks[49usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_256, t_257);
        op_ticks[50usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_257, t_257, t_258, 96usize);
        op_ticks[51usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_pow(t_258, t_167, t_259, 1usize);
        op_ticks[52usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reverse_v2(t_259, &[1usize, 1usize, 96usize], t_260, 2usize);
        op_ticks[53usize] += get_tick() - __t0;
        let __t0 = get_tick();
        transpose(
            t_260,
            &[1usize, 1usize, 96usize],
            t_261,
            &[1usize, 96usize, 1usize],
            &[0usize, 2usize, 1usize],
        );
        op_ticks[54usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_261, t_262);
        op_ticks[55usize] += get_tick() - __t0;
        let __t0 = get_tick();
        {
            let src = t_548;
            for p in 0..96usize {
                for a in 0..1usize {
                    let src_off = p * (1usize * 1usize) + a * 1usize;
                    let dst_off = p * (2usize * 1usize) + (0usize + a) * 1usize;
                    t_263[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        {
            let src = t_262;
            for p in 0..96usize {
                for a in 0..1usize {
                    let src_off = p * (1usize * 1usize) + a * 1usize;
                    let dst_off = p * (2usize * 1usize) + (1usize + a) * 1usize;
                    t_263[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        op_ticks[56usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_263, t_163, t_264, 2usize);
        op_ticks[57usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_264, t_162, t_265, 2usize);
        op_ticks[58usize] += get_tick() - __t0;
        let scratch_38_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                3072usize,
            )
        };
        let scratch_38_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672560usize),
                1536usize,
            )
        };
        scratch_38_1.copy_from_slice(t_122);
        let __t0 = get_tick();
        im2col_padded(
            t_265,
            [1usize, 96usize, 1usize, 2usize],
            [4usize, 8usize],
            [2usize, 2usize],
            [1usize, 1usize, 3usize, 4usize],
            [48usize, 1usize],
            scratch_38_0,
        );
        op_ticks[59usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_38_0, scratch_38_1, t_266, 12usize, 16usize, 6usize);
        op_ticks[60usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_266, t_51, 48usize, 24usize);
        relu(t_266);
        op_ticks[61usize] += get_tick() - __t0;
        let __t0 = get_tick();
        average_pool2d(
            t_266,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 2usize],
            [1usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_267,
            [1usize, 48usize, 1usize, 24usize],
        );
        op_ticks[62usize] += get_tick() - __t0;
        let __t0 = get_tick();
        max_pool2d(
            t_266,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 2usize],
            [1usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_268,
            [1usize, 48usize, 1usize, 24usize],
        );
        op_ticks[63usize] += get_tick() - __t0;
        let __t0 = get_tick();
        {
            let src = t_268;
            for p in 0..48usize {
                for a in 0..24usize {
                    let src_off = p * (24usize * 1usize) + a * 1usize;
                    let dst_off = p * (48usize * 1usize) + (0usize + a) * 1usize;
                    t_269[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        {
            let src = t_267;
            for p in 0..48usize {
                for a in 0..24usize {
                    let src_off = p * (24usize * 1usize) + a * 1usize;
                    let dst_off = p * (48usize * 1usize) + (24usize + a) * 1usize;
                    t_269[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        op_ticks[64usize] += get_tick() - __t0;
        let scratch_42_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2304usize,
            )
        };
        let scratch_42_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(671792usize),
                1152usize,
            )
        };
        scratch_42_1.copy_from_slice(t_121);
        let __t0 = get_tick();
        im2col_padded(
            t_269,
            [1usize, 48usize, 1usize, 48usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [48usize, 1usize],
            scratch_42_0,
        );
        op_ticks[65usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_42_0, scratch_42_1, t_270, 12usize, 12usize, 6usize);
        op_ticks[66usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_270, t_50, 48usize, 24usize);
        op_ticks[67usize] += get_tick() - __t0;
        let scratch_43_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(671216usize),
                1152usize,
            )
        };
        let scratch_43_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                1728usize,
            )
        };
        scratch_43_1.copy_from_slice(t_120);
        let __t0 = get_tick();
        im2col_padded(
            t_270,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [48usize, 1usize],
            scratch_43_0,
        );
        op_ticks[68usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_43_0, scratch_43_1, t_271, 12usize, 6usize, 18usize);
        op_ticks[69usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_271, t_49, 48usize, 72usize);
        op_ticks[70usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_271, t_272);
        op_ticks[71usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_271, t_272, t_273, 3456usize);
        op_ticks[72usize] += get_tick() - __t0;
        let __t0 = get_tick();
        pad(
            t_273,
            [1usize, 48usize, 1usize, 72usize],
            t_274,
            [1usize, 50usize, 3usize, 72usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[73usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_274,
            [1usize, 50usize, 3usize, 72usize],
            t_48,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_47),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_275,
            [1usize, 24usize, 1usize, 72usize],
        );
        op_ticks[74usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_275, t_276);
        op_ticks[75usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_275, t_276, t_277, 1728usize);
        op_ticks[76usize] += get_tick() - __t0;
        let scratch_50_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_50_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_50_1.copy_from_slice(t_118);
        let __t0 = get_tick();
        im2col_padded(
            t_277,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_50_0,
        );
        op_ticks[77usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_50_0, scratch_50_1, t_278, 6usize, 18usize, 9usize);
        op_ticks[78usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_278, t_117, 24usize, 36usize);
        op_ticks[79usize] += get_tick() - __t0;
        let scratch_51_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                864usize,
            )
        };
        let scratch_51_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_51_1.copy_from_slice(t_116);
        let __t0 = get_tick();
        im2col_padded(
            t_278,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_51_0,
        );
        op_ticks[80usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_51_0, scratch_51_1, t_279, 6usize, 9usize, 18usize);
        op_ticks[81usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_279, t_46, 24usize, 72usize);
        op_ticks[82usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_279, t_280);
        op_ticks[83usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_279, t_280, t_281, 1728usize);
        op_ticks[84usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_281,
            [1usize, 24usize, 1usize, 72usize],
            t_45,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_44),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_282,
            [1usize, 24usize, 1usize, 72usize],
        );
        op_ticks[85usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_282, t_283);
        op_ticks[86usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_282, t_283, t_284, 1728usize);
        op_ticks[87usize] += get_tick() - __t0;
        let scratch_57_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_57_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_57_1.copy_from_slice(t_115);
        let __t0 = get_tick();
        im2col_padded(
            t_284,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_57_0,
        );
        op_ticks[88usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_57_0, scratch_57_1, t_285, 6usize, 18usize, 9usize);
        op_ticks[89usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_285, t_117, 24usize, 36usize);
        op_ticks[90usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_285, t_278, t_286, 864usize);
        op_ticks[91usize] += get_tick() - __t0;
        let scratch_59_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                864usize,
            )
        };
        let scratch_59_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_59_1.copy_from_slice(t_114);
        let __t0 = get_tick();
        im2col_padded(
            t_286,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_59_0,
        );
        op_ticks[92usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_59_0, scratch_59_1, t_287, 6usize, 9usize, 18usize);
        op_ticks[93usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_287, t_43, 24usize, 72usize);
        op_ticks[94usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_287, t_288);
        op_ticks[95usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_287, t_288, t_289, 1728usize);
        op_ticks[96usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_289,
            [1usize, 24usize, 1usize, 72usize],
            t_42,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_41),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_290,
            [1usize, 24usize, 1usize, 72usize],
        );
        op_ticks[97usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_290, t_291);
        op_ticks[98usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_290, t_291, t_292, 1728usize);
        op_ticks[99usize] += get_tick() - __t0;
        let scratch_65_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_65_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_65_1.copy_from_slice(t_113);
        let __t0 = get_tick();
        im2col_padded(
            t_292,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_65_0,
        );
        op_ticks[100usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_65_0, scratch_65_1, t_293, 6usize, 18usize, 9usize);
        op_ticks[101usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_293, t_117, 24usize, 36usize);
        op_ticks[102usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_293, t_286, t_294, 864usize);
        op_ticks[103usize] += get_tick() - __t0;
        let scratch_67_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(679856usize),
                864usize,
            )
        };
        let scratch_67_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                10368usize,
            )
        };
        scratch_67_1.copy_from_slice(t_112);
        let __t0 = get_tick();
        im2col_padded(
            t_294,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_67_0,
        );
        op_ticks[104usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_67_0, scratch_67_1, t_295, 6usize, 9usize, 72usize);
        op_ticks[105usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_295, t_40, 24usize, 288usize);
        op_ticks[106usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_295, t_296);
        op_ticks[107usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_295, t_296, t_297, 6912usize);
        op_ticks[108usize] += get_tick() - __t0;
        let __t0 = get_tick();
        pad(
            t_297,
            [1usize, 24usize, 1usize, 288usize],
            t_298,
            [1usize, 26usize, 3usize, 288usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[109usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_298,
            [1usize, 26usize, 3usize, 288usize],
            t_39,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_38),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_299,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[110usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_299, t_300);
        op_ticks[111usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_299, t_300, t_301, 3456usize);
        op_ticks[112usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_301, t_302);
        op_ticks[113usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_302, t_306);
        op_ticks[114usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_306,
            [1usize, 1usize, 1usize, 288usize],
            t_110,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_307,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[115usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_307, t_308);
        op_ticks[116usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_307, t_308, t_309, 18usize);
        op_ticks[117usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_309,
            [1usize, 1usize, 1usize, 18usize],
            t_108,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_310,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[118usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_310, t_311);
        op_ticks[119usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_301, t_311, t_312, 288usize);
        op_ticks[120usize] += get_tick() - __t0;
        let scratch_82_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_82_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_82_1.copy_from_slice(t_107);
        let __t0 = get_tick();
        im2col_padded(
            t_312,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_82_0,
        );
        op_ticks[121usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_82_0, scratch_82_1, t_313, 3usize, 72usize, 18usize);
        op_ticks[122usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_313, t_119, 12usize, 72usize);
        op_ticks[123usize] += get_tick() - __t0;
        let scratch_83_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_83_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_83_1.copy_from_slice(t_106);
        let __t0 = get_tick();
        im2col_padded(
            t_313,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_83_0,
        );
        op_ticks[124usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_83_0, scratch_83_1, t_314, 3usize, 18usize, 72usize);
        op_ticks[125usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_314, t_37, 12usize, 288usize);
        op_ticks[126usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_314, t_315);
        op_ticks[127usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_314, t_315, t_316, 3456usize);
        op_ticks[128usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_316,
            [1usize, 12usize, 1usize, 288usize],
            t_36,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_35),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_317,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[129usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_317, t_318);
        op_ticks[130usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_317, t_318, t_319, 3456usize);
        op_ticks[131usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_319, t_320);
        op_ticks[132usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_320, t_324);
        op_ticks[133usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_324,
            [1usize, 1usize, 1usize, 288usize],
            t_105,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_325,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[134usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_325, t_326);
        op_ticks[135usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_325, t_326, t_327, 18usize);
        op_ticks[136usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_327,
            [1usize, 1usize, 1usize, 18usize],
            t_104,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_328,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[137usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_328, t_329);
        op_ticks[138usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_319, t_329, t_330, 288usize);
        op_ticks[139usize] += get_tick() - __t0;
        let scratch_97_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_97_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_97_1.copy_from_slice(t_103);
        let __t0 = get_tick();
        im2col_padded(
            t_330,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_97_0,
        );
        op_ticks[140usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_97_0, scratch_97_1, t_331, 3usize, 72usize, 18usize);
        op_ticks[141usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_331, t_119, 12usize, 72usize);
        op_ticks[142usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_331, t_313, t_332, 864usize);
        op_ticks[143usize] += get_tick() - __t0;
        let scratch_99_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_99_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_99_1.copy_from_slice(t_102);
        let __t0 = get_tick();
        im2col_padded(
            t_332,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_99_0,
        );
        op_ticks[144usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_99_0, scratch_99_1, t_333, 3usize, 18usize, 72usize);
        op_ticks[145usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_333, t_34, 12usize, 288usize);
        op_ticks[146usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_333, t_334);
        op_ticks[147usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_333, t_334, t_335, 3456usize);
        op_ticks[148usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_335,
            [1usize, 12usize, 1usize, 288usize],
            t_33,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_32),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_336,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[149usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_336, t_337);
        op_ticks[150usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_336, t_337, t_338, 3456usize);
        op_ticks[151usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_338, t_339);
        op_ticks[152usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_339, t_343);
        op_ticks[153usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_343,
            [1usize, 1usize, 1usize, 288usize],
            t_101,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_344,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[154usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_344, t_345);
        op_ticks[155usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_344, t_345, t_346, 18usize);
        op_ticks[156usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_346,
            [1usize, 1usize, 1usize, 18usize],
            t_100,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_347,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[157usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_347, t_348);
        op_ticks[158usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_338, t_348, t_349, 288usize);
        op_ticks[159usize] += get_tick() - __t0;
        let scratch_113_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_113_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_113_1.copy_from_slice(t_99);
        let __t0 = get_tick();
        im2col_padded(
            t_349,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_113_0,
        );
        op_ticks[160usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_113_0, scratch_113_1, t_350, 3usize, 72usize, 18usize);
        op_ticks[161usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_350, t_119, 12usize, 72usize);
        op_ticks[162usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_350, t_332, t_351, 864usize);
        op_ticks[163usize] += get_tick() - __t0;
        let scratch_115_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_115_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_115_1.copy_from_slice(t_98);
        let __t0 = get_tick();
        im2col_padded(
            t_351,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_115_0,
        );
        op_ticks[164usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_115_0, scratch_115_1, t_352, 3usize, 18usize, 72usize);
        op_ticks[165usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_352, t_31, 12usize, 288usize);
        op_ticks[166usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_352, t_353);
        op_ticks[167usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_352, t_353, t_354, 3456usize);
        op_ticks[168usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_354,
            [1usize, 12usize, 1usize, 288usize],
            t_30,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_29),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_355,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[169usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_355, t_356);
        op_ticks[170usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_355, t_356, t_357, 3456usize);
        op_ticks[171usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_357, t_358);
        op_ticks[172usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_358, t_362);
        op_ticks[173usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_362,
            [1usize, 1usize, 1usize, 288usize],
            t_97,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_363,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[174usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_363, t_364);
        op_ticks[175usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_363, t_364, t_365, 18usize);
        op_ticks[176usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_365,
            [1usize, 1usize, 1usize, 18usize],
            t_96,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_366,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[177usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_366, t_367);
        op_ticks[178usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_357, t_367, t_368, 288usize);
        op_ticks[179usize] += get_tick() - __t0;
        let scratch_129_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_129_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_129_1.copy_from_slice(t_95);
        let __t0 = get_tick();
        im2col_padded(
            t_368,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_129_0,
        );
        op_ticks[180usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_129_0, scratch_129_1, t_369, 3usize, 72usize, 18usize);
        op_ticks[181usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_369, t_119, 12usize, 72usize);
        op_ticks[182usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_369, t_351, t_370, 864usize);
        op_ticks[183usize] += get_tick() - __t0;
        let scratch_131_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(731696usize),
                864usize,
            )
        };
        let scratch_131_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                62208usize,
            )
        };
        scratch_131_1.copy_from_slice(t_94);
        let __t0 = get_tick();
        im2col_padded(
            t_370,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_131_0,
        );
        op_ticks[184usize] += get_tick() - __t0;
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_131_0, scratch_131_1, t_371, 3usize, 18usize, 216usize);
        op_ticks[185usize] += get_tick() - __t0;
        let __t0 = get_tick();
        bias_add(t_371, t_28, 12usize, 864usize);
        op_ticks[186usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_371, t_372);
        op_ticks[187usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_371, t_372, t_373, 10368usize);
        op_ticks[188usize] += get_tick() - __t0;
        let __t0 = get_tick();
        pad(
            t_373,
            [1usize, 12usize, 1usize, 864usize],
            t_374,
            [1usize, 14usize, 3usize, 864usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[189usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_374,
            [1usize, 14usize, 3usize, 864usize],
            t_27,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_26),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_375,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[190usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_375, t_376);
        op_ticks[191usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_375, t_376, t_377, 5184usize);
        op_ticks[192usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_377, t_378);
        op_ticks[193usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_378, t_382);
        op_ticks[194usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_382,
            [1usize, 1usize, 1usize, 864usize],
            t_92,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_383,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[195usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_383, t_384);
        op_ticks[196usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_383, t_384, t_385, 27usize);
        op_ticks[197usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_385,
            [1usize, 1usize, 1usize, 27usize],
            t_90,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_386,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[198usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_386, t_387);
        op_ticks[199usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_377, t_387, t_388, 864usize);
        op_ticks[200usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_388,
            [1usize, 6usize, 1usize, 864usize],
            t_89,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_389,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[201usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_389,
            [1usize, 6usize, 1usize, 108usize],
            t_87,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_25),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_390,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[202usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_390, t_391);
        op_ticks[203usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_390, t_391, t_392, 5184usize);
        op_ticks[204usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_392,
            [1usize, 6usize, 1usize, 864usize],
            t_24,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_23),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_393,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[205usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_393, t_394);
        op_ticks[206usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_393, t_394, t_395, 5184usize);
        op_ticks[207usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_395, t_396);
        op_ticks[208usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_396, t_400);
        op_ticks[209usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_400,
            [1usize, 1usize, 1usize, 864usize],
            t_86,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_401,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[210usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_401, t_402);
        op_ticks[211usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_401, t_402, t_403, 27usize);
        op_ticks[212usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_403,
            [1usize, 1usize, 1usize, 27usize],
            t_85,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_404,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[213usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_404, t_405);
        op_ticks[214usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_395, t_405, t_406, 864usize);
        op_ticks[215usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_406,
            [1usize, 6usize, 1usize, 864usize],
            t_84,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_407,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[216usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_407, t_389, t_408, 648usize);
        op_ticks[217usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_408,
            [1usize, 6usize, 1usize, 108usize],
            t_83,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_22),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_409,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[218usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_409, t_410);
        op_ticks[219usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_409, t_410, t_411, 5184usize);
        op_ticks[220usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_411,
            [1usize, 6usize, 1usize, 864usize],
            t_21,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_20),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_412,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[221usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_412, t_413);
        op_ticks[222usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_412, t_413, t_414, 5184usize);
        op_ticks[223usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_414, t_415);
        op_ticks[224usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_415, t_419);
        op_ticks[225usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_419,
            [1usize, 1usize, 1usize, 864usize],
            t_82,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_420,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[226usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_420, t_421);
        op_ticks[227usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_420, t_421, t_422, 27usize);
        op_ticks[228usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_422,
            [1usize, 1usize, 1usize, 27usize],
            t_81,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_423,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[229usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_423, t_424);
        op_ticks[230usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_414, t_424, t_425, 864usize);
        op_ticks[231usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_425,
            [1usize, 6usize, 1usize, 864usize],
            t_80,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_426,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[232usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_426, t_408, t_427, 648usize);
        op_ticks[233usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_427,
            [1usize, 6usize, 1usize, 108usize],
            t_79,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_19),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_428,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[234usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_428, t_429);
        op_ticks[235usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_428, t_429, t_430, 5184usize);
        op_ticks[236usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_430,
            [1usize, 6usize, 1usize, 864usize],
            t_18,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_17),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_431,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[237usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_431, t_432);
        op_ticks[238usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_431, t_432, t_433, 5184usize);
        op_ticks[239usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_433, t_434);
        op_ticks[240usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_434, t_438);
        op_ticks[241usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_438,
            [1usize, 1usize, 1usize, 864usize],
            t_78,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_439,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[242usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_439, t_440);
        op_ticks[243usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_439, t_440, t_441, 27usize);
        op_ticks[244usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_441,
            [1usize, 1usize, 1usize, 27usize],
            t_77,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_442,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[245usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_442, t_443);
        op_ticks[246usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_433, t_443, t_444, 864usize);
        op_ticks[247usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_444,
            [1usize, 6usize, 1usize, 864usize],
            t_76,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_445,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[248usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_445, t_427, t_446, 648usize);
        op_ticks[249usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_446,
            [1usize, 6usize, 1usize, 108usize],
            t_75,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_16),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_447,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[250usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_447, t_448);
        op_ticks[251usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_447, t_448, t_449, 5184usize);
        op_ticks[252usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_449,
            [1usize, 6usize, 1usize, 864usize],
            t_15,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_14),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_450,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[253usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_450, t_451);
        op_ticks[254usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_450, t_451, t_452, 5184usize);
        op_ticks[255usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_452, t_453);
        op_ticks[256usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_453, t_457);
        op_ticks[257usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_457,
            [1usize, 1usize, 1usize, 864usize],
            t_74,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_458,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[258usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_458, t_459);
        op_ticks[259usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_458, t_459, t_460, 27usize);
        op_ticks[260usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_460,
            [1usize, 1usize, 1usize, 27usize],
            t_73,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_461,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[261usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_461, t_462);
        op_ticks[262usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_452, t_462, t_463, 864usize);
        op_ticks[263usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_463,
            [1usize, 6usize, 1usize, 864usize],
            t_72,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_464,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[264usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_464, t_446, t_465, 648usize);
        op_ticks[265usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_465,
            [1usize, 6usize, 1usize, 108usize],
            t_71,
            [1536usize, 1usize, 1usize, 108usize],
            Some(t_13),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_466,
            [1usize, 6usize, 1usize, 1536usize],
        );
        op_ticks[266usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_466, t_467);
        op_ticks[267usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_466, t_467, t_468, 9216usize);
        op_ticks[268usize] += get_tick() - __t0;
        let __t0 = get_tick();
        pad(
            t_468,
            [1usize, 6usize, 1usize, 1536usize],
            t_469,
            [1usize, 8usize, 3usize, 1536usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[269usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_469,
            [1usize, 8usize, 3usize, 1536usize],
            t_12,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_11),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_470,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[270usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_470, t_471);
        op_ticks[271usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_470, t_471, t_472, 4608usize);
        op_ticks[272usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_472, t_473);
        op_ticks[273usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_473, t_477);
        op_ticks[274usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_477,
            [1usize, 1usize, 1usize, 1536usize],
            t_69,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_478,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[275usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_478, t_479);
        op_ticks[276usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_478, t_479, t_480, 48usize);
        op_ticks[277usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_480,
            [1usize, 1usize, 1usize, 48usize],
            t_67,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_481,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[278usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_481, t_482);
        op_ticks[279usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_472, t_482, t_483, 1536usize);
        op_ticks[280usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_483,
            [1usize, 3usize, 1usize, 1536usize],
            t_66,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_484,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[281usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_484,
            [1usize, 3usize, 1usize, 192usize],
            t_64,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_10),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_485,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[282usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_485, t_486);
        op_ticks[283usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_485, t_486, t_487, 4608usize);
        op_ticks[284usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_487,
            [1usize, 3usize, 1usize, 1536usize],
            t_9,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_8),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_488,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[285usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_488, t_489);
        op_ticks[286usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_488, t_489, t_490, 4608usize);
        op_ticks[287usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_490, t_491);
        op_ticks[288usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_491, t_495);
        op_ticks[289usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_495,
            [1usize, 1usize, 1usize, 1536usize],
            t_63,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_496,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[290usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_496, t_497);
        op_ticks[291usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_496, t_497, t_498, 48usize);
        op_ticks[292usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_498,
            [1usize, 1usize, 1usize, 48usize],
            t_62,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_499,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[293usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_499, t_500);
        op_ticks[294usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_490, t_500, t_501, 1536usize);
        op_ticks[295usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_501,
            [1usize, 3usize, 1usize, 1536usize],
            t_61,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_502,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[296usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_502, t_484, t_503, 576usize);
        op_ticks[297usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_503,
            [1usize, 3usize, 1usize, 192usize],
            t_60,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_7),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_504,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[298usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_504, t_505);
        op_ticks[299usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_504, t_505, t_506, 4608usize);
        op_ticks[300usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_506,
            [1usize, 3usize, 1usize, 1536usize],
            t_6,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_5),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_507,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[301usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_507, t_508);
        op_ticks[302usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_507, t_508, t_509, 4608usize);
        op_ticks[303usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_509, t_510);
        op_ticks[304usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_510, t_514);
        op_ticks[305usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_514,
            [1usize, 1usize, 1usize, 1536usize],
            t_59,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_515,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[306usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_515, t_516);
        op_ticks[307usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_515, t_516, t_517, 48usize);
        op_ticks[308usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_517,
            [1usize, 1usize, 1usize, 48usize],
            t_58,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_518,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[309usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_518, t_519);
        op_ticks[310usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_509, t_519, t_520, 1536usize);
        op_ticks[311usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_520,
            [1usize, 3usize, 1usize, 1536usize],
            t_57,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_521,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[312usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_521, t_503, t_522, 576usize);
        op_ticks[313usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_522,
            [1usize, 3usize, 1usize, 192usize],
            t_56,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_4),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_523,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[314usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_523, t_524);
        op_ticks[315usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_523, t_524, t_525, 4608usize);
        op_ticks[316usize] += get_tick() - __t0;
        let __t0 = get_tick();
        depthwise_conv2d(
            t_525,
            [1usize, 3usize, 1usize, 1536usize],
            t_3,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_2),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_526,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[317usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_526, t_527);
        op_ticks[318usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_526, t_527, t_528, 4608usize);
        op_ticks[319usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_528, t_529);
        op_ticks[320usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reshape(t_529, t_533);
        op_ticks[321usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_533,
            [1usize, 1usize, 1usize, 1536usize],
            t_55,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_534,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[322usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_534, t_535);
        op_ticks[323usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_534, t_535, t_536, 48usize);
        op_ticks[324usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_536,
            [1usize, 1usize, 1usize, 48usize],
            t_54,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_537,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[325usize] += get_tick() - __t0;
        let __t0 = get_tick();
        unary_logistic(t_537, t_538);
        op_ticks[326usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_528, t_538, t_539, 1536usize);
        op_ticks[327usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d(
            t_539,
            [1usize, 3usize, 1usize, 1536usize],
            t_53,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_540,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[328usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_540, t_522, t_541, 576usize);
        op_ticks[329usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_mul(t_541, t_161, t_542, 192usize);
        op_ticks[330usize] += get_tick() - __t0;
        let __t0 = get_tick();
        binary_add(t_542, t_160, t_543, 192usize);
        op_ticks[331usize] += get_tick() - __t0;
        let __t0 = get_tick();
        conv2d_relu(
            t_543,
            [1usize, 3usize, 1usize, 192usize],
            t_52,
            [1024usize, 3usize, 3usize, 192usize],
            Some(t_1),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_544,
            [1usize, 1usize, 1usize, 1024usize],
        );
        op_ticks[332usize] += get_tick() - __t0;
        let __t0 = get_tick();
        reduce_mean_hw(t_544, t_545);
        op_ticks[333usize] += get_tick() - __t0;
        let __t0 = get_tick();
        fully_connected(t_545, 1024usize, t_164, Some(t_142), t_546, 6522usize);
        op_ticks[334usize] += get_tick() - __t0;
    }
    output.copy_from_slice(&t_546);
}
/// Instrumented inference with per-op hardware profiling counters.
pub fn forward_profiled(
    input: &[f32; 144000usize],
    output: &mut [f32; 6522usize],
    op_ticks: &mut [u64; NUM_OPS],
    #[allow(unused)]
    op_profile: &mut [psp_ml::profiler::OpProfileStats; NUM_OPS],
    get_tick: fn() -> u64,
) {
    let t_169 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1usize,
        )
    };
    let t_170 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(4usize),
            144000usize,
        )
    };
    let t_171 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            1usize,
        )
    };
    let t_172 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144004usize),
            1usize,
        )
    };
    let t_173 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144008usize),
            144000usize,
        )
    };
    let t_174 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_175 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144000usize),
            144000usize,
        )
    };
    let t_187 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_188 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288008usize),
            144000usize,
        )
    };
    let t_200 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(432008usize),
            1046528usize,
        )
    };
    let t_201 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1478536usize),
            1046528usize,
        )
    };
    let t_202 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            1046528usize,
        )
    };
    let t_206 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(1334528usize),
            523775usize,
        )
    };
    let t_214 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            523775usize,
        )
    };
    let t_215 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_216 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    let t_217 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_218 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    let t_219 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            49056usize,
        )
    };
    let t_220 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(49056usize),
            49056usize,
        )
    };
    static mut T_221_BUF: Aligned16<49056usize> = Aligned16([0.0f32; 49056usize]);
    let t_221 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_221_BUF) as *mut f32,
            49056usize,
        )
    };
    let t_228 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            144000usize,
        )
    };
    let t_229 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(144000usize),
            144000usize,
        )
    };
    let t_241 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(288000usize),
            523264usize,
        )
    };
    let t_242 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(811264usize),
            523264usize,
        )
    };
    let t_243 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            523264usize,
        )
    };
    static mut T_247_BUF: Aligned16<262143usize> = Aligned16([0.0f32; 262143usize]);
    let t_247 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_247_BUF) as *mut f32,
            262143usize,
        )
    };
    static mut T_546_BUF: Aligned16<6522usize> = Aligned16([0.0f32; 6522usize]);
    let t_546 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_546_BUF) as *mut f32,
            6522usize,
        )
    };
    let tensor_data = tensor_data_f32();
    let t_1 = &tensor_data[T_1_OFFSET..T_1_OFFSET + T_1_LEN];
    let t_2 = &tensor_data[T_2_OFFSET..T_2_OFFSET + T_2_LEN];
    let t_3 = &tensor_data[T_3_OFFSET..T_3_OFFSET + T_3_LEN];
    let t_4 = &tensor_data[T_4_OFFSET..T_4_OFFSET + T_4_LEN];
    let t_5 = &tensor_data[T_5_OFFSET..T_5_OFFSET + T_5_LEN];
    let t_6 = &tensor_data[T_6_OFFSET..T_6_OFFSET + T_6_LEN];
    let t_7 = &tensor_data[T_7_OFFSET..T_7_OFFSET + T_7_LEN];
    let t_8 = &tensor_data[T_8_OFFSET..T_8_OFFSET + T_8_LEN];
    let t_9 = &tensor_data[T_9_OFFSET..T_9_OFFSET + T_9_LEN];
    let t_10 = &tensor_data[T_10_OFFSET..T_10_OFFSET + T_10_LEN];
    let t_11 = &tensor_data[T_11_OFFSET..T_11_OFFSET + T_11_LEN];
    let t_12 = &tensor_data[T_12_OFFSET..T_12_OFFSET + T_12_LEN];
    let t_13 = &tensor_data[T_13_OFFSET..T_13_OFFSET + T_13_LEN];
    let t_14 = &tensor_data[T_14_OFFSET..T_14_OFFSET + T_14_LEN];
    let t_15 = &tensor_data[T_15_OFFSET..T_15_OFFSET + T_15_LEN];
    let t_16 = &tensor_data[T_16_OFFSET..T_16_OFFSET + T_16_LEN];
    let t_17 = &tensor_data[T_17_OFFSET..T_17_OFFSET + T_17_LEN];
    let t_18 = &tensor_data[T_18_OFFSET..T_18_OFFSET + T_18_LEN];
    let t_19 = &tensor_data[T_19_OFFSET..T_19_OFFSET + T_19_LEN];
    let t_20 = &tensor_data[T_20_OFFSET..T_20_OFFSET + T_20_LEN];
    let t_21 = &tensor_data[T_21_OFFSET..T_21_OFFSET + T_21_LEN];
    let t_22 = &tensor_data[T_22_OFFSET..T_22_OFFSET + T_22_LEN];
    let t_23 = &tensor_data[T_23_OFFSET..T_23_OFFSET + T_23_LEN];
    let t_24 = &tensor_data[T_24_OFFSET..T_24_OFFSET + T_24_LEN];
    let t_25 = &tensor_data[T_25_OFFSET..T_25_OFFSET + T_25_LEN];
    let t_26 = &tensor_data[T_26_OFFSET..T_26_OFFSET + T_26_LEN];
    let t_27 = &tensor_data[T_27_OFFSET..T_27_OFFSET + T_27_LEN];
    let t_28 = &tensor_data[T_28_OFFSET..T_28_OFFSET + T_28_LEN];
    let t_29 = &tensor_data[T_29_OFFSET..T_29_OFFSET + T_29_LEN];
    let t_30 = &tensor_data[T_30_OFFSET..T_30_OFFSET + T_30_LEN];
    let t_31 = &tensor_data[T_31_OFFSET..T_31_OFFSET + T_31_LEN];
    let t_32 = &tensor_data[T_32_OFFSET..T_32_OFFSET + T_32_LEN];
    let t_33 = &tensor_data[T_33_OFFSET..T_33_OFFSET + T_33_LEN];
    let t_34 = &tensor_data[T_34_OFFSET..T_34_OFFSET + T_34_LEN];
    let t_35 = &tensor_data[T_35_OFFSET..T_35_OFFSET + T_35_LEN];
    let t_36 = &tensor_data[T_36_OFFSET..T_36_OFFSET + T_36_LEN];
    let t_37 = &tensor_data[T_37_OFFSET..T_37_OFFSET + T_37_LEN];
    let t_38 = &tensor_data[T_38_OFFSET..T_38_OFFSET + T_38_LEN];
    let t_39 = &tensor_data[T_39_OFFSET..T_39_OFFSET + T_39_LEN];
    let t_40 = &tensor_data[T_40_OFFSET..T_40_OFFSET + T_40_LEN];
    let t_41 = &tensor_data[T_41_OFFSET..T_41_OFFSET + T_41_LEN];
    let t_42 = &tensor_data[T_42_OFFSET..T_42_OFFSET + T_42_LEN];
    let t_43 = &tensor_data[T_43_OFFSET..T_43_OFFSET + T_43_LEN];
    let t_44 = &tensor_data[T_44_OFFSET..T_44_OFFSET + T_44_LEN];
    let t_45 = &tensor_data[T_45_OFFSET..T_45_OFFSET + T_45_LEN];
    let t_46 = &tensor_data[T_46_OFFSET..T_46_OFFSET + T_46_LEN];
    let t_47 = &tensor_data[T_47_OFFSET..T_47_OFFSET + T_47_LEN];
    let t_48 = &tensor_data[T_48_OFFSET..T_48_OFFSET + T_48_LEN];
    let t_49 = &tensor_data[T_49_OFFSET..T_49_OFFSET + T_49_LEN];
    let t_50 = &tensor_data[T_50_OFFSET..T_50_OFFSET + T_50_LEN];
    let t_51 = &tensor_data[T_51_OFFSET..T_51_OFFSET + T_51_LEN];
    let t_52 = &tensor_data[T_52_OFFSET..T_52_OFFSET + T_52_LEN];
    let t_53 = &tensor_data[T_53_OFFSET..T_53_OFFSET + T_53_LEN];
    let t_54 = &tensor_data[T_54_OFFSET..T_54_OFFSET + T_54_LEN];
    let t_55 = &tensor_data[T_55_OFFSET..T_55_OFFSET + T_55_LEN];
    let t_56 = &tensor_data[T_56_OFFSET..T_56_OFFSET + T_56_LEN];
    let t_57 = &tensor_data[T_57_OFFSET..T_57_OFFSET + T_57_LEN];
    let t_58 = &tensor_data[T_58_OFFSET..T_58_OFFSET + T_58_LEN];
    let t_59 = &tensor_data[T_59_OFFSET..T_59_OFFSET + T_59_LEN];
    let t_60 = &tensor_data[T_60_OFFSET..T_60_OFFSET + T_60_LEN];
    let t_61 = &tensor_data[T_61_OFFSET..T_61_OFFSET + T_61_LEN];
    let t_62 = &tensor_data[T_62_OFFSET..T_62_OFFSET + T_62_LEN];
    let t_63 = &tensor_data[T_63_OFFSET..T_63_OFFSET + T_63_LEN];
    let t_64 = &tensor_data[T_64_OFFSET..T_64_OFFSET + T_64_LEN];
    let t_65 = &tensor_data[T_65_OFFSET..T_65_OFFSET + T_65_LEN];
    let t_66 = &tensor_data[T_66_OFFSET..T_66_OFFSET + T_66_LEN];
    let t_67 = &tensor_data[T_67_OFFSET..T_67_OFFSET + T_67_LEN];
    let t_68 = &tensor_data[T_68_OFFSET..T_68_OFFSET + T_68_LEN];
    let t_69 = &tensor_data[T_69_OFFSET..T_69_OFFSET + T_69_LEN];
    let t_70 = &tensor_data[T_70_OFFSET..T_70_OFFSET + T_70_LEN];
    let t_71 = &tensor_data[T_71_OFFSET..T_71_OFFSET + T_71_LEN];
    let t_72 = &tensor_data[T_72_OFFSET..T_72_OFFSET + T_72_LEN];
    let t_73 = &tensor_data[T_73_OFFSET..T_73_OFFSET + T_73_LEN];
    let t_74 = &tensor_data[T_74_OFFSET..T_74_OFFSET + T_74_LEN];
    let t_75 = &tensor_data[T_75_OFFSET..T_75_OFFSET + T_75_LEN];
    let t_76 = &tensor_data[T_76_OFFSET..T_76_OFFSET + T_76_LEN];
    let t_77 = &tensor_data[T_77_OFFSET..T_77_OFFSET + T_77_LEN];
    let t_78 = &tensor_data[T_78_OFFSET..T_78_OFFSET + T_78_LEN];
    let t_79 = &tensor_data[T_79_OFFSET..T_79_OFFSET + T_79_LEN];
    let t_80 = &tensor_data[T_80_OFFSET..T_80_OFFSET + T_80_LEN];
    let t_81 = &tensor_data[T_81_OFFSET..T_81_OFFSET + T_81_LEN];
    let t_82 = &tensor_data[T_82_OFFSET..T_82_OFFSET + T_82_LEN];
    let t_83 = &tensor_data[T_83_OFFSET..T_83_OFFSET + T_83_LEN];
    let t_84 = &tensor_data[T_84_OFFSET..T_84_OFFSET + T_84_LEN];
    let t_85 = &tensor_data[T_85_OFFSET..T_85_OFFSET + T_85_LEN];
    let t_86 = &tensor_data[T_86_OFFSET..T_86_OFFSET + T_86_LEN];
    let t_87 = &tensor_data[T_87_OFFSET..T_87_OFFSET + T_87_LEN];
    let t_88 = &tensor_data[T_88_OFFSET..T_88_OFFSET + T_88_LEN];
    let t_89 = &tensor_data[T_89_OFFSET..T_89_OFFSET + T_89_LEN];
    let t_90 = &tensor_data[T_90_OFFSET..T_90_OFFSET + T_90_LEN];
    let t_91 = &tensor_data[T_91_OFFSET..T_91_OFFSET + T_91_LEN];
    let t_92 = &tensor_data[T_92_OFFSET..T_92_OFFSET + T_92_LEN];
    let t_93 = &tensor_data[T_93_OFFSET..T_93_OFFSET + T_93_LEN];
    let t_94 = &tensor_data[T_94_OFFSET..T_94_OFFSET + T_94_LEN];
    let t_95 = &tensor_data[T_95_OFFSET..T_95_OFFSET + T_95_LEN];
    let t_96 = &tensor_data[T_96_OFFSET..T_96_OFFSET + T_96_LEN];
    let t_97 = &tensor_data[T_97_OFFSET..T_97_OFFSET + T_97_LEN];
    let t_98 = &tensor_data[T_98_OFFSET..T_98_OFFSET + T_98_LEN];
    let t_99 = &tensor_data[T_99_OFFSET..T_99_OFFSET + T_99_LEN];
    let t_100 = &tensor_data[T_100_OFFSET..T_100_OFFSET + T_100_LEN];
    let t_101 = &tensor_data[T_101_OFFSET..T_101_OFFSET + T_101_LEN];
    let t_102 = &tensor_data[T_102_OFFSET..T_102_OFFSET + T_102_LEN];
    let t_103 = &tensor_data[T_103_OFFSET..T_103_OFFSET + T_103_LEN];
    let t_104 = &tensor_data[T_104_OFFSET..T_104_OFFSET + T_104_LEN];
    let t_105 = &tensor_data[T_105_OFFSET..T_105_OFFSET + T_105_LEN];
    let t_106 = &tensor_data[T_106_OFFSET..T_106_OFFSET + T_106_LEN];
    let t_107 = &tensor_data[T_107_OFFSET..T_107_OFFSET + T_107_LEN];
    let t_108 = &tensor_data[T_108_OFFSET..T_108_OFFSET + T_108_LEN];
    let t_109 = &tensor_data[T_109_OFFSET..T_109_OFFSET + T_109_LEN];
    let t_110 = &tensor_data[T_110_OFFSET..T_110_OFFSET + T_110_LEN];
    let t_111 = &tensor_data[T_111_OFFSET..T_111_OFFSET + T_111_LEN];
    let t_112 = &tensor_data[T_112_OFFSET..T_112_OFFSET + T_112_LEN];
    let t_113 = &tensor_data[T_113_OFFSET..T_113_OFFSET + T_113_LEN];
    let t_114 = &tensor_data[T_114_OFFSET..T_114_OFFSET + T_114_LEN];
    let t_115 = &tensor_data[T_115_OFFSET..T_115_OFFSET + T_115_LEN];
    let t_116 = &tensor_data[T_116_OFFSET..T_116_OFFSET + T_116_LEN];
    let t_117 = &tensor_data[T_117_OFFSET..T_117_OFFSET + T_117_LEN];
    let t_118 = &tensor_data[T_118_OFFSET..T_118_OFFSET + T_118_LEN];
    let t_119 = &tensor_data[T_119_OFFSET..T_119_OFFSET + T_119_LEN];
    let t_120 = &tensor_data[T_120_OFFSET..T_120_OFFSET + T_120_LEN];
    let t_121 = &tensor_data[T_121_OFFSET..T_121_OFFSET + T_121_LEN];
    let t_122 = &tensor_data[T_122_OFFSET..T_122_OFFSET + T_122_LEN];
    let t_126 = &tensor_data[T_126_OFFSET..T_126_OFFSET + T_126_LEN];
    let t_129 = &tensor_data[T_129_OFFSET..T_129_OFFSET + T_129_LEN];
    let t_133 = &tensor_data[T_133_OFFSET..T_133_OFFSET + T_133_LEN];
    let t_134 = &tensor_data[T_134_OFFSET..T_134_OFFSET + T_134_LEN];
    let t_135 = &tensor_data[T_135_OFFSET..T_135_OFFSET + T_135_LEN];
    let t_136 = &tensor_data[T_136_OFFSET..T_136_OFFSET + T_136_LEN];
    let t_142 = &tensor_data[T_142_OFFSET..T_142_OFFSET + T_142_LEN];
    let t_144 = &tensor_data[T_144_OFFSET..T_144_OFFSET + T_144_LEN];
    let t_145 = &tensor_data[T_145_OFFSET..T_145_OFFSET + T_145_LEN];
    let t_146 = &tensor_data[T_146_OFFSET..T_146_OFFSET + T_146_LEN];
    let t_150 = &tensor_data[T_150_OFFSET..T_150_OFFSET + T_150_LEN];
    let t_153 = &tensor_data[T_153_OFFSET..T_153_OFFSET + T_153_LEN];
    let t_156 = &tensor_data[T_156_OFFSET..T_156_OFFSET + T_156_LEN];
    let t_160 = &tensor_data[T_160_OFFSET..T_160_OFFSET + T_160_LEN];
    let t_161 = &tensor_data[T_161_OFFSET..T_161_OFFSET + T_161_LEN];
    let t_162 = &tensor_data[T_162_OFFSET..T_162_OFFSET + T_162_LEN];
    let t_163 = &tensor_data[T_163_OFFSET..T_163_OFFSET + T_163_LEN];
    let t_164 = &tensor_data[T_164_OFFSET..T_164_OFFSET + T_164_LEN];
    let t_165 = &tensor_data[T_165_OFFSET..T_165_OFFSET + T_165_LEN];
    let t_166 = &tensor_data[T_166_OFFSET..T_166_OFFSET + T_166_LEN];
    let t_167 = &tensor_data[T_167_OFFSET..T_167_OFFSET + T_167_LEN];
    let t_168 = &tensor_data[T_168_OFFSET..T_168_OFFSET + T_168_LEN];
    let t_185 = &tensor_data[T_185_OFFSET..T_185_OFFSET + T_185_LEN];
    let t_199 = &tensor_data[T_199_OFFSET..T_199_OFFSET + T_199_LEN];
    let t_226 = &tensor_data[T_226_OFFSET..T_226_OFFSET + T_226_LEN];
    let t_240 = &tensor_data[T_240_OFFSET..T_240_OFFSET + T_240_LEN];
    let t_549 = &tensor_data[T_549_OFFSET..T_549_OFFSET + T_549_LEN];
    let t_550 = &tensor_data[T_550_OFFSET..T_550_OFFSET + T_550_LEN];
    let t_551 = &tensor_data[T_551_OFFSET..T_551_OFFSET + T_551_LEN];
    let t_552 = &tensor_data[T_552_OFFSET..T_552_OFFSET + T_552_LEN];
    let t_553 = &tensor_data[T_553_OFFSET..T_553_OFFSET + T_553_LEN];
    let t_554 = &tensor_data[T_554_OFFSET..T_554_OFFSET + T_554_LEN];
    let t_555 = &tensor_data[T_555_OFFSET..T_555_OFFSET + T_555_LEN];
    let t_556 = &tensor_data[T_556_OFFSET..T_556_OFFSET + T_556_LEN];
    let t_557 = &tensor_data[T_557_OFFSET..T_557_OFFSET + T_557_LEN];
    let t_558 = &tensor_data[T_558_OFFSET..T_558_OFFSET + T_558_LEN];
    let t_559 = &tensor_data[T_559_OFFSET..T_559_OFFSET + T_559_LEN];
    let t_560 = &tensor_data[T_560_OFFSET..T_560_OFFSET + T_560_LEN];
    let t_561 = &tensor_data[T_561_OFFSET..T_561_OFFSET + T_561_LEN];
    let t_562 = &tensor_data[T_562_OFFSET..T_562_OFFSET + T_562_LEN];
    let t_563 = &tensor_data[T_563_OFFSET..T_563_OFFSET + T_563_LEN];
    let t_564 = &tensor_data[T_564_OFFSET..T_564_OFFSET + T_564_LEN];
    let t_565 = &tensor_data[T_565_OFFSET..T_565_OFFSET + T_565_LEN];
    let t_566 = &tensor_data[T_566_OFFSET..T_566_OFFSET + T_566_LEN];
    let t_567 = &tensor_data[T_567_OFFSET..T_567_OFFSET + T_567_LEN];
    let t_568 = &tensor_data[T_568_OFFSET..T_568_OFFSET + T_568_LEN];
    let t_569 = &tensor_data[T_569_OFFSET..T_569_OFFSET + T_569_LEN];
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reduce_min(input, t_169);
    op_ticks[0usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[0usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_sub(input, t_169, t_170, 1usize);
    op_ticks[1usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[1usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reduce_max(t_170, t_171);
    op_ticks[2usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[2usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_add(t_171, t_150, t_172, 1usize);
    op_ticks[3usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[3usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_div(t_170, t_172, t_173, 1usize);
    op_ticks[4usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[4usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_sub(t_173, t_146, t_174, 1usize);
    op_ticks[5usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[5usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_mul(t_174, t_144, t_175, 1usize);
    op_ticks[6usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[6usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    strided_slice(
        t_175,
        &[1usize, 144000usize],
        t_187,
        &[0i32, 0i32],
        &[1i32, 144000i32],
        &[1i32, 1i32],
        0i32,
        0i32,
        0i32,
    );
    op_ticks[7usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[7usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_187, t_188);
    op_ticks[8usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[8usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    gather(
        t_188,
        &[1usize, 72000usize, 2usize],
        unsafe { core::slice::from_raw_parts(t_199.as_ptr() as *const i32, 1024usize) },
        t_200,
        &[1usize, 511usize, 1024usize, 2usize],
        1usize,
    );
    op_ticks[9usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[9usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_200, t_201);
    op_ticks[10usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[10usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_mul(t_201, t_129, t_202, 2048usize);
    op_ticks[11usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[11usize].accumulate(__regs.assume_init_ref());
    }
    let scratch_12_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
            2048usize,
        )
    };
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    rfft_pack(t_202, scratch_12_0, 2048usize);
    op_ticks[12usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[12usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_549, 1024usize, 1usize);
    op_ticks[13usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[13usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_550, 1024usize, 2usize);
    op_ticks[14usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[14usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_551, 1024usize, 4usize);
    op_ticks[15usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[15usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_552, 1024usize, 8usize);
    op_ticks[16usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[16usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_553, 1024usize, 16usize);
    op_ticks[17usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[17usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_554, 1024usize, 32usize);
    op_ticks[18usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[18usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_555, 1024usize, 64usize);
    op_ticks[19usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[19usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_556, 1024usize, 128usize);
    op_ticks[20usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[20usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_557, 1024usize, 256usize);
    op_ticks[21usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[21usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_12_0, t_558, 1024usize, 512usize);
    op_ticks[22usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[22usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    rfft_unpack(scratch_12_0, t_559, t_206, 2048usize);
    op_ticks[23usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[23usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_206, t_214);
    op_ticks[24usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[24usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    for _batch_idx in 0..511usize {
        let _in_off = _batch_idx * 1025usize;
        let _out_off = _batch_idx * 96usize;
        fully_connected(
            &t_214[_in_off.._in_off + 1025usize],
            1025usize,
            t_166,
            None,
            &mut t_215[_out_off.._out_off + 96usize],
            96usize,
        );
    }
    op_ticks[25usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[25usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_215, t_216);
    op_ticks[26usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[26usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_mul(t_216, t_216, t_217, 49056usize);
    op_ticks[27usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[27usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_pow(t_217, t_165, t_218, 1usize);
    op_ticks[28usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[28usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reverse_v2(t_218, &[511usize, 1usize, 96usize], t_219, 2usize);
    op_ticks[29usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[29usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    transpose(
        t_219,
        &[511usize, 1usize, 96usize],
        t_220,
        &[511usize, 96usize, 1usize],
        &[0usize, 2usize, 1usize],
    );
    op_ticks[30usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[30usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_220, t_221);
    op_ticks[31usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[31usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    strided_slice(
        t_175,
        &[1usize, 144000usize],
        t_228,
        &[0i32, 0i32],
        &[1i32, 144000i32],
        &[1i32, 1i32],
        0i32,
        0i32,
        0i32,
    );
    op_ticks[32usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[32usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_228, t_229);
    op_ticks[33usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[33usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    gather(
        t_229,
        &[1usize, 18000usize, 8usize],
        unsafe { core::slice::from_raw_parts(t_240.as_ptr() as *const i32, 511usize) },
        t_241,
        &[1usize, 511usize, 128usize, 8usize],
        1usize,
    );
    op_ticks[34usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[34usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    reshape(t_241, t_242);
    op_ticks[35usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[35usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    binary_mul(t_242, t_126, t_243, 1024usize);
    op_ticks[36usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[36usize].accumulate(__regs.assume_init_ref());
    }
    let scratch_26_0 = unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(523264usize),
            1024usize,
        )
    };
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    rfft_pack(t_243, scratch_26_0, 1024usize);
    op_ticks[37usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[37usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_560, 512usize, 1usize);
    op_ticks[38usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[38usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_561, 512usize, 2usize);
    op_ticks[39usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[39usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_562, 512usize, 4usize);
    op_ticks[40usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[40usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_563, 512usize, 8usize);
    op_ticks[41usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[41usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_564, 512usize, 16usize);
    op_ticks[42usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[42usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_565, 512usize, 32usize);
    op_ticks[43usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[43usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_566, 512usize, 64usize);
    op_ticks[44usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[44usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_567, 512usize, 128usize);
    op_ticks[45usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[45usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_26_0, t_568, 512usize, 256usize);
    op_ticks[46usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[46usize].accumulate(__regs.assume_init_ref());
    }
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileClear();
        psp_ml::profiler::ProfileEnable();
    }
    let __t0 = get_tick();
    rfft_unpack(scratch_26_0, t_569, t_247, 1024usize);
    op_ticks[47usize] += get_tick() - __t0;
    #[cfg(target_os = "psp")]
    unsafe {
        psp_ml::profiler::ProfileDisable();
        let mut __regs = core::mem::MaybeUninit::<
            psp_ml::profiler::ProfileRegs,
        >::zeroed();
        psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
        op_profile[47usize].accumulate(__regs.assume_init_ref());
    }
    for _frame_idx in 0..511usize {
        let t_255 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(661880usize),
                513usize,
            )
        };
        let t_256 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667580usize),
                96usize,
            )
        };
        let t_257 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667676usize),
                96usize,
            )
        };
        let t_258 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667772usize),
                96usize,
            )
        };
        let t_259 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667868usize),
                96usize,
            )
        };
        let t_260 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667964usize),
                96usize,
            )
        };
        let t_261 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668060usize),
                96usize,
            )
        };
        let t_262 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668156usize),
                96usize,
            )
        };
        let t_263 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667004usize),
                192usize,
            )
        };
        let t_264 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667196usize),
                192usize,
            )
        };
        let t_265 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(667388usize),
                192usize,
            )
        };
        let t_266 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(616560usize),
                1152usize,
            )
        };
        let t_267 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(617712usize),
                1152usize,
            )
        };
        let t_268 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(618864usize),
                1152usize,
            )
        };
        let t_269 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(563760usize),
                2304usize,
            )
        };
        let t_270 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(620016usize),
                1152usize,
            )
        };
        let t_271 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(466992usize),
                3456usize,
            )
        };
        let t_272 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(470448usize),
                3456usize,
            )
        };
        let t_273 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(473904usize),
                3456usize,
            )
        };
        let t_274 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(95616usize),
                10800usize,
            )
        };
        let t_275 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(566064usize),
                1728usize,
            )
        };
        let t_276 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(567792usize),
                1728usize,
            )
        };
        let t_277 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(569520usize),
                1728usize,
            )
        };
        let t_278 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(623216usize),
                864usize,
            )
        };
        let t_279 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(571248usize),
                1728usize,
            )
        };
        let t_280 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(572976usize),
                1728usize,
            )
        };
        let t_281 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(574704usize),
                1728usize,
            )
        };
        let t_282 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(576432usize),
                1728usize,
            )
        };
        let t_283 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(578160usize),
                1728usize,
            )
        };
        let t_284 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(579888usize),
                1728usize,
            )
        };
        let t_285 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(624080usize),
                864usize,
            )
        };
        let t_286 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(624944usize),
                864usize,
            )
        };
        let t_287 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(581616usize),
                1728usize,
            )
        };
        let t_288 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(583344usize),
                1728usize,
            )
        };
        let t_289 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(585072usize),
                1728usize,
            )
        };
        let t_290 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(586800usize),
                1728usize,
            )
        };
        let t_291 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(588528usize),
                1728usize,
            )
        };
        let t_292 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(590256usize),
                1728usize,
            )
        };
        let t_293 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(625808usize),
                864usize,
            )
        };
        let t_294 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(626672usize),
                864usize,
            )
        };
        let t_295 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(165168usize),
                6912usize,
            )
        };
        let t_296 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(172080usize),
                6912usize,
            )
        };
        let t_297 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(178992usize),
                6912usize,
            )
        };
        let t_298 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(73152usize),
                22464usize,
            )
        };
        let t_299 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(477360usize),
                3456usize,
            )
        };
        let t_300 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(480816usize),
                3456usize,
            )
        };
        let t_301 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(484272usize),
                3456usize,
            )
        };
        let t_302 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662396usize),
                288usize,
            )
        };
        let t_306 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662684usize),
                288usize,
            )
        };
        let t_307 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669248usize),
                18usize,
            )
        };
        let t_308 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669268usize),
                18usize,
            )
        };
        let t_309 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669288usize),
                18usize,
            )
        };
        let t_310 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(662972usize),
                288usize,
            )
        };
        let t_311 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663260usize),
                288usize,
            )
        };
        let t_312 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(487728usize),
                3456usize,
            )
        };
        let t_313 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(627536usize),
                864usize,
            )
        };
        let t_314 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(491184usize),
                3456usize,
            )
        };
        let t_315 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(494640usize),
                3456usize,
            )
        };
        let t_316 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(498096usize),
                3456usize,
            )
        };
        let t_317 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(501552usize),
                3456usize,
            )
        };
        let t_318 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(505008usize),
                3456usize,
            )
        };
        let t_319 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(508464usize),
                3456usize,
            )
        };
        let t_320 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663548usize),
                288usize,
            )
        };
        let t_324 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(663836usize),
                288usize,
            )
        };
        let t_325 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669308usize),
                18usize,
            )
        };
        let t_326 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669328usize),
                18usize,
            )
        };
        let t_327 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669348usize),
                18usize,
            )
        };
        let t_328 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664124usize),
                288usize,
            )
        };
        let t_329 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664412usize),
                288usize,
            )
        };
        let t_330 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(511920usize),
                3456usize,
            )
        };
        let t_331 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(628400usize),
                864usize,
            )
        };
        let t_332 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(629264usize),
                864usize,
            )
        };
        let t_333 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(515376usize),
                3456usize,
            )
        };
        let t_334 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(518832usize),
                3456usize,
            )
        };
        let t_335 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(522288usize),
                3456usize,
            )
        };
        let t_336 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(525744usize),
                3456usize,
            )
        };
        let t_337 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(529200usize),
                3456usize,
            )
        };
        let t_338 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(532656usize),
                3456usize,
            )
        };
        let t_339 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664700usize),
                288usize,
            )
        };
        let t_343 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(664988usize),
                288usize,
            )
        };
        let t_344 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669368usize),
                18usize,
            )
        };
        let t_345 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669388usize),
                18usize,
            )
        };
        let t_346 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669408usize),
                18usize,
            )
        };
        let t_347 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665276usize),
                288usize,
            )
        };
        let t_348 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665564usize),
                288usize,
            )
        };
        let t_349 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(536112usize),
                3456usize,
            )
        };
        let t_350 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(630128usize),
                864usize,
            )
        };
        let t_351 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(630992usize),
                864usize,
            )
        };
        let t_352 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(539568usize),
                3456usize,
            )
        };
        let t_353 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(543024usize),
                3456usize,
            )
        };
        let t_354 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(546480usize),
                3456usize,
            )
        };
        let t_355 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(549936usize),
                3456usize,
            )
        };
        let t_356 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(553392usize),
                3456usize,
            )
        };
        let t_357 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(556848usize),
                3456usize,
            )
        };
        let t_358 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(665852usize),
                288usize,
            )
        };
        let t_362 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666140usize),
                288usize,
            )
        };
        let t_363 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669428usize),
                18usize,
            )
        };
        let t_364 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669448usize),
                18usize,
            )
        };
        let t_365 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669468usize),
                18usize,
            )
        };
        let t_366 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666428usize),
                288usize,
            )
        };
        let t_367 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(666716usize),
                288usize,
            )
        };
        let t_368 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(560304usize),
                3456usize,
            )
        };
        let t_369 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(631856usize),
                864usize,
            )
        };
        let t_370 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(632720usize),
                864usize,
            )
        };
        let t_371 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(106416usize),
                10368usize,
            )
        };
        let t_372 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(116784usize),
                10368usize,
            )
        };
        let t_373 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(127152usize),
                10368usize,
            )
        };
        let t_374 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(36864usize),
                36288usize,
            )
        };
        let t_375 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(185904usize),
                5184usize,
            )
        };
        let t_376 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(191088usize),
                5184usize,
            )
        };
        let t_377 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(196272usize),
                5184usize,
            )
        };
        let t_378 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(633584usize),
                864usize,
            )
        };
        let t_382 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(634448usize),
                864usize,
            )
        };
        let t_383 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668828usize),
                27usize,
            )
        };
        let t_384 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668856usize),
                27usize,
            )
        };
        let t_385 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668884usize),
                27usize,
            )
        };
        let t_386 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(635312usize),
                864usize,
            )
        };
        let t_387 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(636176usize),
                864usize,
            )
        };
        let t_388 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(201456usize),
                5184usize,
            )
        };
        let t_389 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(650864usize),
                648usize,
            )
        };
        let t_390 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(206640usize),
                5184usize,
            )
        };
        let t_391 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(211824usize),
                5184usize,
            )
        };
        let t_392 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(217008usize),
                5184usize,
            )
        };
        let t_393 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(222192usize),
                5184usize,
            )
        };
        let t_394 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(227376usize),
                5184usize,
            )
        };
        let t_395 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(232560usize),
                5184usize,
            )
        };
        let t_396 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(637040usize),
                864usize,
            )
        };
        let t_400 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(637904usize),
                864usize,
            )
        };
        let t_401 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668912usize),
                27usize,
            )
        };
        let t_402 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668940usize),
                27usize,
            )
        };
        let t_403 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668968usize),
                27usize,
            )
        };
        let t_404 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(638768usize),
                864usize,
            )
        };
        let t_405 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(639632usize),
                864usize,
            )
        };
        let t_406 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(237744usize),
                5184usize,
            )
        };
        let t_407 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(651512usize),
                648usize,
            )
        };
        let t_408 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(652160usize),
                648usize,
            )
        };
        let t_409 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(242928usize),
                5184usize,
            )
        };
        let t_410 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(248112usize),
                5184usize,
            )
        };
        let t_411 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(253296usize),
                5184usize,
            )
        };
        let t_412 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(258480usize),
                5184usize,
            )
        };
        let t_413 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(263664usize),
                5184usize,
            )
        };
        let t_414 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(268848usize),
                5184usize,
            )
        };
        let t_415 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(640496usize),
                864usize,
            )
        };
        let t_419 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(641360usize),
                864usize,
            )
        };
        let t_420 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668996usize),
                27usize,
            )
        };
        let t_421 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669024usize),
                27usize,
            )
        };
        let t_422 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669052usize),
                27usize,
            )
        };
        let t_423 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(642224usize),
                864usize,
            )
        };
        let t_424 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(643088usize),
                864usize,
            )
        };
        let t_425 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(274032usize),
                5184usize,
            )
        };
        let t_426 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(652808usize),
                648usize,
            )
        };
        let t_427 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(653456usize),
                648usize,
            )
        };
        let t_428 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(279216usize),
                5184usize,
            )
        };
        let t_429 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(284400usize),
                5184usize,
            )
        };
        let t_430 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(289584usize),
                5184usize,
            )
        };
        let t_431 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(294768usize),
                5184usize,
            )
        };
        let t_432 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(299952usize),
                5184usize,
            )
        };
        let t_433 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(305136usize),
                5184usize,
            )
        };
        let t_434 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(643952usize),
                864usize,
            )
        };
        let t_438 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(644816usize),
                864usize,
            )
        };
        let t_439 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669080usize),
                27usize,
            )
        };
        let t_440 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669108usize),
                27usize,
            )
        };
        let t_441 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669136usize),
                27usize,
            )
        };
        let t_442 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(645680usize),
                864usize,
            )
        };
        let t_443 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(646544usize),
                864usize,
            )
        };
        let t_444 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(310320usize),
                5184usize,
            )
        };
        let t_445 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(654104usize),
                648usize,
            )
        };
        let t_446 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(654752usize),
                648usize,
            )
        };
        let t_447 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(315504usize),
                5184usize,
            )
        };
        let t_448 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(320688usize),
                5184usize,
            )
        };
        let t_449 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(325872usize),
                5184usize,
            )
        };
        let t_450 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(331056usize),
                5184usize,
            )
        };
        let t_451 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(336240usize),
                5184usize,
            )
        };
        let t_452 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(341424usize),
                5184usize,
            )
        };
        let t_453 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(647408usize),
                864usize,
            )
        };
        let t_457 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(648272usize),
                864usize,
            )
        };
        let t_458 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669164usize),
                27usize,
            )
        };
        let t_459 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669192usize),
                27usize,
            )
        };
        let t_460 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669220usize),
                27usize,
            )
        };
        let t_461 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(649136usize),
                864usize,
            )
        };
        let t_462 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(650000usize),
                864usize,
            )
        };
        let t_463 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(346608usize),
                5184usize,
            )
        };
        let t_464 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(655400usize),
                648usize,
            )
        };
        let t_465 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(656048usize),
                648usize,
            )
        };
        let t_466 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(137520usize),
                9216usize,
            )
        };
        let t_467 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(146736usize),
                9216usize,
            )
        };
        let t_468 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(155952usize),
                9216usize,
            )
        };
        let t_469 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(0usize),
                36864usize,
            )
        };
        let t_470 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(351792usize),
                4608usize,
            )
        };
        let t_471 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(356400usize),
                4608usize,
            )
        };
        let t_472 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(361008usize),
                4608usize,
            )
        };
        let t_473 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(591984usize),
                1536usize,
            )
        };
        let t_477 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(593520usize),
                1536usize,
            )
        };
        let t_478 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668252usize),
                48usize,
            )
        };
        let t_479 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668300usize),
                48usize,
            )
        };
        let t_480 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668348usize),
                48usize,
            )
        };
        let t_481 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(595056usize),
                1536usize,
            )
        };
        let t_482 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(596592usize),
                1536usize,
            )
        };
        let t_483 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(365616usize),
                4608usize,
            )
        };
        let t_484 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(656696usize),
                576usize,
            )
        };
        let t_485 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(370224usize),
                4608usize,
            )
        };
        let t_486 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(374832usize),
                4608usize,
            )
        };
        let t_487 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(379440usize),
                4608usize,
            )
        };
        let t_488 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(384048usize),
                4608usize,
            )
        };
        let t_489 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(388656usize),
                4608usize,
            )
        };
        let t_490 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(393264usize),
                4608usize,
            )
        };
        let t_491 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(598128usize),
                1536usize,
            )
        };
        let t_495 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(599664usize),
                1536usize,
            )
        };
        let t_496 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668396usize),
                48usize,
            )
        };
        let t_497 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668444usize),
                48usize,
            )
        };
        let t_498 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668492usize),
                48usize,
            )
        };
        let t_499 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(601200usize),
                1536usize,
            )
        };
        let t_500 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(602736usize),
                1536usize,
            )
        };
        let t_501 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(397872usize),
                4608usize,
            )
        };
        let t_502 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(657272usize),
                576usize,
            )
        };
        let t_503 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(657848usize),
                576usize,
            )
        };
        let t_504 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(402480usize),
                4608usize,
            )
        };
        let t_505 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(407088usize),
                4608usize,
            )
        };
        let t_506 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(411696usize),
                4608usize,
            )
        };
        let t_507 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(416304usize),
                4608usize,
            )
        };
        let t_508 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(420912usize),
                4608usize,
            )
        };
        let t_509 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(425520usize),
                4608usize,
            )
        };
        let t_510 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(604272usize),
                1536usize,
            )
        };
        let t_514 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(605808usize),
                1536usize,
            )
        };
        let t_515 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668540usize),
                48usize,
            )
        };
        let t_516 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668588usize),
                48usize,
            )
        };
        let t_517 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668636usize),
                48usize,
            )
        };
        let t_518 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(607344usize),
                1536usize,
            )
        };
        let t_519 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(608880usize),
                1536usize,
            )
        };
        let t_520 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(430128usize),
                4608usize,
            )
        };
        let t_521 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(658424usize),
                576usize,
            )
        };
        let t_522 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(659000usize),
                576usize,
            )
        };
        let t_523 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(434736usize),
                4608usize,
            )
        };
        let t_524 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(439344usize),
                4608usize,
            )
        };
        let t_525 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(443952usize),
                4608usize,
            )
        };
        let t_526 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(448560usize),
                4608usize,
            )
        };
        let t_527 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(453168usize),
                4608usize,
            )
        };
        let t_528 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(457776usize),
                4608usize,
            )
        };
        let t_529 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(610416usize),
                1536usize,
            )
        };
        let t_533 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(611952usize),
                1536usize,
            )
        };
        let t_534 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668684usize),
                48usize,
            )
        };
        let t_535 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668732usize),
                48usize,
            )
        };
        let t_536 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(668780usize),
                48usize,
            )
        };
        let t_537 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(613488usize),
                1536usize,
            )
        };
        let t_538 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(615024usize),
                1536usize,
            )
        };
        let t_539 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(462384usize),
                4608usize,
            )
        };
        let t_540 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(659576usize),
                576usize,
            )
        };
        let t_541 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(660152usize),
                576usize,
            )
        };
        let t_542 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(660728usize),
                576usize,
            )
        };
        let t_543 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(661304usize),
                576usize,
            )
        };
        let t_544 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(621168usize),
                1024usize,
            )
        };
        let t_545 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(622192usize),
                1024usize,
            )
        };
        let t_547 = &t_247[_frame_idx * 513usize..(_frame_idx + 1) * 513usize];
        let t_548 = &t_221[_frame_idx * 96usize..(_frame_idx + 1) * 96usize];
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_547, t_255);
        op_ticks[48usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[48usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        fully_connected(t_255, 513usize, t_168, None, t_256, 96usize);
        op_ticks[49usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[49usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_256, t_257);
        op_ticks[50usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[50usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_257, t_257, t_258, 96usize);
        op_ticks[51usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[51usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_pow(t_258, t_167, t_259, 1usize);
        op_ticks[52usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[52usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reverse_v2(t_259, &[1usize, 1usize, 96usize], t_260, 2usize);
        op_ticks[53usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[53usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        transpose(
            t_260,
            &[1usize, 1usize, 96usize],
            t_261,
            &[1usize, 96usize, 1usize],
            &[0usize, 2usize, 1usize],
        );
        op_ticks[54usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[54usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_261, t_262);
        op_ticks[55usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[55usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        {
            let src = t_548;
            for p in 0..96usize {
                for a in 0..1usize {
                    let src_off = p * (1usize * 1usize) + a * 1usize;
                    let dst_off = p * (2usize * 1usize) + (0usize + a) * 1usize;
                    t_263[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        {
            let src = t_262;
            for p in 0..96usize {
                for a in 0..1usize {
                    let src_off = p * (1usize * 1usize) + a * 1usize;
                    let dst_off = p * (2usize * 1usize) + (1usize + a) * 1usize;
                    t_263[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        op_ticks[56usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[56usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_263, t_163, t_264, 2usize);
        op_ticks[57usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[57usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_264, t_162, t_265, 2usize);
        op_ticks[58usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[58usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_38_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                3072usize,
            )
        };
        let scratch_38_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672560usize),
                1536usize,
            )
        };
        scratch_38_1.copy_from_slice(t_122);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_265,
            [1usize, 96usize, 1usize, 2usize],
            [4usize, 8usize],
            [2usize, 2usize],
            [1usize, 1usize, 3usize, 4usize],
            [48usize, 1usize],
            scratch_38_0,
        );
        op_ticks[59usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[59usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_38_0, scratch_38_1, t_266, 12usize, 16usize, 6usize);
        op_ticks[60usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[60usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_266, t_51, 48usize, 24usize);
        relu(t_266);
        op_ticks[61usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[61usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        average_pool2d(
            t_266,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 2usize],
            [1usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_267,
            [1usize, 48usize, 1usize, 24usize],
        );
        op_ticks[62usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[62usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        max_pool2d(
            t_266,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 2usize],
            [1usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_268,
            [1usize, 48usize, 1usize, 24usize],
        );
        op_ticks[63usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[63usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        {
            let src = t_268;
            for p in 0..48usize {
                for a in 0..24usize {
                    let src_off = p * (24usize * 1usize) + a * 1usize;
                    let dst_off = p * (48usize * 1usize) + (0usize + a) * 1usize;
                    t_269[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        {
            let src = t_267;
            for p in 0..48usize {
                for a in 0..24usize {
                    let src_off = p * (24usize * 1usize) + a * 1usize;
                    let dst_off = p * (48usize * 1usize) + (24usize + a) * 1usize;
                    t_269[dst_off..dst_off + 1usize]
                        .copy_from_slice(&src[src_off..src_off + 1usize]);
                }
            }
        }
        op_ticks[64usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[64usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_42_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2304usize,
            )
        };
        let scratch_42_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(671792usize),
                1152usize,
            )
        };
        scratch_42_1.copy_from_slice(t_121);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_269,
            [1usize, 48usize, 1usize, 48usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [48usize, 1usize],
            scratch_42_0,
        );
        op_ticks[65usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[65usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_42_0, scratch_42_1, t_270, 12usize, 12usize, 6usize);
        op_ticks[66usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[66usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_270, t_50, 48usize, 24usize);
        op_ticks[67usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[67usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_43_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(671216usize),
                1152usize,
            )
        };
        let scratch_43_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                1728usize,
            )
        };
        scratch_43_1.copy_from_slice(t_120);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_270,
            [1usize, 48usize, 1usize, 24usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [48usize, 1usize],
            scratch_43_0,
        );
        op_ticks[68usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[68usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_43_0, scratch_43_1, t_271, 12usize, 6usize, 18usize);
        op_ticks[69usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[69usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_271, t_49, 48usize, 72usize);
        op_ticks[70usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[70usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_271, t_272);
        op_ticks[71usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[71usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_271, t_272, t_273, 3456usize);
        op_ticks[72usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[72usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        pad(
            t_273,
            [1usize, 48usize, 1usize, 72usize],
            t_274,
            [1usize, 50usize, 3usize, 72usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[73usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[73usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_274,
            [1usize, 50usize, 3usize, 72usize],
            t_48,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_47),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_275,
            [1usize, 24usize, 1usize, 72usize],
        );
        op_ticks[74usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[74usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_275, t_276);
        op_ticks[75usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[75usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_275, t_276, t_277, 1728usize);
        op_ticks[76usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[76usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_50_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_50_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_50_1.copy_from_slice(t_118);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_277,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_50_0,
        );
        op_ticks[77usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[77usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_50_0, scratch_50_1, t_278, 6usize, 18usize, 9usize);
        op_ticks[78usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[78usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_278, t_117, 24usize, 36usize);
        op_ticks[79usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[79usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_51_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                864usize,
            )
        };
        let scratch_51_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_51_1.copy_from_slice(t_116);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_278,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_51_0,
        );
        op_ticks[80usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[80usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_51_0, scratch_51_1, t_279, 6usize, 9usize, 18usize);
        op_ticks[81usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[81usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_279, t_46, 24usize, 72usize);
        op_ticks[82usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[82usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_279, t_280);
        op_ticks[83usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[83usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_279, t_280, t_281, 1728usize);
        op_ticks[84usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[84usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_281,
            [1usize, 24usize, 1usize, 72usize],
            t_45,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_44),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_282,
            [1usize, 24usize, 1usize, 72usize],
        );
        op_ticks[85usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[85usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_282, t_283);
        op_ticks[86usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[86usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_282, t_283, t_284, 1728usize);
        op_ticks[87usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[87usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_57_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_57_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_57_1.copy_from_slice(t_115);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_284,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_57_0,
        );
        op_ticks[88usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[88usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_57_0, scratch_57_1, t_285, 6usize, 18usize, 9usize);
        op_ticks[89usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[89usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_285, t_117, 24usize, 36usize);
        op_ticks[90usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[90usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_285, t_278, t_286, 864usize);
        op_ticks[91usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[91usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_59_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                864usize,
            )
        };
        let scratch_59_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_59_1.copy_from_slice(t_114);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_286,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_59_0,
        );
        op_ticks[92usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[92usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_59_0, scratch_59_1, t_287, 6usize, 9usize, 18usize);
        op_ticks[93usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[93usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_287, t_43, 24usize, 72usize);
        op_ticks[94usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[94usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_287, t_288);
        op_ticks[95usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[95usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_287, t_288, t_289, 1728usize);
        op_ticks[96usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[96usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_289,
            [1usize, 24usize, 1usize, 72usize],
            t_42,
            [1usize, 3usize, 3usize, 72usize],
            Some(t_41),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_290,
            [1usize, 24usize, 1usize, 72usize],
        );
        op_ticks[97usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[97usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_290, t_291);
        op_ticks[98usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[98usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_290, t_291, t_292, 1728usize);
        op_ticks[99usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[99usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_65_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(672080usize),
                1728usize,
            )
        };
        let scratch_65_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                2592usize,
            )
        };
        scratch_65_1.copy_from_slice(t_113);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_292,
            [1usize, 24usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_65_0,
        );
        op_ticks[100usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[100usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_65_0, scratch_65_1, t_293, 6usize, 18usize, 9usize);
        op_ticks[101usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[101usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_293, t_117, 24usize, 36usize);
        op_ticks[102usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[102usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_293, t_286, t_294, 864usize);
        op_ticks[103usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[103usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_67_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(679856usize),
                864usize,
            )
        };
        let scratch_67_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                10368usize,
            )
        };
        scratch_67_1.copy_from_slice(t_112);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_294,
            [1usize, 24usize, 1usize, 36usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [24usize, 1usize],
            scratch_67_0,
        );
        op_ticks[104usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[104usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_67_0, scratch_67_1, t_295, 6usize, 9usize, 72usize);
        op_ticks[105usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[105usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_295, t_40, 24usize, 288usize);
        op_ticks[106usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[106usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_295, t_296);
        op_ticks[107usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[107usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_295, t_296, t_297, 6912usize);
        op_ticks[108usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[108usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        pad(
            t_297,
            [1usize, 24usize, 1usize, 288usize],
            t_298,
            [1usize, 26usize, 3usize, 288usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[109usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[109usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_298,
            [1usize, 26usize, 3usize, 288usize],
            t_39,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_38),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_299,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[110usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[110usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_299, t_300);
        op_ticks[111usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[111usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_299, t_300, t_301, 3456usize);
        op_ticks[112usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[112usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_301, t_302);
        op_ticks[113usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[113usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_302, t_306);
        op_ticks[114usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[114usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_306,
            [1usize, 1usize, 1usize, 288usize],
            t_110,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_307,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[115usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[115usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_307, t_308);
        op_ticks[116usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[116usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_307, t_308, t_309, 18usize);
        op_ticks[117usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[117usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_309,
            [1usize, 1usize, 1usize, 18usize],
            t_108,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_310,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[118usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[118usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_310, t_311);
        op_ticks[119usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[119usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_301, t_311, t_312, 288usize);
        op_ticks[120usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[120usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_82_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_82_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_82_1.copy_from_slice(t_107);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_312,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_82_0,
        );
        op_ticks[121usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[121usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_82_0, scratch_82_1, t_313, 3usize, 72usize, 18usize);
        op_ticks[122usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[122usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_313, t_119, 12usize, 72usize);
        op_ticks[123usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[123usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_83_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_83_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_83_1.copy_from_slice(t_106);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_313,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_83_0,
        );
        op_ticks[124usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[124usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_83_0, scratch_83_1, t_314, 3usize, 18usize, 72usize);
        op_ticks[125usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[125usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_314, t_37, 12usize, 288usize);
        op_ticks[126usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[126usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_314, t_315);
        op_ticks[127usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[127usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_314, t_315, t_316, 3456usize);
        op_ticks[128usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[128usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_316,
            [1usize, 12usize, 1usize, 288usize],
            t_36,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_35),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_317,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[129usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[129usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_317, t_318);
        op_ticks[130usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[130usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_317, t_318, t_319, 3456usize);
        op_ticks[131usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[131usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_319, t_320);
        op_ticks[132usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[132usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_320, t_324);
        op_ticks[133usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[133usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_324,
            [1usize, 1usize, 1usize, 288usize],
            t_105,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_325,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[134usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[134usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_325, t_326);
        op_ticks[135usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[135usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_325, t_326, t_327, 18usize);
        op_ticks[136usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[136usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_327,
            [1usize, 1usize, 1usize, 18usize],
            t_104,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_328,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[137usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[137usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_328, t_329);
        op_ticks[138usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[138usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_319, t_329, t_330, 288usize);
        op_ticks[139usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[139usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_97_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_97_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_97_1.copy_from_slice(t_103);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_330,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_97_0,
        );
        op_ticks[140usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[140usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_97_0, scratch_97_1, t_331, 3usize, 72usize, 18usize);
        op_ticks[141usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[141usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_331, t_119, 12usize, 72usize);
        op_ticks[142usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[142usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_331, t_313, t_332, 864usize);
        op_ticks[143usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[143usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_99_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_99_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_99_1.copy_from_slice(t_102);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_332,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_99_0,
        );
        op_ticks[144usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[144usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_99_0, scratch_99_1, t_333, 3usize, 18usize, 72usize);
        op_ticks[145usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[145usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_333, t_34, 12usize, 288usize);
        op_ticks[146usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[146usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_333, t_334);
        op_ticks[147usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[147usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_333, t_334, t_335, 3456usize);
        op_ticks[148usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[148usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_335,
            [1usize, 12usize, 1usize, 288usize],
            t_33,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_32),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_336,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[149usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[149usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_336, t_337);
        op_ticks[150usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[150usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_336, t_337, t_338, 3456usize);
        op_ticks[151usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[151usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_338, t_339);
        op_ticks[152usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[152usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_339, t_343);
        op_ticks[153usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[153usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_343,
            [1usize, 1usize, 1usize, 288usize],
            t_101,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_344,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[154usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[154usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_344, t_345);
        op_ticks[155usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[155usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_344, t_345, t_346, 18usize);
        op_ticks[156usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[156usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_346,
            [1usize, 1usize, 1usize, 18usize],
            t_100,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_347,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[157usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[157usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_347, t_348);
        op_ticks[158usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[158usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_338, t_348, t_349, 288usize);
        op_ticks[159usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[159usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_113_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_113_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_113_1.copy_from_slice(t_99);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_349,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_113_0,
        );
        op_ticks[160usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[160usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_113_0, scratch_113_1, t_350, 3usize, 72usize, 18usize);
        op_ticks[161usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[161usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_350, t_119, 12usize, 72usize);
        op_ticks[162usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[162usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_350, t_332, t_351, 864usize);
        op_ticks[163usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[163usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_115_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                864usize,
            )
        };
        let scratch_115_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_115_1.copy_from_slice(t_98);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_351,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_115_0,
        );
        op_ticks[164usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[164usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_115_0, scratch_115_1, t_352, 3usize, 18usize, 72usize);
        op_ticks[165usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[165usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_352, t_31, 12usize, 288usize);
        op_ticks[166usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[166usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_352, t_353);
        op_ticks[167usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[167usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_352, t_353, t_354, 3456usize);
        op_ticks[168usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[168usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_354,
            [1usize, 12usize, 1usize, 288usize],
            t_30,
            [1usize, 3usize, 3usize, 288usize],
            Some(t_29),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_355,
            [1usize, 12usize, 1usize, 288usize],
        );
        op_ticks[169usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[169usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_355, t_356);
        op_ticks[170usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[170usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_355, t_356, t_357, 3456usize);
        op_ticks[171usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[171usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_357, t_358);
        op_ticks[172usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[172usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_358, t_362);
        op_ticks[173usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[173usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_362,
            [1usize, 1usize, 1usize, 288usize],
            t_97,
            [18usize, 1usize, 1usize, 288usize],
            Some(t_109),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_363,
            [1usize, 1usize, 1usize, 18usize],
        );
        op_ticks[174usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[174usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_363, t_364);
        op_ticks[175usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[175usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_363, t_364, t_365, 18usize);
        op_ticks[176usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[176usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_365,
            [1usize, 1usize, 1usize, 18usize],
            t_96,
            [288usize, 1usize, 1usize, 18usize],
            Some(t_111),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_366,
            [1usize, 1usize, 1usize, 288usize],
        );
        op_ticks[177usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[177usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_366, t_367);
        op_ticks[178usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[178usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_357, t_367, t_368, 288usize);
        op_ticks[179usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[179usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_129_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(690224usize),
                3456usize,
            )
        };
        let scratch_129_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                20736usize,
            )
        };
        scratch_129_1.copy_from_slice(t_95);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_368,
            [1usize, 12usize, 1usize, 288usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_129_0,
        );
        op_ticks[180usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[180usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_129_0, scratch_129_1, t_369, 3usize, 72usize, 18usize);
        op_ticks[181usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[181usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_369, t_119, 12usize, 72usize);
        op_ticks[182usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[182usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_369, t_351, t_370, 864usize);
        op_ticks[183usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[183usize].accumulate(__regs.assume_init_ref());
        }
        let scratch_131_0 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(731696usize),
                864usize,
            )
        };
        let scratch_131_1 = unsafe {
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!(ARENA) as *mut f32).add(669488usize),
                62208usize,
            )
        };
        scratch_131_1.copy_from_slice(t_94);
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        im2col_padded(
            t_370,
            [1usize, 12usize, 1usize, 72usize],
            [1usize, 1usize],
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            [12usize, 1usize],
            scratch_131_0,
        );
        op_ticks[184usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[184usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        matmul_bt_tiled(scratch_131_0, scratch_131_1, t_371, 3usize, 18usize, 216usize);
        op_ticks[185usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[185usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        bias_add(t_371, t_28, 12usize, 864usize);
        op_ticks[186usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[186usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_371, t_372);
        op_ticks[187usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[187usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_371, t_372, t_373, 10368usize);
        op_ticks[188usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[188usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        pad(
            t_373,
            [1usize, 12usize, 1usize, 864usize],
            t_374,
            [1usize, 14usize, 3usize, 864usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[189usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[189usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_374,
            [1usize, 14usize, 3usize, 864usize],
            t_27,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_26),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_375,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[190usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[190usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_375, t_376);
        op_ticks[191usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[191usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_375, t_376, t_377, 5184usize);
        op_ticks[192usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[192usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_377, t_378);
        op_ticks[193usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[193usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_378, t_382);
        op_ticks[194usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[194usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_382,
            [1usize, 1usize, 1usize, 864usize],
            t_92,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_383,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[195usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[195usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_383, t_384);
        op_ticks[196usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[196usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_383, t_384, t_385, 27usize);
        op_ticks[197usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[197usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_385,
            [1usize, 1usize, 1usize, 27usize],
            t_90,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_386,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[198usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[198usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_386, t_387);
        op_ticks[199usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[199usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_377, t_387, t_388, 864usize);
        op_ticks[200usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[200usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_388,
            [1usize, 6usize, 1usize, 864usize],
            t_89,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_389,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[201usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[201usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_389,
            [1usize, 6usize, 1usize, 108usize],
            t_87,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_25),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_390,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[202usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[202usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_390, t_391);
        op_ticks[203usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[203usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_390, t_391, t_392, 5184usize);
        op_ticks[204usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[204usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_392,
            [1usize, 6usize, 1usize, 864usize],
            t_24,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_23),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_393,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[205usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[205usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_393, t_394);
        op_ticks[206usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[206usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_393, t_394, t_395, 5184usize);
        op_ticks[207usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[207usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_395, t_396);
        op_ticks[208usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[208usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_396, t_400);
        op_ticks[209usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[209usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_400,
            [1usize, 1usize, 1usize, 864usize],
            t_86,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_401,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[210usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[210usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_401, t_402);
        op_ticks[211usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[211usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_401, t_402, t_403, 27usize);
        op_ticks[212usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[212usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_403,
            [1usize, 1usize, 1usize, 27usize],
            t_85,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_404,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[213usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[213usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_404, t_405);
        op_ticks[214usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[214usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_395, t_405, t_406, 864usize);
        op_ticks[215usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[215usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_406,
            [1usize, 6usize, 1usize, 864usize],
            t_84,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_407,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[216usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[216usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_407, t_389, t_408, 648usize);
        op_ticks[217usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[217usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_408,
            [1usize, 6usize, 1usize, 108usize],
            t_83,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_22),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_409,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[218usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[218usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_409, t_410);
        op_ticks[219usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[219usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_409, t_410, t_411, 5184usize);
        op_ticks[220usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[220usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_411,
            [1usize, 6usize, 1usize, 864usize],
            t_21,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_20),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_412,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[221usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[221usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_412, t_413);
        op_ticks[222usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[222usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_412, t_413, t_414, 5184usize);
        op_ticks[223usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[223usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_414, t_415);
        op_ticks[224usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[224usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_415, t_419);
        op_ticks[225usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[225usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_419,
            [1usize, 1usize, 1usize, 864usize],
            t_82,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_420,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[226usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[226usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_420, t_421);
        op_ticks[227usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[227usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_420, t_421, t_422, 27usize);
        op_ticks[228usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[228usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_422,
            [1usize, 1usize, 1usize, 27usize],
            t_81,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_423,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[229usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[229usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_423, t_424);
        op_ticks[230usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[230usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_414, t_424, t_425, 864usize);
        op_ticks[231usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[231usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_425,
            [1usize, 6usize, 1usize, 864usize],
            t_80,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_426,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[232usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[232usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_426, t_408, t_427, 648usize);
        op_ticks[233usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[233usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_427,
            [1usize, 6usize, 1usize, 108usize],
            t_79,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_19),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_428,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[234usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[234usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_428, t_429);
        op_ticks[235usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[235usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_428, t_429, t_430, 5184usize);
        op_ticks[236usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[236usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_430,
            [1usize, 6usize, 1usize, 864usize],
            t_18,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_17),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_431,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[237usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[237usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_431, t_432);
        op_ticks[238usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[238usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_431, t_432, t_433, 5184usize);
        op_ticks[239usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[239usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_433, t_434);
        op_ticks[240usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[240usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_434, t_438);
        op_ticks[241usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[241usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_438,
            [1usize, 1usize, 1usize, 864usize],
            t_78,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_439,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[242usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[242usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_439, t_440);
        op_ticks[243usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[243usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_439, t_440, t_441, 27usize);
        op_ticks[244usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[244usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_441,
            [1usize, 1usize, 1usize, 27usize],
            t_77,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_442,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[245usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[245usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_442, t_443);
        op_ticks[246usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[246usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_433, t_443, t_444, 864usize);
        op_ticks[247usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[247usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_444,
            [1usize, 6usize, 1usize, 864usize],
            t_76,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_445,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[248usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[248usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_445, t_427, t_446, 648usize);
        op_ticks[249usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[249usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_446,
            [1usize, 6usize, 1usize, 108usize],
            t_75,
            [864usize, 1usize, 1usize, 108usize],
            Some(t_16),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_447,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[250usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[250usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_447, t_448);
        op_ticks[251usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[251usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_447, t_448, t_449, 5184usize);
        op_ticks[252usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[252usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_449,
            [1usize, 6usize, 1usize, 864usize],
            t_15,
            [1usize, 3usize, 3usize, 864usize],
            Some(t_14),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_450,
            [1usize, 6usize, 1usize, 864usize],
        );
        op_ticks[253usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[253usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_450, t_451);
        op_ticks[254usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[254usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_450, t_451, t_452, 5184usize);
        op_ticks[255usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[255usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_452, t_453);
        op_ticks[256usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[256usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_453, t_457);
        op_ticks[257usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[257usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_457,
            [1usize, 1usize, 1usize, 864usize],
            t_74,
            [27usize, 1usize, 1usize, 864usize],
            Some(t_91),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_458,
            [1usize, 1usize, 1usize, 27usize],
        );
        op_ticks[258usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[258usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_458, t_459);
        op_ticks[259usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[259usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_458, t_459, t_460, 27usize);
        op_ticks[260usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[260usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_460,
            [1usize, 1usize, 1usize, 27usize],
            t_73,
            [864usize, 1usize, 1usize, 27usize],
            Some(t_93),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_461,
            [1usize, 1usize, 1usize, 864usize],
        );
        op_ticks[261usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[261usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_461, t_462);
        op_ticks[262usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[262usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_452, t_462, t_463, 864usize);
        op_ticks[263usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[263usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_463,
            [1usize, 6usize, 1usize, 864usize],
            t_72,
            [108usize, 1usize, 1usize, 864usize],
            Some(t_88),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_464,
            [1usize, 6usize, 1usize, 108usize],
        );
        op_ticks[264usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[264usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_464, t_446, t_465, 648usize);
        op_ticks[265usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[265usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_465,
            [1usize, 6usize, 1usize, 108usize],
            t_71,
            [1536usize, 1usize, 1usize, 108usize],
            Some(t_13),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_466,
            [1usize, 6usize, 1usize, 1536usize],
        );
        op_ticks[266usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[266usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_466, t_467);
        op_ticks[267usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[267usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_466, t_467, t_468, 9216usize);
        op_ticks[268usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[268usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        pad(
            t_468,
            [1usize, 6usize, 1usize, 1536usize],
            t_469,
            [1usize, 8usize, 3usize, 1536usize],
            [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
        );
        op_ticks[269usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[269usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_469,
            [1usize, 8usize, 3usize, 1536usize],
            t_12,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_11),
            [2usize, 2usize],
            [0usize, 0usize, 0usize, 0usize],
            t_470,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[270usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[270usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_470, t_471);
        op_ticks[271usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[271usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_470, t_471, t_472, 4608usize);
        op_ticks[272usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[272usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_472, t_473);
        op_ticks[273usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[273usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_473, t_477);
        op_ticks[274usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[274usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_477,
            [1usize, 1usize, 1usize, 1536usize],
            t_69,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_478,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[275usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[275usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_478, t_479);
        op_ticks[276usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[276usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_478, t_479, t_480, 48usize);
        op_ticks[277usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[277usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_480,
            [1usize, 1usize, 1usize, 48usize],
            t_67,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_481,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[278usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[278usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_481, t_482);
        op_ticks[279usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[279usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_472, t_482, t_483, 1536usize);
        op_ticks[280usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[280usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_483,
            [1usize, 3usize, 1usize, 1536usize],
            t_66,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_484,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[281usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[281usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_484,
            [1usize, 3usize, 1usize, 192usize],
            t_64,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_10),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_485,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[282usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[282usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_485, t_486);
        op_ticks[283usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[283usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_485, t_486, t_487, 4608usize);
        op_ticks[284usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[284usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_487,
            [1usize, 3usize, 1usize, 1536usize],
            t_9,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_8),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_488,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[285usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[285usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_488, t_489);
        op_ticks[286usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[286usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_488, t_489, t_490, 4608usize);
        op_ticks[287usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[287usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_490, t_491);
        op_ticks[288usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[288usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_491, t_495);
        op_ticks[289usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[289usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_495,
            [1usize, 1usize, 1usize, 1536usize],
            t_63,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_496,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[290usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[290usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_496, t_497);
        op_ticks[291usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[291usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_496, t_497, t_498, 48usize);
        op_ticks[292usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[292usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_498,
            [1usize, 1usize, 1usize, 48usize],
            t_62,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_499,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[293usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[293usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_499, t_500);
        op_ticks[294usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[294usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_490, t_500, t_501, 1536usize);
        op_ticks[295usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[295usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_501,
            [1usize, 3usize, 1usize, 1536usize],
            t_61,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_502,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[296usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[296usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_502, t_484, t_503, 576usize);
        op_ticks[297usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[297usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_503,
            [1usize, 3usize, 1usize, 192usize],
            t_60,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_7),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_504,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[298usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[298usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_504, t_505);
        op_ticks[299usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[299usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_504, t_505, t_506, 4608usize);
        op_ticks[300usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[300usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_506,
            [1usize, 3usize, 1usize, 1536usize],
            t_6,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_5),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_507,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[301usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[301usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_507, t_508);
        op_ticks[302usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[302usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_507, t_508, t_509, 4608usize);
        op_ticks[303usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[303usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_509, t_510);
        op_ticks[304usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[304usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_510, t_514);
        op_ticks[305usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[305usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_514,
            [1usize, 1usize, 1usize, 1536usize],
            t_59,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_515,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[306usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[306usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_515, t_516);
        op_ticks[307usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[307usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_515, t_516, t_517, 48usize);
        op_ticks[308usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[308usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_517,
            [1usize, 1usize, 1usize, 48usize],
            t_58,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_518,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[309usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[309usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_518, t_519);
        op_ticks[310usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[310usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_509, t_519, t_520, 1536usize);
        op_ticks[311usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[311usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_520,
            [1usize, 3usize, 1usize, 1536usize],
            t_57,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_521,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[312usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[312usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_521, t_503, t_522, 576usize);
        op_ticks[313usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[313usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_522,
            [1usize, 3usize, 1usize, 192usize],
            t_56,
            [1536usize, 1usize, 1usize, 192usize],
            Some(t_4),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_523,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[314usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[314usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_523, t_524);
        op_ticks[315usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[315usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_523, t_524, t_525, 4608usize);
        op_ticks[316usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[316usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        depthwise_conv2d(
            t_525,
            [1usize, 3usize, 1usize, 1536usize],
            t_3,
            [1usize, 3usize, 3usize, 1536usize],
            Some(t_2),
            [1usize, 1usize],
            [1usize, 1usize, 1usize, 1usize],
            t_526,
            [1usize, 3usize, 1usize, 1536usize],
        );
        op_ticks[317usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[317usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_526, t_527);
        op_ticks[318usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[318usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_526, t_527, t_528, 4608usize);
        op_ticks[319usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[319usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_528, t_529);
        op_ticks[320usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[320usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reshape(t_529, t_533);
        op_ticks[321usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[321usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_533,
            [1usize, 1usize, 1usize, 1536usize],
            t_55,
            [48usize, 1usize, 1usize, 1536usize],
            Some(t_68),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_534,
            [1usize, 1usize, 1usize, 48usize],
        );
        op_ticks[322usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[322usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_534, t_535);
        op_ticks[323usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[323usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_534, t_535, t_536, 48usize);
        op_ticks[324usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[324usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_536,
            [1usize, 1usize, 1usize, 48usize],
            t_54,
            [1536usize, 1usize, 1usize, 48usize],
            Some(t_70),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_537,
            [1usize, 1usize, 1usize, 1536usize],
        );
        op_ticks[325usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[325usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        unary_logistic(t_537, t_538);
        op_ticks[326usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[326usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_528, t_538, t_539, 1536usize);
        op_ticks[327usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[327usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d(
            t_539,
            [1usize, 3usize, 1usize, 1536usize],
            t_53,
            [192usize, 1usize, 1usize, 1536usize],
            Some(t_65),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_540,
            [1usize, 3usize, 1usize, 192usize],
        );
        op_ticks[328usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[328usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_540, t_522, t_541, 576usize);
        op_ticks[329usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[329usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_mul(t_541, t_161, t_542, 192usize);
        op_ticks[330usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[330usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        binary_add(t_542, t_160, t_543, 192usize);
        op_ticks[331usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[331usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        conv2d_relu(
            t_543,
            [1usize, 3usize, 1usize, 192usize],
            t_52,
            [1024usize, 3usize, 3usize, 192usize],
            Some(t_1),
            [1usize, 1usize],
            [0usize, 0usize, 0usize, 0usize],
            t_544,
            [1usize, 1usize, 1usize, 1024usize],
        );
        op_ticks[332usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[332usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        reduce_mean_hw(t_544, t_545);
        op_ticks[333usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[333usize].accumulate(__regs.assume_init_ref());
        }
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileClear();
            psp_ml::profiler::ProfileEnable();
        }
        let __t0 = get_tick();
        fully_connected(t_545, 1024usize, t_164, Some(t_142), t_546, 6522usize);
        op_ticks[334usize] += get_tick() - __t0;
        #[cfg(target_os = "psp")]
        unsafe {
            psp_ml::profiler::ProfileDisable();
            let mut __regs = core::mem::MaybeUninit::<
                psp_ml::profiler::ProfileRegs,
            >::zeroed();
            psp_ml::profiler::ProfileGetRegs(__regs.as_mut_ptr());
            op_profile[334usize].accumulate(__regs.assume_init_ref());
        }
    }
    output.copy_from_slice(&t_546);
}
pub const NUM_OPS: usize = 335usize;
pub const OP_NAMES: [&str; NUM_OPS] = [
    "reduce_min",
    "binary_sub",
    "reduce_max",
    "binary_add",
    "binary_div",
    "binary_sub",
    "binary_mul",
    "strided_slice",
    "reshape",
    "gather",
    "reshape",
    "binary_mul",
    "rfft_pack",
    "fft_butterfly_s0",
    "fft_butterfly_s1",
    "fft_butterfly_s2",
    "fft_butterfly_s3",
    "fft_butterfly_s4",
    "fft_butterfly_s5",
    "fft_butterfly_s6",
    "fft_butterfly_s7",
    "fft_butterfly_s8",
    "fft_butterfly_s9",
    "rfft_unpack",
    "reshape",
    "fully_connected",
    "reshape",
    "binary_mul",
    "binary_pow",
    "reverse_v2",
    "transpose",
    "reshape",
    "strided_slice",
    "reshape",
    "gather",
    "reshape",
    "binary_mul",
    "rfft_pack",
    "fft_butterfly_s0",
    "fft_butterfly_s1",
    "fft_butterfly_s2",
    "fft_butterfly_s3",
    "fft_butterfly_s4",
    "fft_butterfly_s5",
    "fft_butterfly_s6",
    "fft_butterfly_s7",
    "fft_butterfly_s8",
    "rfft_unpack",
    "reshape",
    "fully_connected",
    "reshape",
    "binary_mul",
    "binary_pow",
    "reverse_v2",
    "transpose",
    "reshape",
    "concatenation",
    "binary_mul",
    "binary_add",
    "im2col",
    "matmul",
    "bias_add_relu",
    "average_pool2d",
    "max_pool2d",
    "concatenation",
    "im2col",
    "matmul",
    "bias_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "pad",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "im2col",
    "matmul",
    "bias_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "im2col",
    "matmul",
    "bias_add",
    "binary_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "im2col",
    "matmul",
    "bias_add",
    "binary_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "pad",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "im2col",
    "matmul",
    "bias_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "im2col",
    "matmul",
    "bias_add",
    "binary_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "im2col",
    "matmul",
    "bias_add",
    "binary_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "im2col",
    "matmul",
    "bias_add",
    "binary_add",
    "im2col",
    "matmul",
    "bias_add",
    "logistic",
    "binary_mul",
    "pad",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "conv2d",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "binary_add",
    "conv2d",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "binary_add",
    "conv2d",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "binary_add",
    "conv2d",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "binary_add",
    "conv2d",
    "logistic",
    "binary_mul",
    "pad",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "conv2d",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "binary_add",
    "conv2d",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "binary_add",
    "conv2d",
    "logistic",
    "binary_mul",
    "depthwise_conv2d",
    "logistic",
    "binary_mul",
    "reduce_mean_hw",
    "reshape",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "logistic",
    "binary_mul",
    "conv2d",
    "binary_add",
    "binary_mul",
    "binary_add",
    "conv2d_relu",
    "reduce_mean_hw",
    "fully_connected",
];
#[allow(dead_code)]
#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);
/// 16-byte aligned f32 array for VFPU `lv.q`/`sv.q`.
#[repr(C, align(16))]
struct Aligned16<const N: usize>([f32; N]);
const WEIGHT_BYTES: usize = 51670200usize;
const TENSOR_DATA_FLOATS: usize = 12917550usize;
static mut WEIGHT_PTR: *const u8 = core::ptr::null();
const T_1_OFFSET: usize = 0usize;
const T_1_LEN: usize = 1024usize;
const T_2_OFFSET: usize = 1024usize;
const T_2_LEN: usize = 1536usize;
const T_3_OFFSET: usize = 2560usize;
const T_3_LEN: usize = 13824usize;
const T_4_OFFSET: usize = 16384usize;
const T_4_LEN: usize = 1536usize;
const T_5_OFFSET: usize = 17920usize;
const T_5_LEN: usize = 1536usize;
const T_6_OFFSET: usize = 19456usize;
const T_6_LEN: usize = 13824usize;
const T_7_OFFSET: usize = 33280usize;
const T_7_LEN: usize = 1536usize;
const T_8_OFFSET: usize = 34816usize;
const T_8_LEN: usize = 1536usize;
const T_9_OFFSET: usize = 36352usize;
const T_9_LEN: usize = 13824usize;
const T_10_OFFSET: usize = 50176usize;
const T_10_LEN: usize = 1536usize;
const T_11_OFFSET: usize = 51712usize;
const T_11_LEN: usize = 1536usize;
const T_12_OFFSET: usize = 53248usize;
const T_12_LEN: usize = 13824usize;
const T_13_OFFSET: usize = 67072usize;
const T_13_LEN: usize = 1536usize;
const T_14_OFFSET: usize = 68608usize;
const T_14_LEN: usize = 864usize;
const T_15_OFFSET: usize = 69472usize;
const T_15_LEN: usize = 7776usize;
const T_16_OFFSET: usize = 77248usize;
const T_16_LEN: usize = 864usize;
const T_17_OFFSET: usize = 78112usize;
const T_17_LEN: usize = 864usize;
const T_18_OFFSET: usize = 78976usize;
const T_18_LEN: usize = 7776usize;
const T_19_OFFSET: usize = 86752usize;
const T_19_LEN: usize = 864usize;
const T_20_OFFSET: usize = 87616usize;
const T_20_LEN: usize = 864usize;
const T_21_OFFSET: usize = 88480usize;
const T_21_LEN: usize = 7776usize;
const T_22_OFFSET: usize = 96256usize;
const T_22_LEN: usize = 864usize;
const T_23_OFFSET: usize = 97120usize;
const T_23_LEN: usize = 864usize;
const T_24_OFFSET: usize = 97984usize;
const T_24_LEN: usize = 7776usize;
const T_25_OFFSET: usize = 105760usize;
const T_25_LEN: usize = 864usize;
const T_26_OFFSET: usize = 106624usize;
const T_26_LEN: usize = 864usize;
const T_27_OFFSET: usize = 107488usize;
const T_27_LEN: usize = 7776usize;
const T_28_OFFSET: usize = 115264usize;
const T_28_LEN: usize = 864usize;
const T_29_OFFSET: usize = 116128usize;
const T_29_LEN: usize = 288usize;
const T_30_OFFSET: usize = 116416usize;
const T_30_LEN: usize = 2592usize;
const T_31_OFFSET: usize = 119008usize;
const T_31_LEN: usize = 288usize;
const T_32_OFFSET: usize = 119296usize;
const T_32_LEN: usize = 288usize;
const T_33_OFFSET: usize = 119584usize;
const T_33_LEN: usize = 2592usize;
const T_34_OFFSET: usize = 122176usize;
const T_34_LEN: usize = 288usize;
const T_35_OFFSET: usize = 122464usize;
const T_35_LEN: usize = 288usize;
const T_36_OFFSET: usize = 122752usize;
const T_36_LEN: usize = 2592usize;
const T_37_OFFSET: usize = 125344usize;
const T_37_LEN: usize = 288usize;
const T_38_OFFSET: usize = 125632usize;
const T_38_LEN: usize = 288usize;
const T_39_OFFSET: usize = 125920usize;
const T_39_LEN: usize = 2592usize;
const T_40_OFFSET: usize = 128512usize;
const T_40_LEN: usize = 288usize;
const T_41_OFFSET: usize = 128800usize;
const T_41_LEN: usize = 72usize;
const T_42_OFFSET: usize = 128872usize;
const T_42_LEN: usize = 648usize;
const T_43_OFFSET: usize = 129520usize;
const T_43_LEN: usize = 72usize;
const T_44_OFFSET: usize = 129592usize;
const T_44_LEN: usize = 72usize;
const T_45_OFFSET: usize = 129664usize;
const T_45_LEN: usize = 648usize;
const T_46_OFFSET: usize = 130312usize;
const T_46_LEN: usize = 72usize;
const T_47_OFFSET: usize = 130384usize;
const T_47_LEN: usize = 72usize;
const T_48_OFFSET: usize = 130456usize;
const T_48_LEN: usize = 648usize;
const T_49_OFFSET: usize = 131104usize;
const T_49_LEN: usize = 72usize;
const T_50_OFFSET: usize = 131176usize;
const T_50_LEN: usize = 24usize;
const T_51_OFFSET: usize = 131200usize;
const T_51_LEN: usize = 24usize;
const T_52_OFFSET: usize = 131224usize;
const T_52_LEN: usize = 1769472usize;
const T_53_OFFSET: usize = 1900696usize;
const T_53_LEN: usize = 294912usize;
const T_54_OFFSET: usize = 2195608usize;
const T_54_LEN: usize = 73728usize;
const T_55_OFFSET: usize = 2269336usize;
const T_55_LEN: usize = 73728usize;
const T_56_OFFSET: usize = 2343064usize;
const T_56_LEN: usize = 294912usize;
const T_57_OFFSET: usize = 2637976usize;
const T_57_LEN: usize = 294912usize;
const T_58_OFFSET: usize = 2932888usize;
const T_58_LEN: usize = 73728usize;
const T_59_OFFSET: usize = 3006616usize;
const T_59_LEN: usize = 73728usize;
const T_60_OFFSET: usize = 3080344usize;
const T_60_LEN: usize = 294912usize;
const T_61_OFFSET: usize = 3375256usize;
const T_61_LEN: usize = 294912usize;
const T_62_OFFSET: usize = 3670168usize;
const T_62_LEN: usize = 73728usize;
const T_63_OFFSET: usize = 3743896usize;
const T_63_LEN: usize = 73728usize;
const T_64_OFFSET: usize = 3817624usize;
const T_64_LEN: usize = 294912usize;
const T_65_OFFSET: usize = 4112536usize;
const T_65_LEN: usize = 192usize;
const T_66_OFFSET: usize = 4112728usize;
const T_66_LEN: usize = 294912usize;
const T_67_OFFSET: usize = 4407640usize;
const T_67_LEN: usize = 73728usize;
const T_68_OFFSET: usize = 4481368usize;
const T_68_LEN: usize = 48usize;
const T_69_OFFSET: usize = 4481416usize;
const T_69_LEN: usize = 73728usize;
const T_70_OFFSET: usize = 4555144usize;
const T_70_LEN: usize = 1536usize;
const T_71_OFFSET: usize = 4556680usize;
const T_71_LEN: usize = 165888usize;
const T_72_OFFSET: usize = 4722568usize;
const T_72_LEN: usize = 93312usize;
const T_73_OFFSET: usize = 4815880usize;
const T_73_LEN: usize = 23328usize;
const T_74_OFFSET: usize = 4839208usize;
const T_74_LEN: usize = 23328usize;
const T_75_OFFSET: usize = 4862536usize;
const T_75_LEN: usize = 93312usize;
const T_76_OFFSET: usize = 4955848usize;
const T_76_LEN: usize = 93312usize;
const T_77_OFFSET: usize = 5049160usize;
const T_77_LEN: usize = 23328usize;
const T_78_OFFSET: usize = 5072488usize;
const T_78_LEN: usize = 23328usize;
const T_79_OFFSET: usize = 5095816usize;
const T_79_LEN: usize = 93312usize;
const T_80_OFFSET: usize = 5189128usize;
const T_80_LEN: usize = 93312usize;
const T_81_OFFSET: usize = 5282440usize;
const T_81_LEN: usize = 23328usize;
const T_82_OFFSET: usize = 5305768usize;
const T_82_LEN: usize = 23328usize;
const T_83_OFFSET: usize = 5329096usize;
const T_83_LEN: usize = 93312usize;
const T_84_OFFSET: usize = 5422408usize;
const T_84_LEN: usize = 93312usize;
const T_85_OFFSET: usize = 5515720usize;
const T_85_LEN: usize = 23328usize;
const T_86_OFFSET: usize = 5539048usize;
const T_86_LEN: usize = 23328usize;
const T_87_OFFSET: usize = 5562376usize;
const T_87_LEN: usize = 93312usize;
const T_88_OFFSET: usize = 5655688usize;
const T_88_LEN: usize = 108usize;
const T_89_OFFSET: usize = 5655796usize;
const T_89_LEN: usize = 93312usize;
const T_90_OFFSET: usize = 5749108usize;
const T_90_LEN: usize = 23328usize;
const T_91_OFFSET: usize = 5772436usize;
const T_91_LEN: usize = 27usize;
const T_92_OFFSET: usize = 5772464usize;
const T_92_LEN: usize = 23328usize;
const T_93_OFFSET: usize = 5795792usize;
const T_93_LEN: usize = 864usize;
const T_94_OFFSET: usize = 5796656usize;
const T_94_LEN: usize = 62208usize;
const T_95_OFFSET: usize = 5858864usize;
const T_95_LEN: usize = 20736usize;
const T_96_OFFSET: usize = 5879600usize;
const T_96_LEN: usize = 5184usize;
const T_97_OFFSET: usize = 5884784usize;
const T_97_LEN: usize = 5184usize;
const T_98_OFFSET: usize = 5889968usize;
const T_98_LEN: usize = 20736usize;
const T_99_OFFSET: usize = 5910704usize;
const T_99_LEN: usize = 20736usize;
const T_100_OFFSET: usize = 5931440usize;
const T_100_LEN: usize = 5184usize;
const T_101_OFFSET: usize = 5936624usize;
const T_101_LEN: usize = 5184usize;
const T_102_OFFSET: usize = 5941808usize;
const T_102_LEN: usize = 20736usize;
const T_103_OFFSET: usize = 5962544usize;
const T_103_LEN: usize = 20736usize;
const T_104_OFFSET: usize = 5983280usize;
const T_104_LEN: usize = 5184usize;
const T_105_OFFSET: usize = 5988464usize;
const T_105_LEN: usize = 5184usize;
const T_106_OFFSET: usize = 5993648usize;
const T_106_LEN: usize = 20736usize;
const T_107_OFFSET: usize = 6014384usize;
const T_107_LEN: usize = 20736usize;
const T_108_OFFSET: usize = 6035120usize;
const T_108_LEN: usize = 5184usize;
const T_109_OFFSET: usize = 6040304usize;
const T_109_LEN: usize = 18usize;
const T_110_OFFSET: usize = 6040324usize;
const T_110_LEN: usize = 5184usize;
const T_111_OFFSET: usize = 6045508usize;
const T_111_LEN: usize = 288usize;
const T_112_OFFSET: usize = 6045796usize;
const T_112_LEN: usize = 10368usize;
const T_113_OFFSET: usize = 6056164usize;
const T_113_LEN: usize = 2592usize;
const T_114_OFFSET: usize = 6058756usize;
const T_114_LEN: usize = 2592usize;
const T_115_OFFSET: usize = 6061348usize;
const T_115_LEN: usize = 2592usize;
const T_116_OFFSET: usize = 6063940usize;
const T_116_LEN: usize = 2592usize;
const T_117_OFFSET: usize = 6066532usize;
const T_117_LEN: usize = 36usize;
const T_118_OFFSET: usize = 6066568usize;
const T_118_LEN: usize = 2592usize;
const T_119_OFFSET: usize = 6069160usize;
const T_119_LEN: usize = 72usize;
const T_120_OFFSET: usize = 6069232usize;
const T_120_LEN: usize = 1728usize;
const T_121_OFFSET: usize = 6070960usize;
const T_121_LEN: usize = 1152usize;
const T_122_OFFSET: usize = 6072112usize;
const T_122_LEN: usize = 1536usize;
const T_126_OFFSET: usize = 6073648usize;
const T_126_LEN: usize = 1024usize;
const T_129_OFFSET: usize = 6074672usize;
const T_129_LEN: usize = 2048usize;
const T_133_OFFSET: usize = 6076720usize;
const T_133_LEN: usize = 1usize;
const T_134_OFFSET: usize = 6076724usize;
const T_134_LEN: usize = 2usize;
const T_135_OFFSET: usize = 6076728usize;
const T_135_LEN: usize = 8usize;
const T_136_OFFSET: usize = 6076736usize;
const T_136_LEN: usize = 2usize;
const T_142_OFFSET: usize = 6076740usize;
const T_142_LEN: usize = 6522usize;
const T_144_OFFSET: usize = 6083264usize;
const T_144_LEN: usize = 1usize;
const T_145_OFFSET: usize = 6083268usize;
const T_145_LEN: usize = 1usize;
const T_146_OFFSET: usize = 6083272usize;
const T_146_LEN: usize = 1usize;
const T_150_OFFSET: usize = 6083276usize;
const T_150_LEN: usize = 1usize;
const T_153_OFFSET: usize = 6083280usize;
const T_153_LEN: usize = 2usize;
const T_156_OFFSET: usize = 6083284usize;
const T_156_LEN: usize = 3usize;
const T_160_OFFSET: usize = 6083288usize;
const T_160_LEN: usize = 192usize;
const T_161_OFFSET: usize = 6083480usize;
const T_161_LEN: usize = 192usize;
const T_162_OFFSET: usize = 6083672usize;
const T_162_LEN: usize = 2usize;
const T_163_OFFSET: usize = 6083676usize;
const T_163_LEN: usize = 2usize;
const T_164_OFFSET: usize = 6083680usize;
const T_164_LEN: usize = 6678528usize;
const T_165_OFFSET: usize = 12762208usize;
const T_165_LEN: usize = 1usize;
const T_166_OFFSET: usize = 12762212usize;
const T_166_LEN: usize = 98400usize;
const T_167_OFFSET: usize = 12860612usize;
const T_167_LEN: usize = 1usize;
const T_168_OFFSET: usize = 12860616usize;
const T_168_LEN: usize = 49248usize;
const T_185_OFFSET: usize = 12909864usize;
const T_185_LEN: usize = 2usize;
const T_199_OFFSET: usize = 12909868usize;
const T_199_LEN: usize = 1024usize;
const T_226_OFFSET: usize = 12910892usize;
const T_226_LEN: usize = 2usize;
const T_240_OFFSET: usize = 12910896usize;
const T_240_LEN: usize = 511usize;
const T_549_OFFSET: usize = 12911408usize;
const T_549_LEN: usize = 2usize;
const T_550_OFFSET: usize = 12911412usize;
const T_550_LEN: usize = 4usize;
const T_551_OFFSET: usize = 12911416usize;
const T_551_LEN: usize = 8usize;
const T_552_OFFSET: usize = 12911424usize;
const T_552_LEN: usize = 16usize;
const T_553_OFFSET: usize = 12911440usize;
const T_553_LEN: usize = 32usize;
const T_554_OFFSET: usize = 12911472usize;
const T_554_LEN: usize = 64usize;
const T_555_OFFSET: usize = 12911536usize;
const T_555_LEN: usize = 128usize;
const T_556_OFFSET: usize = 12911664usize;
const T_556_LEN: usize = 256usize;
const T_557_OFFSET: usize = 12911920usize;
const T_557_LEN: usize = 512usize;
const T_558_OFFSET: usize = 12912432usize;
const T_558_LEN: usize = 1024usize;
const T_559_OFFSET: usize = 12913456usize;
const T_559_LEN: usize = 2046usize;
const T_560_OFFSET: usize = 12915504usize;
const T_560_LEN: usize = 2usize;
const T_561_OFFSET: usize = 12915508usize;
const T_561_LEN: usize = 4usize;
const T_562_OFFSET: usize = 12915512usize;
const T_562_LEN: usize = 8usize;
const T_563_OFFSET: usize = 12915520usize;
const T_563_LEN: usize = 16usize;
const T_564_OFFSET: usize = 12915536usize;
const T_564_LEN: usize = 32usize;
const T_565_OFFSET: usize = 12915568usize;
const T_565_LEN: usize = 64usize;
const T_566_OFFSET: usize = 12915632usize;
const T_566_LEN: usize = 128usize;
const T_567_OFFSET: usize = 12915760usize;
const T_567_LEN: usize = 256usize;
const T_568_OFFSET: usize = 12916016usize;
const T_568_LEN: usize = 512usize;
const T_569_OFFSET: usize = 12916528usize;
const T_569_LEN: usize = 1022usize;
fn tensor_data_f32() -> &'static [f32] {
    unsafe { core::slice::from_raw_parts(WEIGHT_PTR as *const f32, TENSOR_DATA_FLOATS) }
}
/// Initialize the model by loading weights from file.
/// Must be called once before `forward()` or `forward_timed()`.
#[cfg(target_os = "psp")]
pub fn init() {
    use psp::sys::{
        sceIoClose, sceIoOpen, sceIoRead, sceKernelAllocPartitionMemory,
        sceKernelGetBlockHeadAddr, IoOpenFlags, SceSysMemBlockTypes,
    };
    let uid = unsafe {
        sceKernelAllocPartitionMemory(
            core::mem::transmute(2i32),
            b"weights\0".as_ptr(),
            SceSysMemBlockTypes::Low,
            WEIGHT_BYTES as u32,
            core::ptr::null_mut(),
        )
    };
    if uid.0 < 0 {
        psp_ml::dprintln!("FATAL: weight alloc failed (0x{:08X})", uid.0 as u32);
        loop {}
    }
    let ptr = unsafe { sceKernelGetBlockHeadAddr(uid) } as *mut u8;
    let fd = unsafe {
        sceIoOpen(b"host0:/weights.bin\0".as_ptr(), IoOpenFlags::RD_ONLY, 0)
    };
    if fd.0 < 0 {
        psp_ml::dprintln!("FATAL: could not open host0:/weights.bin");
        loop {}
    }
    let mut loaded = 0usize;
    while loaded < WEIGHT_BYTES {
        let chunk = if WEIGHT_BYTES - loaded < 65536 {
            WEIGHT_BYTES - loaded
        } else {
            65536
        };
        let n = unsafe {
            sceIoRead(fd, ptr.add(loaded) as *mut core::ffi::c_void, chunk as u32)
        };
        if n <= 0 {
            break;
        }
        loaded += n as usize;
    }
    unsafe { sceIoClose(fd) };
    if loaded != WEIGHT_BYTES {
        psp_ml::dprintln!(
            "FATAL: incomplete weight load: {} / {} bytes", loaded, WEIGHT_BYTES
        );
        loop {}
    }
    unsafe { WEIGHT_PTR = ptr };
    psp_ml::dprintln!("Loaded weights: {} bytes", WEIGHT_BYTES);
}
/// Initialize the model by loading weights from file.
#[cfg(not(target_os = "psp"))]
pub fn init() {
    let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/src/weights.bin"))
        .expect("failed to read weights.bin");
    assert_eq!(data.len(), WEIGHT_BYTES, "weights.bin size mismatch");
    let ptr = data.as_ptr();
    std::mem::forget(data);
    unsafe { WEIGHT_PTR = ptr };
}
