//! Generated inference module
#[allow(unused_imports)]
use psp_ml::kernels::naive::*;
#[allow(unused_imports)]
use psp_ml::kernels::*;
pub fn forward(input: &[f32; 144000usize]) -> [f32; 6522usize] {
    static mut T_169_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_169 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_169_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_170_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_170 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_170_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_171_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_171 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_171_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_172_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_172 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_172_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_173_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_173 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_173_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_174_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_174 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_174_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_175_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_175 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_175_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_187_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_187 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_187_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_188_BUF: Aligned16<2usize> = Aligned16([0.0f32; 2usize]);
    let t_188 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_188_BUF) as *mut f32,
            2usize,
        )
    };
    static mut T_200_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_200 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_200_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_201_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_201 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_201_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_202_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_202 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_202_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_203_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_203 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_203_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_206_BUF: Aligned16<1025usize> = Aligned16([0.0f32; 1025usize]);
    let t_206 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_206_BUF) as *mut f32,
            1025usize,
        )
    };
    static mut T_214_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_214 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_214_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_215_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_215 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_215_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_216_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_216 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_216_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_217_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_217 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_217_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_218_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_218 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_218_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_219_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_219 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_219_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_220_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_220 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_220_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_221_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_221 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_221_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_228_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_228 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_228_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_229_BUF: Aligned16<8usize> = Aligned16([0.0f32; 8usize]);
    let t_229 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_229_BUF) as *mut f32,
            8usize,
        )
    };
    static mut T_241_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_241 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_241_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_242_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_242 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_242_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_243_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_243 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_243_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_244_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_244 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_244_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_247_BUF: Aligned16<513usize> = Aligned16([0.0f32; 513usize]);
    let t_247 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_247_BUF) as *mut f32,
            513usize,
        )
    };
    static mut T_255_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_255 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_255_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_256_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_256 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_256_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_257_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_257 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_257_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_258_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_258 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_258_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_259_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_259 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_259_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_260_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_260 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_260_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_261_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_261 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_261_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_262_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_262 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_262_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_263_BUF: Aligned16<192usize> = Aligned16([0.0f32; 192usize]);
    let t_263 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_263_BUF) as *mut f32,
            192usize,
        )
    };
    static mut T_264_BUF: Aligned16<192usize> = Aligned16([0.0f32; 192usize]);
    let t_264 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_264_BUF) as *mut f32,
            192usize,
        )
    };
    static mut T_265_BUF: Aligned16<192usize> = Aligned16([0.0f32; 192usize]);
    let t_265 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_265_BUF) as *mut f32,
            192usize,
        )
    };
    static mut T_266_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_266 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_266_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_267_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_267 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_267_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_268_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_268 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_268_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_269_BUF: Aligned16<2304usize> = Aligned16([0.0f32; 2304usize]);
    let t_269 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_269_BUF) as *mut f32,
            2304usize,
        )
    };
    static mut T_270_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_270 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_270_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_271_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_271 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_271_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_272_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_272 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_272_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_273_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_273 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_273_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_274_BUF: Aligned16<3600usize> = Aligned16([0.0f32; 3600usize]);
    let t_274 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_274_BUF) as *mut f32,
            3600usize,
        )
    };
    static mut T_275_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_275 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_275_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_276_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_276 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_276_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_277_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_277 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_277_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_278_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_278 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_278_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_279_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_279 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_279_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_280_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_280 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_280_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_281_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_281 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_281_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_282_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_282 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_282_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_283_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_283 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_283_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_284_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_284 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_284_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_285_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_285 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_285_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_286_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_286 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_286_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_287_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_287 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_287_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_288_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_288 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_288_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_289_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_289 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_289_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_290_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_290 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_290_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_291_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_291 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_291_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_292_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_292 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_292_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_293_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_293 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_293_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_294_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_294 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_294_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_295_BUF: Aligned16<6912usize> = Aligned16([0.0f32; 6912usize]);
    let t_295 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_295_BUF) as *mut f32,
            6912usize,
        )
    };
    static mut T_296_BUF: Aligned16<6912usize> = Aligned16([0.0f32; 6912usize]);
    let t_296 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_296_BUF) as *mut f32,
            6912usize,
        )
    };
    static mut T_297_BUF: Aligned16<6912usize> = Aligned16([0.0f32; 6912usize]);
    let t_297 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_297_BUF) as *mut f32,
            6912usize,
        )
    };
    static mut T_298_BUF: Aligned16<7488usize> = Aligned16([0.0f32; 7488usize]);
    let t_298 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_298_BUF) as *mut f32,
            7488usize,
        )
    };
    static mut T_299_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_299 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_299_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_300_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_300 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_300_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_301_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_301 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_301_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_302_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_302 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_302_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_306_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_306 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_306_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_307_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_307 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_307_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_308_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_308 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_308_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_309_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_309 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_309_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_310_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_310 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_310_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_311_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_311 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_311_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_312_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_312 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_312_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_313_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_313 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_313_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_314_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_314 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_314_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_315_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_315 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_315_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_316_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_316 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_316_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_317_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_317 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_317_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_318_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_318 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_318_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_319_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_319 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_319_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_320_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_320 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_320_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_324_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_324 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_324_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_325_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_325 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_325_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_326_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_326 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_326_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_327_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_327 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_327_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_328_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_328 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_328_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_329_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_329 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_329_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_330_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_330 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_330_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_331_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_331 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_331_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_332_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_332 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_332_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_333_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_333 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_333_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_334_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_334 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_334_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_335_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_335 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_335_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_336_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_336 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_336_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_337_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_337 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_337_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_338_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_338 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_338_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_339_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_339 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_339_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_343_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_343 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_343_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_344_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_344 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_344_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_345_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_345 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_345_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_346_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_346 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_346_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_347_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_347 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_347_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_348_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_348 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_348_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_349_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_349 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_349_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_350_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_350 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_350_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_351_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_351 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_351_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_352_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_352 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_352_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_353_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_353 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_353_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_354_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_354 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_354_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_355_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_355 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_355_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_356_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_356 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_356_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_357_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_357 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_357_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_358_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_358 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_358_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_362_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_362 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_362_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_363_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_363 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_363_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_364_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_364 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_364_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_365_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_365 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_365_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_366_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_366 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_366_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_367_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_367 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_367_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_368_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_368 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_368_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_369_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_369 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_369_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_370_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_370 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_370_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_371_BUF: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let t_371 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_371_BUF) as *mut f32,
            10368usize,
        )
    };
    static mut T_372_BUF: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let t_372 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_372_BUF) as *mut f32,
            10368usize,
        )
    };
    static mut T_373_BUF: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let t_373 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_373_BUF) as *mut f32,
            10368usize,
        )
    };
    static mut T_374_BUF: Aligned16<12096usize> = Aligned16([0.0f32; 12096usize]);
    let t_374 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_374_BUF) as *mut f32,
            12096usize,
        )
    };
    static mut T_375_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_375 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_375_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_376_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_376 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_376_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_377_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_377 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_377_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_378_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_378 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_378_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_382_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_382 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_382_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_383_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_383 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_383_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_384_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_384 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_384_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_385_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_385 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_385_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_386_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_386 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_386_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_387_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_387 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_387_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_388_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_388 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_388_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_389_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_389 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_389_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_390_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_390 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_390_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_391_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_391 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_391_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_392_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_392 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_392_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_393_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_393 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_393_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_394_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_394 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_394_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_395_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_395 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_395_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_396_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_396 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_396_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_400_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_400 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_400_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_401_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_401 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_401_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_402_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_402 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_402_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_403_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_403 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_403_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_404_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_404 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_404_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_405_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_405 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_405_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_406_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_406 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_406_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_407_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_407 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_407_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_408_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_408 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_408_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_409_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_409 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_409_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_410_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_410 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_410_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_411_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_411 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_411_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_412_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_412 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_412_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_413_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_413 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_413_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_414_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_414 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_414_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_415_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_415 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_415_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_419_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_419 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_419_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_420_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_420 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_420_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_421_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_421 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_421_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_422_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_422 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_422_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_423_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_423 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_423_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_424_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_424 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_424_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_425_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_425 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_425_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_426_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_426 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_426_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_427_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_427 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_427_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_428_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_428 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_428_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_429_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_429 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_429_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_430_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_430 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_430_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_431_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_431 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_431_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_432_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_432 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_432_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_433_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_433 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_433_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_434_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_434 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_434_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_438_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_438 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_438_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_439_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_439 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_439_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_440_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_440 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_440_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_441_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_441 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_441_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_442_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_442 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_442_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_443_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_443 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_443_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_444_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_444 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_444_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_445_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_445 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_445_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_446_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_446 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_446_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_447_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_447 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_447_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_448_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_448 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_448_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_449_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_449 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_449_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_450_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_450 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_450_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_451_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_451 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_451_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_452_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_452 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_452_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_453_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_453 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_453_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_457_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_457 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_457_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_458_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_458 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_458_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_459_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_459 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_459_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_460_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_460 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_460_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_461_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_461 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_461_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_462_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_462 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_462_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_463_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_463 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_463_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_464_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_464 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_464_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_465_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_465 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_465_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_466_BUF: Aligned16<9216usize> = Aligned16([0.0f32; 9216usize]);
    let t_466 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_466_BUF) as *mut f32,
            9216usize,
        )
    };
    static mut T_467_BUF: Aligned16<9216usize> = Aligned16([0.0f32; 9216usize]);
    let t_467 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_467_BUF) as *mut f32,
            9216usize,
        )
    };
    static mut T_468_BUF: Aligned16<9216usize> = Aligned16([0.0f32; 9216usize]);
    let t_468 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_468_BUF) as *mut f32,
            9216usize,
        )
    };
    static mut T_469_BUF: Aligned16<12288usize> = Aligned16([0.0f32; 12288usize]);
    let t_469 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_469_BUF) as *mut f32,
            12288usize,
        )
    };
    static mut T_470_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_470 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_470_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_471_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_471 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_471_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_472_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_472 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_472_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_473_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_473 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_473_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_477_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_477 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_477_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_478_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_478 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_478_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_479_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_479 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_479_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_480_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_480 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_480_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_481_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_481 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_481_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_482_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_482 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_482_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_483_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_483 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_483_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_484_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_484 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_484_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_485_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_485 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_485_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_486_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_486 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_486_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_487_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_487 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_487_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_488_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_488 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_488_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_489_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_489 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_489_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_490_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_490 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_490_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_491_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_491 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_491_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_495_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_495 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_495_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_496_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_496 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_496_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_497_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_497 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_497_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_498_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_498 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_498_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_499_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_499 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_499_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_500_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_500 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_500_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_501_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_501 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_501_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_502_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_502 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_502_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_503_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_503 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_503_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_504_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_504 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_504_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_505_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_505 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_505_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_506_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_506 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_506_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_507_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_507 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_507_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_508_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_508 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_508_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_509_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_509 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_509_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_510_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_510 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_510_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_514_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_514 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_514_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_515_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_515 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_515_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_516_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_516 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_516_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_517_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_517 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_517_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_518_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_518 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_518_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_519_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_519 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_519_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_520_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_520 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_520_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_521_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_521 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_521_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_522_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_522 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_522_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_523_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_523 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_523_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_524_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_524 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_524_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_525_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_525 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_525_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_526_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_526 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_526_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_527_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_527 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_527_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_528_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_528 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_528_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_529_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_529 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_529_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_533_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_533 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_533_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_534_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_534 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_534_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_535_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_535 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_535_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_536_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_536 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_536_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_537_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_537 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_537_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_538_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_538 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_538_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_539_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_539 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_539_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_540_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_540 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_540_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_541_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_541 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_541_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_542_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_542 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_542_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_543_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_543 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_543_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_544_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_544 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_544_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_545_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_545 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_545_BUF) as *mut f32,
            1024usize,
        )
    };
    let mut t_546 = [0.0f32; 6522usize];
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
    let t_547 = &tensor_data[T_547_OFFSET..T_547_OFFSET + T_547_LEN];
    let t_548 = &tensor_data[T_548_OFFSET..T_548_OFFSET + T_548_LEN];
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
        &[1usize, 1usize, 2usize],
        unsafe { core::slice::from_raw_parts(t_199.as_ptr() as *const i32, 1024usize) },
        t_200,
        &[1usize, 1usize, 1024usize, 2usize],
        1usize,
    );
    reshape(t_200, t_201);
    binary_mul(t_201, t_129, t_202, 2048usize);
    reshape(t_202, t_203);
    static mut SCRATCH_13_0: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let scratch_13_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_13_0) as *mut f32,
            2048usize,
        )
    };
    rfft_pack(t_203, scratch_13_0, 2048usize);
    fft_butterfly_stage(scratch_13_0, t_547, 1024usize, 1usize);
    fft_butterfly_stage(scratch_13_0, t_548, 1024usize, 2usize);
    fft_butterfly_stage(scratch_13_0, t_549, 1024usize, 4usize);
    fft_butterfly_stage(scratch_13_0, t_550, 1024usize, 8usize);
    fft_butterfly_stage(scratch_13_0, t_551, 1024usize, 16usize);
    fft_butterfly_stage(scratch_13_0, t_552, 1024usize, 32usize);
    fft_butterfly_stage(scratch_13_0, t_553, 1024usize, 64usize);
    fft_butterfly_stage(scratch_13_0, t_554, 1024usize, 128usize);
    fft_butterfly_stage(scratch_13_0, t_555, 1024usize, 256usize);
    fft_butterfly_stage(scratch_13_0, t_556, 1024usize, 512usize);
    rfft_unpack(scratch_13_0, t_557, t_206, 2048usize);
    reshape(t_206, t_214);
    fully_connected(t_214, 1usize, t_166, None, t_215, 96usize);
    reshape(t_215, t_216);
    binary_mul(t_216, t_216, t_217, 96usize);
    binary_pow(t_217, t_165, t_218, 1usize);
    reverse_v2(t_218, &[1usize, 1usize, 96usize], t_219, 2usize);
    transpose(
        t_219,
        &[1usize, 1usize, 96usize],
        t_220,
        &[1usize, 96usize, 1usize],
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
        &[1usize, 1usize, 8usize],
        unsafe { core::slice::from_raw_parts(t_240.as_ptr() as *const i32, 511usize) },
        t_241,
        &[1usize, 1usize, 128usize, 8usize],
        1usize,
    );
    reshape(t_241, t_242);
    binary_mul(t_242, t_126, t_243, 1024usize);
    reshape(t_243, t_244);
    static mut SCRATCH_28_0: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let scratch_28_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_28_0) as *mut f32,
            1024usize,
        )
    };
    rfft_pack(t_244, scratch_28_0, 1024usize);
    fft_butterfly_stage(scratch_28_0, t_558, 512usize, 1usize);
    fft_butterfly_stage(scratch_28_0, t_559, 512usize, 2usize);
    fft_butterfly_stage(scratch_28_0, t_560, 512usize, 4usize);
    fft_butterfly_stage(scratch_28_0, t_561, 512usize, 8usize);
    fft_butterfly_stage(scratch_28_0, t_562, 512usize, 16usize);
    fft_butterfly_stage(scratch_28_0, t_563, 512usize, 32usize);
    fft_butterfly_stage(scratch_28_0, t_564, 512usize, 64usize);
    fft_butterfly_stage(scratch_28_0, t_565, 512usize, 128usize);
    fft_butterfly_stage(scratch_28_0, t_566, 512usize, 256usize);
    rfft_unpack(scratch_28_0, t_567, t_247, 1024usize);
    reshape(t_247, t_255);
    fully_connected(t_255, 1usize, t_168, None, t_256, 96usize);
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
        let src = t_221;
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
    static mut SCRATCH_40_0: Aligned16<3072usize> = Aligned16([0.0f32; 3072usize]);
    let scratch_40_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_40_0) as *mut f32,
            3072usize,
        )
    };
    static mut SCRATCH_40_1: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let scratch_40_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_40_1) as *mut f32,
            1536usize,
        )
    };
    scratch_40_1.copy_from_slice(t_122);
    im2col_padded(
        t_265,
        [1usize, 96usize, 1usize, 2usize],
        [4usize, 8usize],
        [2usize, 2usize],
        [1usize, 1usize, 3usize, 4usize],
        [48usize, 1usize],
        scratch_40_0,
    );
    matmul_bt_tiled(scratch_40_0, scratch_40_1, t_266, 12usize, 16usize, 6usize);
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
    static mut SCRATCH_44_0: Aligned16<2304usize> = Aligned16([0.0f32; 2304usize]);
    let scratch_44_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_44_0) as *mut f32,
            2304usize,
        )
    };
    static mut SCRATCH_44_1: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let scratch_44_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_44_1) as *mut f32,
            1152usize,
        )
    };
    scratch_44_1.copy_from_slice(t_121);
    im2col_padded(
        t_269,
        [1usize, 48usize, 1usize, 48usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [48usize, 1usize],
        scratch_44_0,
    );
    matmul_bt_tiled(scratch_44_0, scratch_44_1, t_270, 12usize, 12usize, 6usize);
    bias_add(t_270, t_50, 48usize, 24usize);
    static mut SCRATCH_45_0: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let scratch_45_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_45_0) as *mut f32,
            1152usize,
        )
    };
    static mut SCRATCH_45_1: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_45_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_45_1) as *mut f32,
            1728usize,
        )
    };
    scratch_45_1.copy_from_slice(t_120);
    im2col_padded(
        t_270,
        [1usize, 48usize, 1usize, 24usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [48usize, 1usize],
        scratch_45_0,
    );
    matmul_bt_tiled(scratch_45_0, scratch_45_1, t_271, 12usize, 6usize, 18usize);
    bias_add(t_271, t_49, 48usize, 72usize);
    unary_logistic(t_271, t_272);
    binary_mul(t_271, t_272, t_273, 3456usize);
    pad(
        t_273,
        [1usize, 48usize, 1usize, 72usize],
        t_274,
        [1usize, 50usize, 1usize, 72usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    depthwise_conv2d(
        t_274,
        [1usize, 50usize, 1usize, 72usize],
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
    static mut SCRATCH_52_0: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_52_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_52_0) as *mut f32,
            1728usize,
        )
    };
    static mut SCRATCH_52_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_52_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_52_1) as *mut f32,
            2592usize,
        )
    };
    scratch_52_1.copy_from_slice(t_118);
    im2col_padded(
        t_277,
        [1usize, 24usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_52_0,
    );
    matmul_bt_tiled(scratch_52_0, scratch_52_1, t_278, 6usize, 18usize, 9usize);
    bias_add(t_278, t_117, 24usize, 36usize);
    static mut SCRATCH_53_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_53_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_53_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_53_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_53_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_53_1) as *mut f32,
            2592usize,
        )
    };
    scratch_53_1.copy_from_slice(t_116);
    im2col_padded(
        t_278,
        [1usize, 24usize, 1usize, 36usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_53_0,
    );
    matmul_bt_tiled(scratch_53_0, scratch_53_1, t_279, 6usize, 9usize, 18usize);
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
    static mut SCRATCH_59_0: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_59_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_59_0) as *mut f32,
            1728usize,
        )
    };
    static mut SCRATCH_59_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_59_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_59_1) as *mut f32,
            2592usize,
        )
    };
    scratch_59_1.copy_from_slice(t_115);
    im2col_padded(
        t_284,
        [1usize, 24usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_59_0,
    );
    matmul_bt_tiled(scratch_59_0, scratch_59_1, t_285, 6usize, 18usize, 9usize);
    bias_add(t_285, t_117, 24usize, 36usize);
    binary_add(t_285, t_278, t_286, 864usize);
    static mut SCRATCH_61_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_61_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_61_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_61_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_61_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_61_1) as *mut f32,
            2592usize,
        )
    };
    scratch_61_1.copy_from_slice(t_114);
    im2col_padded(
        t_286,
        [1usize, 24usize, 1usize, 36usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_61_0,
    );
    matmul_bt_tiled(scratch_61_0, scratch_61_1, t_287, 6usize, 9usize, 18usize);
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
    static mut SCRATCH_67_0: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_67_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_67_0) as *mut f32,
            1728usize,
        )
    };
    static mut SCRATCH_67_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_67_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_67_1) as *mut f32,
            2592usize,
        )
    };
    scratch_67_1.copy_from_slice(t_113);
    im2col_padded(
        t_292,
        [1usize, 24usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_67_0,
    );
    matmul_bt_tiled(scratch_67_0, scratch_67_1, t_293, 6usize, 18usize, 9usize);
    bias_add(t_293, t_117, 24usize, 36usize);
    binary_add(t_293, t_286, t_294, 864usize);
    static mut SCRATCH_69_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_69_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_69_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_69_1: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let scratch_69_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_69_1) as *mut f32,
            10368usize,
        )
    };
    scratch_69_1.copy_from_slice(t_112);
    im2col_padded(
        t_294,
        [1usize, 24usize, 1usize, 36usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_69_0,
    );
    matmul_bt_tiled(scratch_69_0, scratch_69_1, t_295, 6usize, 9usize, 72usize);
    bias_add(t_295, t_40, 24usize, 288usize);
    unary_logistic(t_295, t_296);
    binary_mul(t_295, t_296, t_297, 6912usize);
    pad(
        t_297,
        [1usize, 24usize, 1usize, 288usize],
        t_298,
        [1usize, 26usize, 1usize, 288usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    depthwise_conv2d(
        t_298,
        [1usize, 26usize, 1usize, 288usize],
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
    static mut SCRATCH_84_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_84_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_84_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_84_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_84_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_84_1) as *mut f32,
            20736usize,
        )
    };
    scratch_84_1.copy_from_slice(t_107);
    im2col_padded(
        t_312,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_84_0,
    );
    matmul_bt_tiled(scratch_84_0, scratch_84_1, t_313, 3usize, 72usize, 18usize);
    bias_add(t_313, t_119, 12usize, 72usize);
    static mut SCRATCH_85_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_85_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_85_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_85_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_85_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_85_1) as *mut f32,
            20736usize,
        )
    };
    scratch_85_1.copy_from_slice(t_106);
    im2col_padded(
        t_313,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_85_0,
    );
    matmul_bt_tiled(scratch_85_0, scratch_85_1, t_314, 3usize, 18usize, 72usize);
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
    static mut SCRATCH_99_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_99_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_99_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_99_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_99_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_99_1) as *mut f32,
            20736usize,
        )
    };
    scratch_99_1.copy_from_slice(t_103);
    im2col_padded(
        t_330,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_99_0,
    );
    matmul_bt_tiled(scratch_99_0, scratch_99_1, t_331, 3usize, 72usize, 18usize);
    bias_add(t_331, t_119, 12usize, 72usize);
    binary_add(t_331, t_313, t_332, 864usize);
    static mut SCRATCH_101_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_101_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_101_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_101_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_101_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_101_1) as *mut f32,
            20736usize,
        )
    };
    scratch_101_1.copy_from_slice(t_102);
    im2col_padded(
        t_332,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_101_0,
    );
    matmul_bt_tiled(scratch_101_0, scratch_101_1, t_333, 3usize, 18usize, 72usize);
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
    static mut SCRATCH_115_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_115_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_115_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_115_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_115_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_115_1) as *mut f32,
            20736usize,
        )
    };
    scratch_115_1.copy_from_slice(t_99);
    im2col_padded(
        t_349,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_115_0,
    );
    matmul_bt_tiled(scratch_115_0, scratch_115_1, t_350, 3usize, 72usize, 18usize);
    bias_add(t_350, t_119, 12usize, 72usize);
    binary_add(t_350, t_332, t_351, 864usize);
    static mut SCRATCH_117_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_117_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_117_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_117_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_117_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_117_1) as *mut f32,
            20736usize,
        )
    };
    scratch_117_1.copy_from_slice(t_98);
    im2col_padded(
        t_351,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_117_0,
    );
    matmul_bt_tiled(scratch_117_0, scratch_117_1, t_352, 3usize, 18usize, 72usize);
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
    static mut SCRATCH_131_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_131_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_131_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_131_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_131_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_131_1) as *mut f32,
            20736usize,
        )
    };
    scratch_131_1.copy_from_slice(t_95);
    im2col_padded(
        t_368,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_131_0,
    );
    matmul_bt_tiled(scratch_131_0, scratch_131_1, t_369, 3usize, 72usize, 18usize);
    bias_add(t_369, t_119, 12usize, 72usize);
    binary_add(t_369, t_351, t_370, 864usize);
    static mut SCRATCH_133_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_133_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_133_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_133_1: Aligned16<62208usize> = Aligned16([0.0f32; 62208usize]);
    let scratch_133_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_133_1) as *mut f32,
            62208usize,
        )
    };
    scratch_133_1.copy_from_slice(t_94);
    im2col_padded(
        t_370,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_133_0,
    );
    matmul_bt_tiled(scratch_133_0, scratch_133_1, t_371, 3usize, 18usize, 216usize);
    bias_add(t_371, t_28, 12usize, 864usize);
    unary_logistic(t_371, t_372);
    binary_mul(t_371, t_372, t_373, 10368usize);
    pad(
        t_373,
        [1usize, 12usize, 1usize, 864usize],
        t_374,
        [1usize, 14usize, 1usize, 864usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    depthwise_conv2d(
        t_374,
        [1usize, 14usize, 1usize, 864usize],
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
        [1usize, 8usize, 1usize, 1536usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    depthwise_conv2d(
        t_469,
        [1usize, 8usize, 1usize, 1536usize],
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
    fully_connected(t_545, 1024usize, t_164, Some(t_142), &mut t_546, 6522usize);
    t_546
}
/// Instrumented inference: accumulates per-op tick deltas into `op_ticks`.
pub fn forward_timed(
    input: &[f32; 144000usize],
    op_ticks: &mut [u64; NUM_OPS],
    get_tick: fn() -> u64,
) -> [f32; 6522usize] {
    static mut T_169_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_169 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_169_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_170_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_170 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_170_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_171_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_171 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_171_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_172_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_172 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_172_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_173_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_173 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_173_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_174_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_174 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_174_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_175_BUF: Aligned16<144000usize> = Aligned16([0.0f32; 144000usize]);
    let t_175 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_175_BUF) as *mut f32,
            144000usize,
        )
    };
    static mut T_187_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_187 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_187_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_188_BUF: Aligned16<2usize> = Aligned16([0.0f32; 2usize]);
    let t_188 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_188_BUF) as *mut f32,
            2usize,
        )
    };
    static mut T_200_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_200 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_200_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_201_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_201 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_201_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_202_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_202 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_202_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_203_BUF: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let t_203 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_203_BUF) as *mut f32,
            2048usize,
        )
    };
    static mut T_206_BUF: Aligned16<1025usize> = Aligned16([0.0f32; 1025usize]);
    let t_206 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_206_BUF) as *mut f32,
            1025usize,
        )
    };
    static mut T_214_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_214 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_214_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_215_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_215 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_215_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_216_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_216 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_216_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_217_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_217 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_217_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_218_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_218 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_218_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_219_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_219 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_219_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_220_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_220 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_220_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_221_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_221 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_221_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_228_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_228 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_228_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_229_BUF: Aligned16<8usize> = Aligned16([0.0f32; 8usize]);
    let t_229 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_229_BUF) as *mut f32,
            8usize,
        )
    };
    static mut T_241_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_241 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_241_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_242_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_242 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_242_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_243_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_243 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_243_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_244_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_244 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_244_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_247_BUF: Aligned16<513usize> = Aligned16([0.0f32; 513usize]);
    let t_247 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_247_BUF) as *mut f32,
            513usize,
        )
    };
    static mut T_255_BUF: Aligned16<1usize> = Aligned16([0.0f32; 1usize]);
    let t_255 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_255_BUF) as *mut f32,
            1usize,
        )
    };
    static mut T_256_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_256 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_256_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_257_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_257 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_257_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_258_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_258 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_258_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_259_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_259 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_259_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_260_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_260 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_260_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_261_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_261 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_261_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_262_BUF: Aligned16<96usize> = Aligned16([0.0f32; 96usize]);
    let t_262 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_262_BUF) as *mut f32,
            96usize,
        )
    };
    static mut T_263_BUF: Aligned16<192usize> = Aligned16([0.0f32; 192usize]);
    let t_263 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_263_BUF) as *mut f32,
            192usize,
        )
    };
    static mut T_264_BUF: Aligned16<192usize> = Aligned16([0.0f32; 192usize]);
    let t_264 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_264_BUF) as *mut f32,
            192usize,
        )
    };
    static mut T_265_BUF: Aligned16<192usize> = Aligned16([0.0f32; 192usize]);
    let t_265 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_265_BUF) as *mut f32,
            192usize,
        )
    };
    static mut T_266_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_266 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_266_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_267_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_267 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_267_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_268_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_268 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_268_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_269_BUF: Aligned16<2304usize> = Aligned16([0.0f32; 2304usize]);
    let t_269 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_269_BUF) as *mut f32,
            2304usize,
        )
    };
    static mut T_270_BUF: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let t_270 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_270_BUF) as *mut f32,
            1152usize,
        )
    };
    static mut T_271_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_271 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_271_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_272_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_272 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_272_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_273_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_273 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_273_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_274_BUF: Aligned16<3600usize> = Aligned16([0.0f32; 3600usize]);
    let t_274 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_274_BUF) as *mut f32,
            3600usize,
        )
    };
    static mut T_275_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_275 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_275_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_276_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_276 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_276_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_277_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_277 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_277_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_278_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_278 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_278_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_279_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_279 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_279_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_280_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_280 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_280_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_281_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_281 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_281_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_282_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_282 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_282_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_283_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_283 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_283_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_284_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_284 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_284_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_285_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_285 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_285_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_286_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_286 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_286_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_287_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_287 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_287_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_288_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_288 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_288_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_289_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_289 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_289_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_290_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_290 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_290_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_291_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_291 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_291_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_292_BUF: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let t_292 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_292_BUF) as *mut f32,
            1728usize,
        )
    };
    static mut T_293_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_293 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_293_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_294_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_294 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_294_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_295_BUF: Aligned16<6912usize> = Aligned16([0.0f32; 6912usize]);
    let t_295 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_295_BUF) as *mut f32,
            6912usize,
        )
    };
    static mut T_296_BUF: Aligned16<6912usize> = Aligned16([0.0f32; 6912usize]);
    let t_296 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_296_BUF) as *mut f32,
            6912usize,
        )
    };
    static mut T_297_BUF: Aligned16<6912usize> = Aligned16([0.0f32; 6912usize]);
    let t_297 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_297_BUF) as *mut f32,
            6912usize,
        )
    };
    static mut T_298_BUF: Aligned16<7488usize> = Aligned16([0.0f32; 7488usize]);
    let t_298 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_298_BUF) as *mut f32,
            7488usize,
        )
    };
    static mut T_299_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_299 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_299_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_300_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_300 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_300_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_301_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_301 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_301_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_302_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_302 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_302_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_306_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_306 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_306_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_307_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_307 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_307_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_308_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_308 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_308_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_309_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_309 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_309_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_310_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_310 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_310_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_311_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_311 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_311_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_312_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_312 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_312_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_313_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_313 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_313_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_314_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_314 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_314_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_315_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_315 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_315_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_316_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_316 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_316_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_317_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_317 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_317_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_318_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_318 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_318_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_319_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_319 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_319_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_320_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_320 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_320_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_324_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_324 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_324_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_325_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_325 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_325_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_326_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_326 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_326_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_327_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_327 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_327_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_328_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_328 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_328_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_329_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_329 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_329_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_330_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_330 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_330_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_331_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_331 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_331_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_332_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_332 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_332_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_333_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_333 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_333_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_334_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_334 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_334_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_335_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_335 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_335_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_336_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_336 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_336_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_337_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_337 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_337_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_338_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_338 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_338_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_339_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_339 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_339_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_343_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_343 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_343_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_344_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_344 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_344_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_345_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_345 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_345_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_346_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_346 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_346_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_347_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_347 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_347_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_348_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_348 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_348_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_349_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_349 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_349_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_350_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_350 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_350_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_351_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_351 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_351_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_352_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_352 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_352_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_353_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_353 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_353_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_354_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_354 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_354_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_355_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_355 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_355_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_356_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_356 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_356_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_357_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_357 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_357_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_358_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_358 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_358_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_362_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_362 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_362_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_363_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_363 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_363_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_364_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_364 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_364_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_365_BUF: Aligned16<18usize> = Aligned16([0.0f32; 18usize]);
    let t_365 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_365_BUF) as *mut f32,
            18usize,
        )
    };
    static mut T_366_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_366 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_366_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_367_BUF: Aligned16<288usize> = Aligned16([0.0f32; 288usize]);
    let t_367 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_367_BUF) as *mut f32,
            288usize,
        )
    };
    static mut T_368_BUF: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let t_368 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_368_BUF) as *mut f32,
            3456usize,
        )
    };
    static mut T_369_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_369 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_369_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_370_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_370 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_370_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_371_BUF: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let t_371 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_371_BUF) as *mut f32,
            10368usize,
        )
    };
    static mut T_372_BUF: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let t_372 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_372_BUF) as *mut f32,
            10368usize,
        )
    };
    static mut T_373_BUF: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let t_373 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_373_BUF) as *mut f32,
            10368usize,
        )
    };
    static mut T_374_BUF: Aligned16<12096usize> = Aligned16([0.0f32; 12096usize]);
    let t_374 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_374_BUF) as *mut f32,
            12096usize,
        )
    };
    static mut T_375_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_375 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_375_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_376_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_376 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_376_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_377_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_377 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_377_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_378_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_378 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_378_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_382_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_382 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_382_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_383_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_383 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_383_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_384_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_384 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_384_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_385_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_385 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_385_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_386_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_386 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_386_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_387_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_387 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_387_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_388_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_388 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_388_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_389_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_389 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_389_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_390_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_390 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_390_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_391_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_391 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_391_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_392_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_392 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_392_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_393_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_393 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_393_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_394_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_394 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_394_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_395_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_395 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_395_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_396_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_396 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_396_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_400_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_400 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_400_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_401_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_401 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_401_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_402_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_402 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_402_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_403_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_403 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_403_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_404_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_404 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_404_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_405_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_405 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_405_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_406_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_406 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_406_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_407_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_407 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_407_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_408_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_408 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_408_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_409_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_409 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_409_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_410_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_410 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_410_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_411_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_411 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_411_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_412_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_412 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_412_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_413_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_413 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_413_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_414_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_414 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_414_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_415_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_415 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_415_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_419_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_419 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_419_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_420_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_420 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_420_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_421_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_421 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_421_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_422_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_422 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_422_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_423_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_423 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_423_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_424_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_424 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_424_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_425_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_425 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_425_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_426_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_426 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_426_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_427_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_427 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_427_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_428_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_428 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_428_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_429_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_429 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_429_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_430_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_430 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_430_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_431_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_431 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_431_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_432_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_432 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_432_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_433_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_433 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_433_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_434_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_434 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_434_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_438_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_438 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_438_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_439_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_439 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_439_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_440_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_440 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_440_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_441_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_441 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_441_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_442_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_442 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_442_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_443_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_443 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_443_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_444_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_444 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_444_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_445_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_445 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_445_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_446_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_446 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_446_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_447_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_447 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_447_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_448_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_448 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_448_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_449_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_449 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_449_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_450_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_450 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_450_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_451_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_451 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_451_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_452_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_452 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_452_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_453_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_453 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_453_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_457_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_457 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_457_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_458_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_458 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_458_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_459_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_459 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_459_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_460_BUF: Aligned16<27usize> = Aligned16([0.0f32; 27usize]);
    let t_460 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_460_BUF) as *mut f32,
            27usize,
        )
    };
    static mut T_461_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_461 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_461_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_462_BUF: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let t_462 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_462_BUF) as *mut f32,
            864usize,
        )
    };
    static mut T_463_BUF: Aligned16<5184usize> = Aligned16([0.0f32; 5184usize]);
    let t_463 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_463_BUF) as *mut f32,
            5184usize,
        )
    };
    static mut T_464_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_464 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_464_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_465_BUF: Aligned16<648usize> = Aligned16([0.0f32; 648usize]);
    let t_465 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_465_BUF) as *mut f32,
            648usize,
        )
    };
    static mut T_466_BUF: Aligned16<9216usize> = Aligned16([0.0f32; 9216usize]);
    let t_466 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_466_BUF) as *mut f32,
            9216usize,
        )
    };
    static mut T_467_BUF: Aligned16<9216usize> = Aligned16([0.0f32; 9216usize]);
    let t_467 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_467_BUF) as *mut f32,
            9216usize,
        )
    };
    static mut T_468_BUF: Aligned16<9216usize> = Aligned16([0.0f32; 9216usize]);
    let t_468 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_468_BUF) as *mut f32,
            9216usize,
        )
    };
    static mut T_469_BUF: Aligned16<12288usize> = Aligned16([0.0f32; 12288usize]);
    let t_469 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_469_BUF) as *mut f32,
            12288usize,
        )
    };
    static mut T_470_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_470 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_470_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_471_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_471 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_471_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_472_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_472 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_472_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_473_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_473 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_473_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_477_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_477 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_477_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_478_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_478 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_478_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_479_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_479 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_479_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_480_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_480 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_480_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_481_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_481 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_481_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_482_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_482 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_482_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_483_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_483 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_483_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_484_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_484 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_484_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_485_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_485 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_485_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_486_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_486 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_486_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_487_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_487 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_487_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_488_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_488 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_488_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_489_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_489 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_489_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_490_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_490 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_490_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_491_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_491 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_491_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_495_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_495 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_495_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_496_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_496 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_496_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_497_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_497 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_497_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_498_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_498 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_498_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_499_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_499 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_499_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_500_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_500 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_500_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_501_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_501 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_501_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_502_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_502 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_502_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_503_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_503 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_503_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_504_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_504 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_504_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_505_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_505 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_505_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_506_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_506 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_506_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_507_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_507 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_507_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_508_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_508 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_508_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_509_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_509 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_509_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_510_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_510 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_510_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_514_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_514 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_514_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_515_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_515 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_515_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_516_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_516 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_516_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_517_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_517 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_517_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_518_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_518 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_518_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_519_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_519 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_519_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_520_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_520 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_520_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_521_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_521 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_521_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_522_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_522 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_522_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_523_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_523 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_523_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_524_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_524 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_524_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_525_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_525 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_525_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_526_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_526 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_526_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_527_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_527 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_527_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_528_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_528 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_528_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_529_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_529 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_529_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_533_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_533 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_533_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_534_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_534 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_534_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_535_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_535 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_535_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_536_BUF: Aligned16<48usize> = Aligned16([0.0f32; 48usize]);
    let t_536 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_536_BUF) as *mut f32,
            48usize,
        )
    };
    static mut T_537_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_537 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_537_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_538_BUF: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let t_538 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_538_BUF) as *mut f32,
            1536usize,
        )
    };
    static mut T_539_BUF: Aligned16<4608usize> = Aligned16([0.0f32; 4608usize]);
    let t_539 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_539_BUF) as *mut f32,
            4608usize,
        )
    };
    static mut T_540_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_540 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_540_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_541_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_541 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_541_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_542_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_542 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_542_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_543_BUF: Aligned16<576usize> = Aligned16([0.0f32; 576usize]);
    let t_543 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_543_BUF) as *mut f32,
            576usize,
        )
    };
    static mut T_544_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_544 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_544_BUF) as *mut f32,
            1024usize,
        )
    };
    static mut T_545_BUF: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let t_545 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(T_545_BUF) as *mut f32,
            1024usize,
        )
    };
    let mut t_546 = [0.0f32; 6522usize];
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
    let t_547 = &tensor_data[T_547_OFFSET..T_547_OFFSET + T_547_LEN];
    let t_548 = &tensor_data[T_548_OFFSET..T_548_OFFSET + T_548_LEN];
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
        &[1usize, 1usize, 2usize],
        unsafe { core::slice::from_raw_parts(t_199.as_ptr() as *const i32, 1024usize) },
        t_200,
        &[1usize, 1usize, 1024usize, 2usize],
        1usize,
    );
    op_ticks[9usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_200, t_201);
    op_ticks[10usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_201, t_129, t_202, 2048usize);
    op_ticks[11usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_202, t_203);
    op_ticks[12usize] += get_tick() - __t0;
    static mut SCRATCH_13_0: Aligned16<2048usize> = Aligned16([0.0f32; 2048usize]);
    let scratch_13_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_13_0) as *mut f32,
            2048usize,
        )
    };
    let __t0 = get_tick();
    rfft_pack(t_203, scratch_13_0, 2048usize);
    op_ticks[13usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_547, 1024usize, 1usize);
    op_ticks[14usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_548, 1024usize, 2usize);
    op_ticks[15usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_549, 1024usize, 4usize);
    op_ticks[16usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_550, 1024usize, 8usize);
    op_ticks[17usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_551, 1024usize, 16usize);
    op_ticks[18usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_552, 1024usize, 32usize);
    op_ticks[19usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_553, 1024usize, 64usize);
    op_ticks[20usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_554, 1024usize, 128usize);
    op_ticks[21usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_555, 1024usize, 256usize);
    op_ticks[22usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_13_0, t_556, 1024usize, 512usize);
    op_ticks[23usize] += get_tick() - __t0;
    let __t0 = get_tick();
    rfft_unpack(scratch_13_0, t_557, t_206, 2048usize);
    op_ticks[24usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_206, t_214);
    op_ticks[25usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fully_connected(t_214, 1usize, t_166, None, t_215, 96usize);
    op_ticks[26usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_215, t_216);
    op_ticks[27usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_216, t_216, t_217, 96usize);
    op_ticks[28usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_pow(t_217, t_165, t_218, 1usize);
    op_ticks[29usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reverse_v2(t_218, &[1usize, 1usize, 96usize], t_219, 2usize);
    op_ticks[30usize] += get_tick() - __t0;
    let __t0 = get_tick();
    transpose(
        t_219,
        &[1usize, 1usize, 96usize],
        t_220,
        &[1usize, 96usize, 1usize],
        &[0usize, 2usize, 1usize],
    );
    op_ticks[31usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_220, t_221);
    op_ticks[32usize] += get_tick() - __t0;
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
    op_ticks[33usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_228, t_229);
    op_ticks[34usize] += get_tick() - __t0;
    let __t0 = get_tick();
    gather(
        t_229,
        &[1usize, 1usize, 8usize],
        unsafe { core::slice::from_raw_parts(t_240.as_ptr() as *const i32, 511usize) },
        t_241,
        &[1usize, 1usize, 128usize, 8usize],
        1usize,
    );
    op_ticks[35usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_241, t_242);
    op_ticks[36usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_242, t_126, t_243, 1024usize);
    op_ticks[37usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_243, t_244);
    op_ticks[38usize] += get_tick() - __t0;
    static mut SCRATCH_28_0: Aligned16<1024usize> = Aligned16([0.0f32; 1024usize]);
    let scratch_28_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_28_0) as *mut f32,
            1024usize,
        )
    };
    let __t0 = get_tick();
    rfft_pack(t_244, scratch_28_0, 1024usize);
    op_ticks[39usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_558, 512usize, 1usize);
    op_ticks[40usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_559, 512usize, 2usize);
    op_ticks[41usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_560, 512usize, 4usize);
    op_ticks[42usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_561, 512usize, 8usize);
    op_ticks[43usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_562, 512usize, 16usize);
    op_ticks[44usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_563, 512usize, 32usize);
    op_ticks[45usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_564, 512usize, 64usize);
    op_ticks[46usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_565, 512usize, 128usize);
    op_ticks[47usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fft_butterfly_stage(scratch_28_0, t_566, 512usize, 256usize);
    op_ticks[48usize] += get_tick() - __t0;
    let __t0 = get_tick();
    rfft_unpack(scratch_28_0, t_567, t_247, 1024usize);
    op_ticks[49usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_247, t_255);
    op_ticks[50usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fully_connected(t_255, 1usize, t_168, None, t_256, 96usize);
    op_ticks[51usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_256, t_257);
    op_ticks[52usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_257, t_257, t_258, 96usize);
    op_ticks[53usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_pow(t_258, t_167, t_259, 1usize);
    op_ticks[54usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reverse_v2(t_259, &[1usize, 1usize, 96usize], t_260, 2usize);
    op_ticks[55usize] += get_tick() - __t0;
    let __t0 = get_tick();
    transpose(
        t_260,
        &[1usize, 1usize, 96usize],
        t_261,
        &[1usize, 96usize, 1usize],
        &[0usize, 2usize, 1usize],
    );
    op_ticks[56usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_261, t_262);
    op_ticks[57usize] += get_tick() - __t0;
    let __t0 = get_tick();
    {
        let src = t_221;
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
    op_ticks[58usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_263, t_163, t_264, 2usize);
    op_ticks[59usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_264, t_162, t_265, 2usize);
    op_ticks[60usize] += get_tick() - __t0;
    static mut SCRATCH_40_0: Aligned16<3072usize> = Aligned16([0.0f32; 3072usize]);
    let scratch_40_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_40_0) as *mut f32,
            3072usize,
        )
    };
    static mut SCRATCH_40_1: Aligned16<1536usize> = Aligned16([0.0f32; 1536usize]);
    let scratch_40_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_40_1) as *mut f32,
            1536usize,
        )
    };
    scratch_40_1.copy_from_slice(t_122);
    let __t0 = get_tick();
    im2col_padded(
        t_265,
        [1usize, 96usize, 1usize, 2usize],
        [4usize, 8usize],
        [2usize, 2usize],
        [1usize, 1usize, 3usize, 4usize],
        [48usize, 1usize],
        scratch_40_0,
    );
    op_ticks[61usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_40_0, scratch_40_1, t_266, 12usize, 16usize, 6usize);
    op_ticks[62usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_266, t_51, 48usize, 24usize);
    relu(t_266);
    op_ticks[63usize] += get_tick() - __t0;
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
    op_ticks[64usize] += get_tick() - __t0;
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
    op_ticks[65usize] += get_tick() - __t0;
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
    op_ticks[66usize] += get_tick() - __t0;
    static mut SCRATCH_44_0: Aligned16<2304usize> = Aligned16([0.0f32; 2304usize]);
    let scratch_44_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_44_0) as *mut f32,
            2304usize,
        )
    };
    static mut SCRATCH_44_1: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let scratch_44_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_44_1) as *mut f32,
            1152usize,
        )
    };
    scratch_44_1.copy_from_slice(t_121);
    let __t0 = get_tick();
    im2col_padded(
        t_269,
        [1usize, 48usize, 1usize, 48usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [48usize, 1usize],
        scratch_44_0,
    );
    op_ticks[67usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_44_0, scratch_44_1, t_270, 12usize, 12usize, 6usize);
    op_ticks[68usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_270, t_50, 48usize, 24usize);
    op_ticks[69usize] += get_tick() - __t0;
    static mut SCRATCH_45_0: Aligned16<1152usize> = Aligned16([0.0f32; 1152usize]);
    let scratch_45_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_45_0) as *mut f32,
            1152usize,
        )
    };
    static mut SCRATCH_45_1: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_45_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_45_1) as *mut f32,
            1728usize,
        )
    };
    scratch_45_1.copy_from_slice(t_120);
    let __t0 = get_tick();
    im2col_padded(
        t_270,
        [1usize, 48usize, 1usize, 24usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [48usize, 1usize],
        scratch_45_0,
    );
    op_ticks[70usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_45_0, scratch_45_1, t_271, 12usize, 6usize, 18usize);
    op_ticks[71usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_271, t_49, 48usize, 72usize);
    op_ticks[72usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_271, t_272);
    op_ticks[73usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_271, t_272, t_273, 3456usize);
    op_ticks[74usize] += get_tick() - __t0;
    let __t0 = get_tick();
    pad(
        t_273,
        [1usize, 48usize, 1usize, 72usize],
        t_274,
        [1usize, 50usize, 1usize, 72usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    op_ticks[75usize] += get_tick() - __t0;
    let __t0 = get_tick();
    depthwise_conv2d(
        t_274,
        [1usize, 50usize, 1usize, 72usize],
        t_48,
        [1usize, 3usize, 3usize, 72usize],
        Some(t_47),
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_275,
        [1usize, 24usize, 1usize, 72usize],
    );
    op_ticks[76usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_275, t_276);
    op_ticks[77usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_275, t_276, t_277, 1728usize);
    op_ticks[78usize] += get_tick() - __t0;
    static mut SCRATCH_52_0: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_52_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_52_0) as *mut f32,
            1728usize,
        )
    };
    static mut SCRATCH_52_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_52_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_52_1) as *mut f32,
            2592usize,
        )
    };
    scratch_52_1.copy_from_slice(t_118);
    let __t0 = get_tick();
    im2col_padded(
        t_277,
        [1usize, 24usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_52_0,
    );
    op_ticks[79usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_52_0, scratch_52_1, t_278, 6usize, 18usize, 9usize);
    op_ticks[80usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_278, t_117, 24usize, 36usize);
    op_ticks[81usize] += get_tick() - __t0;
    static mut SCRATCH_53_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_53_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_53_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_53_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_53_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_53_1) as *mut f32,
            2592usize,
        )
    };
    scratch_53_1.copy_from_slice(t_116);
    let __t0 = get_tick();
    im2col_padded(
        t_278,
        [1usize, 24usize, 1usize, 36usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_53_0,
    );
    op_ticks[82usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_53_0, scratch_53_1, t_279, 6usize, 9usize, 18usize);
    op_ticks[83usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_279, t_46, 24usize, 72usize);
    op_ticks[84usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_279, t_280);
    op_ticks[85usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_279, t_280, t_281, 1728usize);
    op_ticks[86usize] += get_tick() - __t0;
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
    op_ticks[87usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_282, t_283);
    op_ticks[88usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_282, t_283, t_284, 1728usize);
    op_ticks[89usize] += get_tick() - __t0;
    static mut SCRATCH_59_0: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_59_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_59_0) as *mut f32,
            1728usize,
        )
    };
    static mut SCRATCH_59_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_59_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_59_1) as *mut f32,
            2592usize,
        )
    };
    scratch_59_1.copy_from_slice(t_115);
    let __t0 = get_tick();
    im2col_padded(
        t_284,
        [1usize, 24usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_59_0,
    );
    op_ticks[90usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_59_0, scratch_59_1, t_285, 6usize, 18usize, 9usize);
    op_ticks[91usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_285, t_117, 24usize, 36usize);
    op_ticks[92usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_285, t_278, t_286, 864usize);
    op_ticks[93usize] += get_tick() - __t0;
    static mut SCRATCH_61_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_61_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_61_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_61_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_61_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_61_1) as *mut f32,
            2592usize,
        )
    };
    scratch_61_1.copy_from_slice(t_114);
    let __t0 = get_tick();
    im2col_padded(
        t_286,
        [1usize, 24usize, 1usize, 36usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_61_0,
    );
    op_ticks[94usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_61_0, scratch_61_1, t_287, 6usize, 9usize, 18usize);
    op_ticks[95usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_287, t_43, 24usize, 72usize);
    op_ticks[96usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_287, t_288);
    op_ticks[97usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_287, t_288, t_289, 1728usize);
    op_ticks[98usize] += get_tick() - __t0;
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
    op_ticks[99usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_290, t_291);
    op_ticks[100usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_290, t_291, t_292, 1728usize);
    op_ticks[101usize] += get_tick() - __t0;
    static mut SCRATCH_67_0: Aligned16<1728usize> = Aligned16([0.0f32; 1728usize]);
    let scratch_67_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_67_0) as *mut f32,
            1728usize,
        )
    };
    static mut SCRATCH_67_1: Aligned16<2592usize> = Aligned16([0.0f32; 2592usize]);
    let scratch_67_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_67_1) as *mut f32,
            2592usize,
        )
    };
    scratch_67_1.copy_from_slice(t_113);
    let __t0 = get_tick();
    im2col_padded(
        t_292,
        [1usize, 24usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_67_0,
    );
    op_ticks[102usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_67_0, scratch_67_1, t_293, 6usize, 18usize, 9usize);
    op_ticks[103usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_293, t_117, 24usize, 36usize);
    op_ticks[104usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_293, t_286, t_294, 864usize);
    op_ticks[105usize] += get_tick() - __t0;
    static mut SCRATCH_69_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_69_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_69_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_69_1: Aligned16<10368usize> = Aligned16([0.0f32; 10368usize]);
    let scratch_69_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_69_1) as *mut f32,
            10368usize,
        )
    };
    scratch_69_1.copy_from_slice(t_112);
    let __t0 = get_tick();
    im2col_padded(
        t_294,
        [1usize, 24usize, 1usize, 36usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [24usize, 1usize],
        scratch_69_0,
    );
    op_ticks[106usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_69_0, scratch_69_1, t_295, 6usize, 9usize, 72usize);
    op_ticks[107usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_295, t_40, 24usize, 288usize);
    op_ticks[108usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_295, t_296);
    op_ticks[109usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_295, t_296, t_297, 6912usize);
    op_ticks[110usize] += get_tick() - __t0;
    let __t0 = get_tick();
    pad(
        t_297,
        [1usize, 24usize, 1usize, 288usize],
        t_298,
        [1usize, 26usize, 1usize, 288usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    op_ticks[111usize] += get_tick() - __t0;
    let __t0 = get_tick();
    depthwise_conv2d(
        t_298,
        [1usize, 26usize, 1usize, 288usize],
        t_39,
        [1usize, 3usize, 3usize, 288usize],
        Some(t_38),
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_299,
        [1usize, 12usize, 1usize, 288usize],
    );
    op_ticks[112usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_299, t_300);
    op_ticks[113usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_299, t_300, t_301, 3456usize);
    op_ticks[114usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_301, t_302);
    op_ticks[115usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_302, t_306);
    op_ticks[116usize] += get_tick() - __t0;
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
    op_ticks[117usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_307, t_308);
    op_ticks[118usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_307, t_308, t_309, 18usize);
    op_ticks[119usize] += get_tick() - __t0;
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
    op_ticks[120usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_310, t_311);
    op_ticks[121usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_301, t_311, t_312, 288usize);
    op_ticks[122usize] += get_tick() - __t0;
    static mut SCRATCH_84_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_84_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_84_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_84_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_84_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_84_1) as *mut f32,
            20736usize,
        )
    };
    scratch_84_1.copy_from_slice(t_107);
    let __t0 = get_tick();
    im2col_padded(
        t_312,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_84_0,
    );
    op_ticks[123usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_84_0, scratch_84_1, t_313, 3usize, 72usize, 18usize);
    op_ticks[124usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_313, t_119, 12usize, 72usize);
    op_ticks[125usize] += get_tick() - __t0;
    static mut SCRATCH_85_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_85_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_85_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_85_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_85_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_85_1) as *mut f32,
            20736usize,
        )
    };
    scratch_85_1.copy_from_slice(t_106);
    let __t0 = get_tick();
    im2col_padded(
        t_313,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_85_0,
    );
    op_ticks[126usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_85_0, scratch_85_1, t_314, 3usize, 18usize, 72usize);
    op_ticks[127usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_314, t_37, 12usize, 288usize);
    op_ticks[128usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_314, t_315);
    op_ticks[129usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_314, t_315, t_316, 3456usize);
    op_ticks[130usize] += get_tick() - __t0;
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
    op_ticks[131usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_317, t_318);
    op_ticks[132usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_317, t_318, t_319, 3456usize);
    op_ticks[133usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_319, t_320);
    op_ticks[134usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_320, t_324);
    op_ticks[135usize] += get_tick() - __t0;
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
    op_ticks[136usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_325, t_326);
    op_ticks[137usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_325, t_326, t_327, 18usize);
    op_ticks[138usize] += get_tick() - __t0;
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
    op_ticks[139usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_328, t_329);
    op_ticks[140usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_319, t_329, t_330, 288usize);
    op_ticks[141usize] += get_tick() - __t0;
    static mut SCRATCH_99_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_99_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_99_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_99_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_99_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_99_1) as *mut f32,
            20736usize,
        )
    };
    scratch_99_1.copy_from_slice(t_103);
    let __t0 = get_tick();
    im2col_padded(
        t_330,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_99_0,
    );
    op_ticks[142usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_99_0, scratch_99_1, t_331, 3usize, 72usize, 18usize);
    op_ticks[143usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_331, t_119, 12usize, 72usize);
    op_ticks[144usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_331, t_313, t_332, 864usize);
    op_ticks[145usize] += get_tick() - __t0;
    static mut SCRATCH_101_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_101_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_101_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_101_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_101_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_101_1) as *mut f32,
            20736usize,
        )
    };
    scratch_101_1.copy_from_slice(t_102);
    let __t0 = get_tick();
    im2col_padded(
        t_332,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_101_0,
    );
    op_ticks[146usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_101_0, scratch_101_1, t_333, 3usize, 18usize, 72usize);
    op_ticks[147usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_333, t_34, 12usize, 288usize);
    op_ticks[148usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_333, t_334);
    op_ticks[149usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_333, t_334, t_335, 3456usize);
    op_ticks[150usize] += get_tick() - __t0;
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
    op_ticks[151usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_336, t_337);
    op_ticks[152usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_336, t_337, t_338, 3456usize);
    op_ticks[153usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_338, t_339);
    op_ticks[154usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_339, t_343);
    op_ticks[155usize] += get_tick() - __t0;
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
    op_ticks[156usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_344, t_345);
    op_ticks[157usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_344, t_345, t_346, 18usize);
    op_ticks[158usize] += get_tick() - __t0;
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
    op_ticks[159usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_347, t_348);
    op_ticks[160usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_338, t_348, t_349, 288usize);
    op_ticks[161usize] += get_tick() - __t0;
    static mut SCRATCH_115_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_115_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_115_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_115_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_115_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_115_1) as *mut f32,
            20736usize,
        )
    };
    scratch_115_1.copy_from_slice(t_99);
    let __t0 = get_tick();
    im2col_padded(
        t_349,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_115_0,
    );
    op_ticks[162usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_115_0, scratch_115_1, t_350, 3usize, 72usize, 18usize);
    op_ticks[163usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_350, t_119, 12usize, 72usize);
    op_ticks[164usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_350, t_332, t_351, 864usize);
    op_ticks[165usize] += get_tick() - __t0;
    static mut SCRATCH_117_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_117_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_117_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_117_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_117_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_117_1) as *mut f32,
            20736usize,
        )
    };
    scratch_117_1.copy_from_slice(t_98);
    let __t0 = get_tick();
    im2col_padded(
        t_351,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_117_0,
    );
    op_ticks[166usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_117_0, scratch_117_1, t_352, 3usize, 18usize, 72usize);
    op_ticks[167usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_352, t_31, 12usize, 288usize);
    op_ticks[168usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_352, t_353);
    op_ticks[169usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_352, t_353, t_354, 3456usize);
    op_ticks[170usize] += get_tick() - __t0;
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
    op_ticks[171usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_355, t_356);
    op_ticks[172usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_355, t_356, t_357, 3456usize);
    op_ticks[173usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_357, t_358);
    op_ticks[174usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_358, t_362);
    op_ticks[175usize] += get_tick() - __t0;
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
    op_ticks[176usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_363, t_364);
    op_ticks[177usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_363, t_364, t_365, 18usize);
    op_ticks[178usize] += get_tick() - __t0;
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
    op_ticks[179usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_366, t_367);
    op_ticks[180usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_357, t_367, t_368, 288usize);
    op_ticks[181usize] += get_tick() - __t0;
    static mut SCRATCH_131_0: Aligned16<3456usize> = Aligned16([0.0f32; 3456usize]);
    let scratch_131_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_131_0) as *mut f32,
            3456usize,
        )
    };
    static mut SCRATCH_131_1: Aligned16<20736usize> = Aligned16([0.0f32; 20736usize]);
    let scratch_131_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_131_1) as *mut f32,
            20736usize,
        )
    };
    scratch_131_1.copy_from_slice(t_95);
    let __t0 = get_tick();
    im2col_padded(
        t_368,
        [1usize, 12usize, 1usize, 288usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_131_0,
    );
    op_ticks[182usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_131_0, scratch_131_1, t_369, 3usize, 72usize, 18usize);
    op_ticks[183usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_369, t_119, 12usize, 72usize);
    op_ticks[184usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_369, t_351, t_370, 864usize);
    op_ticks[185usize] += get_tick() - __t0;
    static mut SCRATCH_133_0: Aligned16<864usize> = Aligned16([0.0f32; 864usize]);
    let scratch_133_0 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_133_0) as *mut f32,
            864usize,
        )
    };
    static mut SCRATCH_133_1: Aligned16<62208usize> = Aligned16([0.0f32; 62208usize]);
    let scratch_133_1 = unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(SCRATCH_133_1) as *mut f32,
            62208usize,
        )
    };
    scratch_133_1.copy_from_slice(t_94);
    let __t0 = get_tick();
    im2col_padded(
        t_370,
        [1usize, 12usize, 1usize, 72usize],
        [1usize, 1usize],
        [1usize, 1usize],
        [0usize, 0usize, 0usize, 0usize],
        [12usize, 1usize],
        scratch_133_0,
    );
    op_ticks[186usize] += get_tick() - __t0;
    let __t0 = get_tick();
    matmul_bt_tiled(scratch_133_0, scratch_133_1, t_371, 3usize, 18usize, 216usize);
    op_ticks[187usize] += get_tick() - __t0;
    let __t0 = get_tick();
    bias_add(t_371, t_28, 12usize, 864usize);
    op_ticks[188usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_371, t_372);
    op_ticks[189usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_371, t_372, t_373, 10368usize);
    op_ticks[190usize] += get_tick() - __t0;
    let __t0 = get_tick();
    pad(
        t_373,
        [1usize, 12usize, 1usize, 864usize],
        t_374,
        [1usize, 14usize, 1usize, 864usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    op_ticks[191usize] += get_tick() - __t0;
    let __t0 = get_tick();
    depthwise_conv2d(
        t_374,
        [1usize, 14usize, 1usize, 864usize],
        t_27,
        [1usize, 3usize, 3usize, 864usize],
        Some(t_26),
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_375,
        [1usize, 6usize, 1usize, 864usize],
    );
    op_ticks[192usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_375, t_376);
    op_ticks[193usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_375, t_376, t_377, 5184usize);
    op_ticks[194usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_377, t_378);
    op_ticks[195usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_378, t_382);
    op_ticks[196usize] += get_tick() - __t0;
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
    op_ticks[197usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_383, t_384);
    op_ticks[198usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_383, t_384, t_385, 27usize);
    op_ticks[199usize] += get_tick() - __t0;
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
    op_ticks[200usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_386, t_387);
    op_ticks[201usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_377, t_387, t_388, 864usize);
    op_ticks[202usize] += get_tick() - __t0;
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
    op_ticks[203usize] += get_tick() - __t0;
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
    op_ticks[204usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_390, t_391);
    op_ticks[205usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_390, t_391, t_392, 5184usize);
    op_ticks[206usize] += get_tick() - __t0;
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
    op_ticks[207usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_393, t_394);
    op_ticks[208usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_393, t_394, t_395, 5184usize);
    op_ticks[209usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_395, t_396);
    op_ticks[210usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_396, t_400);
    op_ticks[211usize] += get_tick() - __t0;
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
    op_ticks[212usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_401, t_402);
    op_ticks[213usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_401, t_402, t_403, 27usize);
    op_ticks[214usize] += get_tick() - __t0;
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
    op_ticks[215usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_404, t_405);
    op_ticks[216usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_395, t_405, t_406, 864usize);
    op_ticks[217usize] += get_tick() - __t0;
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
    op_ticks[218usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_407, t_389, t_408, 648usize);
    op_ticks[219usize] += get_tick() - __t0;
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
    op_ticks[220usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_409, t_410);
    op_ticks[221usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_409, t_410, t_411, 5184usize);
    op_ticks[222usize] += get_tick() - __t0;
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
    op_ticks[223usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_412, t_413);
    op_ticks[224usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_412, t_413, t_414, 5184usize);
    op_ticks[225usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_414, t_415);
    op_ticks[226usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_415, t_419);
    op_ticks[227usize] += get_tick() - __t0;
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
    op_ticks[228usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_420, t_421);
    op_ticks[229usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_420, t_421, t_422, 27usize);
    op_ticks[230usize] += get_tick() - __t0;
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
    op_ticks[231usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_423, t_424);
    op_ticks[232usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_414, t_424, t_425, 864usize);
    op_ticks[233usize] += get_tick() - __t0;
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
    op_ticks[234usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_426, t_408, t_427, 648usize);
    op_ticks[235usize] += get_tick() - __t0;
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
    op_ticks[236usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_428, t_429);
    op_ticks[237usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_428, t_429, t_430, 5184usize);
    op_ticks[238usize] += get_tick() - __t0;
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
    op_ticks[239usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_431, t_432);
    op_ticks[240usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_431, t_432, t_433, 5184usize);
    op_ticks[241usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_433, t_434);
    op_ticks[242usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_434, t_438);
    op_ticks[243usize] += get_tick() - __t0;
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
    op_ticks[244usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_439, t_440);
    op_ticks[245usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_439, t_440, t_441, 27usize);
    op_ticks[246usize] += get_tick() - __t0;
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
    op_ticks[247usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_442, t_443);
    op_ticks[248usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_433, t_443, t_444, 864usize);
    op_ticks[249usize] += get_tick() - __t0;
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
    op_ticks[250usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_445, t_427, t_446, 648usize);
    op_ticks[251usize] += get_tick() - __t0;
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
    op_ticks[252usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_447, t_448);
    op_ticks[253usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_447, t_448, t_449, 5184usize);
    op_ticks[254usize] += get_tick() - __t0;
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
    op_ticks[255usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_450, t_451);
    op_ticks[256usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_450, t_451, t_452, 5184usize);
    op_ticks[257usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_452, t_453);
    op_ticks[258usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_453, t_457);
    op_ticks[259usize] += get_tick() - __t0;
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
    op_ticks[260usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_458, t_459);
    op_ticks[261usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_458, t_459, t_460, 27usize);
    op_ticks[262usize] += get_tick() - __t0;
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
    op_ticks[263usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_461, t_462);
    op_ticks[264usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_452, t_462, t_463, 864usize);
    op_ticks[265usize] += get_tick() - __t0;
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
    op_ticks[266usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_464, t_446, t_465, 648usize);
    op_ticks[267usize] += get_tick() - __t0;
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
    op_ticks[268usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_466, t_467);
    op_ticks[269usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_466, t_467, t_468, 9216usize);
    op_ticks[270usize] += get_tick() - __t0;
    let __t0 = get_tick();
    pad(
        t_468,
        [1usize, 6usize, 1usize, 1536usize],
        t_469,
        [1usize, 8usize, 1usize, 1536usize],
        [[0usize, 0usize], [1usize, 1usize], [1usize, 1usize], [0usize, 0usize]],
    );
    op_ticks[271usize] += get_tick() - __t0;
    let __t0 = get_tick();
    depthwise_conv2d(
        t_469,
        [1usize, 8usize, 1usize, 1536usize],
        t_12,
        [1usize, 3usize, 3usize, 1536usize],
        Some(t_11),
        [2usize, 2usize],
        [0usize, 0usize, 0usize, 0usize],
        t_470,
        [1usize, 3usize, 1usize, 1536usize],
    );
    op_ticks[272usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_470, t_471);
    op_ticks[273usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_470, t_471, t_472, 4608usize);
    op_ticks[274usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_472, t_473);
    op_ticks[275usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_473, t_477);
    op_ticks[276usize] += get_tick() - __t0;
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
    op_ticks[277usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_478, t_479);
    op_ticks[278usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_478, t_479, t_480, 48usize);
    op_ticks[279usize] += get_tick() - __t0;
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
    op_ticks[280usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_481, t_482);
    op_ticks[281usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_472, t_482, t_483, 1536usize);
    op_ticks[282usize] += get_tick() - __t0;
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
    op_ticks[283usize] += get_tick() - __t0;
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
    op_ticks[284usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_485, t_486);
    op_ticks[285usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_485, t_486, t_487, 4608usize);
    op_ticks[286usize] += get_tick() - __t0;
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
    op_ticks[287usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_488, t_489);
    op_ticks[288usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_488, t_489, t_490, 4608usize);
    op_ticks[289usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_490, t_491);
    op_ticks[290usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_491, t_495);
    op_ticks[291usize] += get_tick() - __t0;
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
    op_ticks[292usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_496, t_497);
    op_ticks[293usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_496, t_497, t_498, 48usize);
    op_ticks[294usize] += get_tick() - __t0;
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
    op_ticks[295usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_499, t_500);
    op_ticks[296usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_490, t_500, t_501, 1536usize);
    op_ticks[297usize] += get_tick() - __t0;
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
    op_ticks[298usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_502, t_484, t_503, 576usize);
    op_ticks[299usize] += get_tick() - __t0;
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
    op_ticks[300usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_504, t_505);
    op_ticks[301usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_504, t_505, t_506, 4608usize);
    op_ticks[302usize] += get_tick() - __t0;
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
    op_ticks[303usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_507, t_508);
    op_ticks[304usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_507, t_508, t_509, 4608usize);
    op_ticks[305usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_509, t_510);
    op_ticks[306usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_510, t_514);
    op_ticks[307usize] += get_tick() - __t0;
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
    op_ticks[308usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_515, t_516);
    op_ticks[309usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_515, t_516, t_517, 48usize);
    op_ticks[310usize] += get_tick() - __t0;
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
    op_ticks[311usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_518, t_519);
    op_ticks[312usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_509, t_519, t_520, 1536usize);
    op_ticks[313usize] += get_tick() - __t0;
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
    op_ticks[314usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_521, t_503, t_522, 576usize);
    op_ticks[315usize] += get_tick() - __t0;
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
    op_ticks[316usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_523, t_524);
    op_ticks[317usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_523, t_524, t_525, 4608usize);
    op_ticks[318usize] += get_tick() - __t0;
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
    op_ticks[319usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_526, t_527);
    op_ticks[320usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_526, t_527, t_528, 4608usize);
    op_ticks[321usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_528, t_529);
    op_ticks[322usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reshape(t_529, t_533);
    op_ticks[323usize] += get_tick() - __t0;
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
    op_ticks[324usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_534, t_535);
    op_ticks[325usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_534, t_535, t_536, 48usize);
    op_ticks[326usize] += get_tick() - __t0;
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
    op_ticks[327usize] += get_tick() - __t0;
    let __t0 = get_tick();
    unary_logistic(t_537, t_538);
    op_ticks[328usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_528, t_538, t_539, 1536usize);
    op_ticks[329usize] += get_tick() - __t0;
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
    op_ticks[330usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_540, t_522, t_541, 576usize);
    op_ticks[331usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_mul(t_541, t_161, t_542, 192usize);
    op_ticks[332usize] += get_tick() - __t0;
    let __t0 = get_tick();
    binary_add(t_542, t_160, t_543, 192usize);
    op_ticks[333usize] += get_tick() - __t0;
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
    op_ticks[334usize] += get_tick() - __t0;
    let __t0 = get_tick();
    reduce_mean_hw(t_544, t_545);
    op_ticks[335usize] += get_tick() - __t0;
    let __t0 = get_tick();
    fully_connected(t_545, 1024usize, t_164, Some(t_142), &mut t_546, 6522usize);
    op_ticks[336usize] += get_tick() - __t0;
    t_546
}
pub const NUM_OPS: usize = 337usize;
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
    "reshape",
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
    "reshape",
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
const WEIGHT_BYTES: usize = 51770016usize;
const TENSOR_DATA_FLOATS: usize = 12942504usize;
static mut WEIGHT_PTR: *const u8 = core::ptr::null();
const T_1_OFFSET: usize = 12930528usize;
const T_1_LEN: usize = 1024usize;
const T_2_OFFSET: usize = 12928951usize;
const T_2_LEN: usize = 1536usize;
const T_3_OFFSET: usize = 12915086usize;
const T_3_LEN: usize = 13824usize;
const T_4_OFFSET: usize = 12913513usize;
const T_4_LEN: usize = 1536usize;
const T_5_OFFSET: usize = 12911939usize;
const T_5_LEN: usize = 1536usize;
const T_6_OFFSET: usize = 12898074usize;
const T_6_LEN: usize = 13824usize;
const T_7_OFFSET: usize = 12896501usize;
const T_7_LEN: usize = 1536usize;
const T_8_OFFSET: usize = 12894927usize;
const T_8_LEN: usize = 1536usize;
const T_9_OFFSET: usize = 12881062usize;
const T_9_LEN: usize = 13824usize;
const T_10_OFFSET: usize = 12879489usize;
const T_10_LEN: usize = 1536usize;
const T_11_OFFSET: usize = 12877915usize;
const T_11_LEN: usize = 1536usize;
const T_12_OFFSET: usize = 12864050usize;
const T_12_LEN: usize = 13824usize;
const T_13_OFFSET: usize = 12862485usize;
const T_13_LEN: usize = 1536usize;
const T_14_OFFSET: usize = 12861583usize;
const T_14_LEN: usize = 864usize;
const T_15_OFFSET: usize = 12853766usize;
const T_15_LEN: usize = 7776usize;
const T_16_OFFSET: usize = 12852865usize;
const T_16_LEN: usize = 864usize;
const T_17_OFFSET: usize = 12851963usize;
const T_17_LEN: usize = 864usize;
const T_18_OFFSET: usize = 12844146usize;
const T_18_LEN: usize = 7776usize;
const T_19_OFFSET: usize = 12843245usize;
const T_19_LEN: usize = 864usize;
const T_20_OFFSET: usize = 12842343usize;
const T_20_LEN: usize = 864usize;
const T_21_OFFSET: usize = 12834526usize;
const T_21_LEN: usize = 7776usize;
const T_22_OFFSET: usize = 12833625usize;
const T_22_LEN: usize = 864usize;
const T_23_OFFSET: usize = 12832723usize;
const T_23_LEN: usize = 864usize;
const T_24_OFFSET: usize = 12824906usize;
const T_24_LEN: usize = 7776usize;
const T_25_OFFSET: usize = 12824005usize;
const T_25_LEN: usize = 864usize;
const T_26_OFFSET: usize = 12823103usize;
const T_26_LEN: usize = 864usize;
const T_27_OFFSET: usize = 12815286usize;
const T_27_LEN: usize = 7776usize;
const T_28_OFFSET: usize = 12814393usize;
const T_28_LEN: usize = 864usize;
const T_29_OFFSET: usize = 12814067usize;
const T_29_LEN: usize = 288usize;
const T_30_OFFSET: usize = 12811434usize;
const T_30_LEN: usize = 2592usize;
const T_31_OFFSET: usize = 12811109usize;
const T_31_LEN: usize = 288usize;
const T_32_OFFSET: usize = 12810783usize;
const T_32_LEN: usize = 288usize;
const T_33_OFFSET: usize = 12808150usize;
const T_33_LEN: usize = 2592usize;
const T_34_OFFSET: usize = 12807825usize;
const T_34_LEN: usize = 288usize;
const T_35_OFFSET: usize = 12807499usize;
const T_35_LEN: usize = 288usize;
const T_36_OFFSET: usize = 12804866usize;
const T_36_LEN: usize = 2592usize;
const T_37_OFFSET: usize = 12804541usize;
const T_37_LEN: usize = 288usize;
const T_38_OFFSET: usize = 12804215usize;
const T_38_LEN: usize = 288usize;
const T_39_OFFSET: usize = 12801582usize;
const T_39_LEN: usize = 2592usize;
const T_40_OFFSET: usize = 12801265usize;
const T_40_LEN: usize = 288usize;
const T_41_OFFSET: usize = 12801155usize;
const T_41_LEN: usize = 72usize;
const T_42_OFFSET: usize = 12800466usize;
const T_42_LEN: usize = 648usize;
const T_43_OFFSET: usize = 12800357usize;
const T_43_LEN: usize = 72usize;
const T_44_OFFSET: usize = 12800247usize;
const T_44_LEN: usize = 72usize;
const T_45_OFFSET: usize = 12799558usize;
const T_45_LEN: usize = 648usize;
const T_46_OFFSET: usize = 12799449usize;
const T_46_LEN: usize = 72usize;
const T_47_OFFSET: usize = 12799339usize;
const T_47_LEN: usize = 72usize;
const T_48_OFFSET: usize = 12798650usize;
const T_48_LEN: usize = 648usize;
const T_49_OFFSET: usize = 12798549usize;
const T_49_LEN: usize = 72usize;
const T_50_OFFSET: usize = 12798482usize;
const T_50_LEN: usize = 24usize;
const T_51_OFFSET: usize = 12798433usize;
const T_51_LEN: usize = 24usize;
const T_52_OFFSET: usize = 11028939usize;
const T_52_LEN: usize = 1769472usize;
const T_53_OFFSET: usize = 10734004usize;
const T_53_LEN: usize = 294912usize;
const T_54_OFFSET: usize = 10660252usize;
const T_54_LEN: usize = 73728usize;
const T_55_OFFSET: usize = 10586500usize;
const T_55_LEN: usize = 73728usize;
const T_56_OFFSET: usize = 10291565usize;
const T_56_LEN: usize = 294912usize;
const T_57_OFFSET: usize = 9996630usize;
const T_57_LEN: usize = 294912usize;
const T_58_OFFSET: usize = 9922878usize;
const T_58_LEN: usize = 73728usize;
const T_59_OFFSET: usize = 9849126usize;
const T_59_LEN: usize = 73728usize;
const T_60_OFFSET: usize = 9554191usize;
const T_60_LEN: usize = 294912usize;
const T_61_OFFSET: usize = 9259256usize;
const T_61_LEN: usize = 294912usize;
const T_62_OFFSET: usize = 9185504usize;
const T_62_LEN: usize = 73728usize;
const T_63_OFFSET: usize = 9111752usize;
const T_63_LEN: usize = 73728usize;
const T_64_OFFSET: usize = 8816817usize;
const T_64_LEN: usize = 294912usize;
const T_65_OFFSET: usize = 8816605usize;
const T_65_LEN: usize = 192usize;
const T_66_OFFSET: usize = 8521670usize;
const T_66_LEN: usize = 294912usize;
const T_67_OFFSET: usize = 8447918usize;
const T_67_LEN: usize = 73728usize;
const T_68_OFFSET: usize = 8447849usize;
const T_68_LEN: usize = 48usize;
const T_69_OFFSET: usize = 8374097usize;
const T_69_LEN: usize = 73728usize;
const T_70_OFFSET: usize = 8372541usize;
const T_70_LEN: usize = 1536usize;
const T_71_OFFSET: usize = 8206630usize;
const T_71_LEN: usize = 165888usize;
const T_72_OFFSET: usize = 8113295usize;
const T_72_LEN: usize = 93312usize;
const T_73_OFFSET: usize = 8089943usize;
const T_73_LEN: usize = 23328usize;
const T_74_OFFSET: usize = 8066591usize;
const T_74_LEN: usize = 23328usize;
const T_75_OFFSET: usize = 7973256usize;
const T_75_LEN: usize = 93312usize;
const T_76_OFFSET: usize = 7879921usize;
const T_76_LEN: usize = 93312usize;
const T_77_OFFSET: usize = 7856569usize;
const T_77_LEN: usize = 23328usize;
const T_78_OFFSET: usize = 7833217usize;
const T_78_LEN: usize = 23328usize;
const T_79_OFFSET: usize = 7739882usize;
const T_79_LEN: usize = 93312usize;
const T_80_OFFSET: usize = 7646547usize;
const T_80_LEN: usize = 93312usize;
const T_81_OFFSET: usize = 7623195usize;
const T_81_LEN: usize = 23328usize;
const T_82_OFFSET: usize = 7599843usize;
const T_82_LEN: usize = 23328usize;
const T_83_OFFSET: usize = 7506508usize;
const T_83_LEN: usize = 93312usize;
const T_84_OFFSET: usize = 7413173usize;
const T_84_LEN: usize = 93312usize;
const T_85_OFFSET: usize = 7389821usize;
const T_85_LEN: usize = 23328usize;
const T_86_OFFSET: usize = 7366469usize;
const T_86_LEN: usize = 23328usize;
const T_87_OFFSET: usize = 7273134usize;
const T_87_LEN: usize = 93312usize;
const T_88_OFFSET: usize = 7273006usize;
const T_88_LEN: usize = 108usize;
const T_89_OFFSET: usize = 7179671usize;
const T_89_LEN: usize = 93312usize;
const T_90_OFFSET: usize = 7156319usize;
const T_90_LEN: usize = 23328usize;
const T_91_OFFSET: usize = 7156271usize;
const T_91_LEN: usize = 27usize;
const T_92_OFFSET: usize = 7132919usize;
const T_92_LEN: usize = 23328usize;
const T_93_OFFSET: usize = 7132035usize;
const T_93_LEN: usize = 864usize;
const T_94_OFFSET: usize = 7069804usize;
const T_94_LEN: usize = 62208usize;
const T_95_OFFSET: usize = 7049045usize;
const T_95_LEN: usize = 20736usize;
const T_96_OFFSET: usize = 7043837usize;
const T_96_LEN: usize = 5184usize;
const T_97_OFFSET: usize = 7038629usize;
const T_97_LEN: usize = 5184usize;
const T_98_OFFSET: usize = 7017870usize;
const T_98_LEN: usize = 20736usize;
const T_99_OFFSET: usize = 6997111usize;
const T_99_LEN: usize = 20736usize;
const T_100_OFFSET: usize = 6991903usize;
const T_100_LEN: usize = 5184usize;
const T_101_OFFSET: usize = 6986695usize;
const T_101_LEN: usize = 5184usize;
const T_102_OFFSET: usize = 6965936usize;
const T_102_LEN: usize = 20736usize;
const T_103_OFFSET: usize = 6945177usize;
const T_103_LEN: usize = 20736usize;
const T_104_OFFSET: usize = 6939969usize;
const T_104_LEN: usize = 5184usize;
const T_105_OFFSET: usize = 6934761usize;
const T_105_LEN: usize = 5184usize;
const T_106_OFFSET: usize = 6914002usize;
const T_106_LEN: usize = 20736usize;
const T_107_OFFSET: usize = 6893243usize;
const T_107_LEN: usize = 20736usize;
const T_108_OFFSET: usize = 6888035usize;
const T_108_LEN: usize = 5184usize;
const T_109_OFFSET: usize = 6887996usize;
const T_109_LEN: usize = 18usize;
const T_110_OFFSET: usize = 6882788usize;
const T_110_LEN: usize = 5184usize;
const T_111_OFFSET: usize = 6882480usize;
const T_111_LEN: usize = 288usize;
const T_112_OFFSET: usize = 6872089usize;
const T_112_LEN: usize = 10368usize;
const T_113_OFFSET: usize = 6869474usize;
const T_113_LEN: usize = 2592usize;
const T_114_OFFSET: usize = 6866859usize;
const T_114_LEN: usize = 2592usize;
const T_115_OFFSET: usize = 6864244usize;
const T_115_LEN: usize = 2592usize;
const T_116_OFFSET: usize = 6861629usize;
const T_116_LEN: usize = 2592usize;
const T_117_OFFSET: usize = 6861573usize;
const T_117_LEN: usize = 36usize;
const T_118_OFFSET: usize = 6858958usize;
const T_118_LEN: usize = 2592usize;
const T_119_OFFSET: usize = 6858866usize;
const T_119_LEN: usize = 72usize;
const T_120_OFFSET: usize = 6857115usize;
const T_120_LEN: usize = 1728usize;
const T_121_OFFSET: usize = 6855941usize;
const T_121_LEN: usize = 1152usize;
const T_122_OFFSET: usize = 6854385usize;
const T_122_LEN: usize = 1536usize;
const T_126_OFFSET: usize = 6853271usize;
const T_126_LEN: usize = 1024usize;
const T_129_OFFSET: usize = 6851026usize;
const T_129_LEN: usize = 2048usize;
const T_133_OFFSET: usize = 6849906usize;
const T_133_LEN: usize = 1usize;
const T_134_OFFSET: usize = 6849881usize;
const T_134_LEN: usize = 2usize;
const T_135_OFFSET: usize = 6849850usize;
const T_135_LEN: usize = 8usize;
const T_136_OFFSET: usize = 6849821usize;
const T_136_LEN: usize = 2usize;
const T_142_OFFSET: usize = 6843148usize;
const T_142_LEN: usize = 6522usize;
const T_144_OFFSET: usize = 6843109usize;
const T_144_LEN: usize = 1usize;
const T_145_OFFSET: usize = 6843087usize;
const T_145_LEN: usize = 1usize;
const T_146_OFFSET: usize = 6843069usize;
const T_146_LEN: usize = 1usize;
const T_150_OFFSET: usize = 6842982usize;
const T_150_LEN: usize = 1usize;
const T_153_OFFSET: usize = 6842912usize;
const T_153_LEN: usize = 2usize;
const T_156_OFFSET: usize = 6842845usize;
const T_156_LEN: usize = 3usize;
const T_160_OFFSET: usize = 6842562usize;
const T_160_LEN: usize = 192usize;
const T_161_OFFSET: usize = 6842347usize;
const T_161_LEN: usize = 192usize;
const T_162_OFFSET: usize = 6842322usize;
const T_162_LEN: usize = 2usize;
const T_163_OFFSET: usize = 6842297usize;
const T_163_LEN: usize = 2usize;
const T_164_OFFSET: usize = 163748usize;
const T_164_LEN: usize = 6678528usize;
const T_165_OFFSET: usize = 163729usize;
const T_165_LEN: usize = 1usize;
const T_166_OFFSET: usize = 65307usize;
const T_166_LEN: usize = 98400usize;
const T_167_OFFSET: usize = 65288usize;
const T_167_LEN: usize = 1usize;
const T_168_OFFSET: usize = 16018usize;
const T_168_LEN: usize = 49248usize;
const T_185_OFFSET: usize = 12931613usize;
const T_185_LEN: usize = 2usize;
const T_199_OFFSET: usize = 12933162usize;
const T_199_LEN: usize = 1024usize;
const T_226_OFFSET: usize = 12934204usize;
const T_226_LEN: usize = 2usize;
const T_240_OFFSET: usize = 12935753usize;
const T_240_LEN: usize = 511usize;
const T_547_OFFSET: usize = 12936368usize;
const T_547_LEN: usize = 2usize;
const T_548_OFFSET: usize = 12936370usize;
const T_548_LEN: usize = 4usize;
const T_549_OFFSET: usize = 12936374usize;
const T_549_LEN: usize = 8usize;
const T_550_OFFSET: usize = 12936382usize;
const T_550_LEN: usize = 16usize;
const T_551_OFFSET: usize = 12936398usize;
const T_551_LEN: usize = 32usize;
const T_552_OFFSET: usize = 12936430usize;
const T_552_LEN: usize = 64usize;
const T_553_OFFSET: usize = 12936494usize;
const T_553_LEN: usize = 128usize;
const T_554_OFFSET: usize = 12936622usize;
const T_554_LEN: usize = 256usize;
const T_555_OFFSET: usize = 12936878usize;
const T_555_LEN: usize = 512usize;
const T_556_OFFSET: usize = 12937390usize;
const T_556_LEN: usize = 1024usize;
const T_557_OFFSET: usize = 12938414usize;
const T_557_LEN: usize = 2046usize;
const T_558_OFFSET: usize = 12940460usize;
const T_558_LEN: usize = 2usize;
const T_559_OFFSET: usize = 12940462usize;
const T_559_LEN: usize = 4usize;
const T_560_OFFSET: usize = 12940466usize;
const T_560_LEN: usize = 8usize;
const T_561_OFFSET: usize = 12940474usize;
const T_561_LEN: usize = 16usize;
const T_562_OFFSET: usize = 12940490usize;
const T_562_LEN: usize = 32usize;
const T_563_OFFSET: usize = 12940522usize;
const T_563_LEN: usize = 64usize;
const T_564_OFFSET: usize = 12940586usize;
const T_564_LEN: usize = 128usize;
const T_565_OFFSET: usize = 12940714usize;
const T_565_LEN: usize = 256usize;
const T_566_OFFSET: usize = 12940970usize;
const T_566_LEN: usize = 512usize;
const T_567_OFFSET: usize = 12941482usize;
const T_567_LEN: usize = 1022usize;
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
