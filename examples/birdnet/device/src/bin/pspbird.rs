//! PSPBird: the birding app (design/2026-08-24_pspbird-app.md).
//!
//! Screen flow:
//!   1. region menu   UP/DOWN pick a pruned classifier, X confirms
//!   2. mode          X single-step, CIRCLE live
//!   single-step:     X starts recording, X (or the buffer cap) stops;
//!                    SQUARE plays it back, CIRCLE classifies, TRIANGLE
//!                    records again, SELECT returns to the region menu
//!   live:            the mic streams into a ring buffer while a lower-
//!                    priority VFPU thread classifies the newest 3 s window
//!                    whenever it is free; results replace the page as they
//!                    land. X leaves.
//!
//! Both modes end in `results_page`, the one place a dense score vector
//! becomes a screen (image mode will hang off the same point).
//!
//! The model pipeline is `birdnet::psp_bird`; selecting a region loads its
//! classifier blob into the generated slots, and CIRCLE classifies the
//! first 3 s of the capture.
//!
//! The mic delivers 44.1 kHz mono (`AudioInputFrequency` has no 48 kHz),
//! so the capture is resampled to BirdNET's 48 kHz at classify time, not
//! here. Playback uses a plain hardware channel: those run at 44.1 kHz
//! natively, so no rate conversion is involved. (The SRC channel is a
//! single shared resource and reserving it under psplink fails with
//! 0x80268002, busy.)

#![no_std]
#![no_main]

mod app {
    use birdnet::imfile::{self, device::Pack};
    use birdnet::psp_bird::{self, INPUT_SAMPLES, OUTPUT_CLASSES, REGIONS, TIMED_TICKS};
    use core::ffi::c_void;
    use core::fmt::Write;
    use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
    use psp::sys::{
        sceAudioChRelease, sceAudioChReserve, sceAudioInputBlocking, sceAudioInputInit,
        sceAudioOutputPannedBlocking, sceCtrlPeekBufferPositive, sceCtrlSetSamplingMode,
        sceKernelCreateThread, sceKernelDelayThread, sceKernelDeleteThread,
        sceKernelGetSystemTimeWide, sceKernelStartThread, sceKernelWaitThreadEnd, AudioFormat,
        AudioInputFrequency, CtrlButtons, CtrlMode, SceCtrlData, ThreadAttributes,
        AUDIO_NEXT_CHANNEL, AUDIO_VOLUME_MAX,
    };

    // No watchdog: the user, not a benchmark, decides how long this runs
    // (it ships as an EBOOT). Under psplink the host runner's idle timeout
    // is the only bound.
    psp_rt::module!("pspbird", 1, 0, timeout_secs = 0);

    const SAMPLE_RATE: u32 = 44_100;
    /// One mic read. 1024 samples = 23 ms, short enough that the stop
    /// button feels immediate.
    const CHUNK: usize = 1024;
    const MAX_SECONDS: usize = 15;
    const MAX_SAMPLES: usize = SAMPLE_RATE as usize * MAX_SECONDS;

    /// Static, not stack: the worker thread has 256 KiB and this is 1.3 MB.
    static mut CAPTURE: [i16; MAX_SAMPLES] = [0; MAX_SAMPLES];
    static mut CAPTURED: usize = 0;
    /// The model's 3 s @ 48 kHz window, resampled from the capture.
    static mut MODEL_INPUT: [f32; INPUT_SAMPLES] = [0.0; INPUT_SAMPLES];
    static mut SCORES: [f32; OUTPUT_CLASSES] = [0.0; OUTPUT_CLASSES];
    /// Class indices sorted by descending score, for the results page.
    static mut ORDER: [u16; OUTPUT_CLASSES] = [0; OUTPUT_CLASSES];

    // ---- live mode: capture ring + inference thread -------------------
    /// 3 s of model input plus room for one inference's worth of new audio
    /// to land without overwriting the window being read.
    const RING_LEN: usize = SAMPLE_RATE as usize * 8;
    const WINDOW: usize = SAMPLE_RATE as usize * 3;
    static mut RING: [i16; RING_LEN] = [0; RING_LEN];
    /// Samples ever written; the writer's ring index is this mod RING_LEN.
    static RING_WRITTEN: AtomicUsize = AtomicUsize::new(0);
    static LIVE_STOP: AtomicBool = AtomicBool::new(false);
    /// Bumped by the inference thread after each publish into LIVE_SCORES.
    static LIVE_SEQ: AtomicU32 = AtomicU32::new(0);
    static LIVE_LAST_MS: AtomicU32 = AtomicU32::new(0);
    static mut LIVE_SCORES: [f32; OUTPUT_CLASSES] = [0.0; OUTPUT_CLASSES];
    /// Below the UI thread (32): the screen and mic reads preempt inference.
    const INFER_PRIORITY: i32 = 40;

    // ---- species images -----------------------------------------------
    // `blobs/birds.img` (PBIM, see birdnet::imfile), opened once; the
    // selected region's class -> image map is copied into IMAGE_MAP so a
    // results page goes from class index to pixels with one seek+read
    // (~2 ms) and one blit (~1 ms) per thumbnail.

    /// Beside the module first (Memory Stick install), then the runner's
    /// mount -- same order as the classifier blobs.
    const PACK_PATHS: [&str; 2] = ["blobs/birds.img", "host0:/blobs/birds.img"];
    /// Grid: `GRID_COLS` x `GRID_ROWS` thumbnails at the right edge, best
    /// first, row-major.
    const GRID_COLS: usize = 2;
    const GRID_ROWS: usize = 3;
    const GRID_TOP: usize = 12;
    const GRID_GAP: usize = 4;
    /// Largest thumbnail this build buffers (96x96 RGB565).
    const IMAGE_CAP: usize = 96 * 96 * 2;
    /// Region maps: n_regions x (name + u32[OUTPUT_CLASSES]).
    const REGIONS_CAP: usize = 32 * 1024;

    static mut PACK: Option<Pack> = None;
    static mut IMAGE_BUF: [u8; IMAGE_CAP] = [0; IMAGE_CAP];
    static mut REGIONS_BUF: [u8; REGIONS_CAP] = [0; REGIONS_CAP];
    static mut IMAGE_MAP: [u32; OUTPUT_CLASSES] = [imfile::NO_IMAGE; OUTPUT_CLASSES];
    /// Width of the text area the grid leaves free, in columns.
    static mut TEXT_COLS: usize = 80;

    fn image_buf(len: usize) -> &'static mut [u8] {
        assert!(len <= IMAGE_CAP);
        unsafe { core::slice::from_raw_parts_mut(core::ptr::addr_of_mut!(IMAGE_BUF) as *mut u8, len) }
    }

    /// Open the pack (once, at startup). Without one the app just shows
    /// text.
    fn images_init() {
        match Pack::open(&PACK_PATHS) {
            Ok(pack) if pack.header.image_len() <= IMAGE_CAP => {
                let w = pack.header.w as usize;
                let grid_w = GRID_COLS * (w + GRID_GAP) + GRID_GAP;
                unsafe {
                    TEXT_COLS = (imfile::device::SCREEN_W - grid_w) / 6;
                }
                psp_rt::dprintln!(
                    "pspbird: images: {} at {}x{}, {} regions",
                    pack.header.n, pack.header.w, pack.header.h, pack.header.n_regions
                );
                unsafe { PACK = Some(pack) };
            }
            Ok(pack) => {
                psp_rt::dprintln!("pspbird: images: {}x{} too large for this build", pack.header.w, pack.header.h);
                pack.close();
            }
            Err(e) => psp_rt::dprintln!("pspbird: no species images ({:?})", e),
        }
    }

    fn pack() -> Option<&'static Pack> {
        unsafe { (*core::ptr::addr_of!(PACK)).as_ref() }
    }

    /// Load the class -> image map for `REGIONS[region]`.
    fn images_select_region(region: usize) {
        let map = unsafe { &mut *core::ptr::addr_of_mut!(IMAGE_MAP) };
        map.fill(imfile::NO_IMAGE);
        let Some(pack) = pack() else { return };
        let buf = unsafe { core::slice::from_raw_parts_mut(core::ptr::addr_of_mut!(REGIONS_BUF) as *mut u8, REGIONS_CAP) };
        let name = REGIONS[region].0;
        let found = match pack.read_regions(buf) {
            Ok(regions) => regions.find(name),
            Err(e) => {
                psp_rt::dprintln!("pspbird: images: region maps: {:?}", e);
                None
            }
        };
        match found {
            Some(r) if r.n_classes() == OUTPUT_CLASSES => {
                let mut have = 0;
                for (c, slot) in map.iter_mut().enumerate() {
                    *slot = r.raw(c);
                    have += (*slot != imfile::NO_IMAGE) as usize;
                }
                psp_rt::dprintln!("pspbird: images: [{}] {}/{} classes mapped", name, have, OUTPUT_CLASSES);
            }
            Some(r) => psp_rt::dprintln!(
                "pspbird: images: [{}] map has {} classes, build has {}",
                name, r.n_classes(), OUTPUT_CLASSES
            ),
            None => psp_rt::dprintln!("pspbird: images: no map for [{}]", name),
        }
    }

    /// Draw the top `GRID_COLS * GRID_ROWS` classes of `order` as a grid at
    /// the right edge. Called after the page is shown (the page clears
    /// the screen). Returns the time spent, in microseconds.
    fn images_draw_grid(order: &[u16; OUTPUT_CLASSES]) -> i64 {
        let Some(pack) = pack() else { return 0 };
        let t0 = unsafe { sceKernelGetSystemTimeWide() };
        let (w, h) = (pack.header.w as usize, pack.header.h as usize);
        let x0 = imfile::device::SCREEN_W - GRID_COLS * (w + GRID_GAP);
        let map = unsafe { &*core::ptr::addr_of!(IMAGE_MAP) };
        let img = image_buf(pack.header.image_len());
        for (k, &class) in order.iter().take(GRID_COLS * GRID_ROWS).enumerate() {
            let idx = map[class as usize];
            if idx == imfile::NO_IMAGE {
                continue;
            }
            if let Err(e) = pack.read_image(idx as usize, img) {
                psp_rt::dprintln!("pspbird: images: read {} failed: {:?}", idx, e);
                break;
            }
            let x = x0 + (k % GRID_COLS) * (w + GRID_GAP);
            let y = GRID_TOP + (k / GRID_COLS) * (h + GRID_GAP);
            pack.draw(img, x, y);
        }
        unsafe { sceKernelGetSystemTimeWide() - t0 }
    }

    // ---- inference progress bar ------------------------------------------
    // The generated forward_timed()s call a plain `fn() -> u64` before and
    // after every op, so the bar is driven from inside inference with no
    // thread: `progress::tick` counts calls and paints straight into the
    // framebuffer (no dprint, so the page underneath is not cleared).
    // Progress is weighted by the previous run's elapsed time at the same
    // call index -- ops differ in cost by 100x, so counting them would
    // stall on the big convs -- and by call count on the first run.
    mod progress {
        use super::{sceKernelGetSystemTimeWide, TIMED_TICKS};
        use birdnet::imfile::device::{framebuffer, SCREEN_STRIDE};

        const CAP: usize = TIMED_TICKS + 2;
        /// Elapsed µs at each tick of the previous run; [LAST_N] is its end.
        static mut LAST: [u32; CAP] = [0; CAP];
        static mut LAST_N: usize = 0;
        static mut CALLS: usize = 0;
        static mut START_US: i64 = 0;
        static mut DRAWN: usize = usize::MAX;

        // Geometry: below the three-line "Classifying" page.
        const X: usize = 20;
        const Y: usize = 56;
        const W: usize = 400;
        const H: usize = 12;
        /// Orbiting dot to the right of the bar: shows life between the
        /// big ops, when the bar itself does not move.
        const DOT_CX: usize = X + W + 26;
        const DOT_CY: usize = Y + H / 2;
        const DOT_R: f32 = 9.0;

        const BORDER: u32 = 0xffe0_e0e0;
        const FILL: u32 = 0xff3e_5cff; // coral, matches the icon
        const EMPTY: u32 = 0xff30_2020;

        unsafe fn rect(x: usize, y: usize, w: usize, h: usize, c: u32) {
            let fb = framebuffer();
            for yy in y..y + h {
                let row = fb.add(yy * SCREEN_STRIDE + x);
                for xx in 0..w {
                    *row.add(xx) = c;
                }
            }
        }

        pub fn begin() {
            unsafe {
                CALLS = 0;
                START_US = sceKernelGetSystemTimeWide();
                DRAWN = usize::MAX;
                rect(X - 2, Y - 2, W + 4, H + 4, BORDER);
                rect(X, Y, W, H, EMPTY);
            }
        }

        fn fraction(calls: usize, elapsed_us: i64) -> f32 {
            unsafe {
                if LAST_N > 0 && LAST[LAST_N] > 0 {
                    let i = calls.min(LAST_N);
                    // Within a long op, advance by wall clock toward the
                    // next mark so the bar creeps instead of stalling.
                    let at = LAST[i] as f32;
                    let next = if i + 1 <= LAST_N { LAST[i + 1] as f32 } else { at };
                    let est = (elapsed_us as f32).min(next).max(at);
                    (est / LAST[LAST_N] as f32).min(0.995)
                } else {
                    (calls as f32 / TIMED_TICKS as f32).min(0.995)
                }
            }
        }

        /// The tick handed to `classify_birds_timed`.
        pub fn tick() -> u64 {
            let now = unsafe { sceKernelGetSystemTimeWide() };
            unsafe {
                let elapsed = now - START_US;
                let i = CALLS.min(CAP - 1);
                LAST[i] = elapsed as u32;
                CALLS += 1;
                let filled = (fraction(i, elapsed) * W as f32) as usize;
                if filled != DRAWN {
                    rect(X, Y, filled, H, FILL);
                    DRAWN = filled;
                }
                // Dot: one orbit per second.
                let a = (elapsed % 1_000_000) as f32 / 1_000_000.0 * core::f32::consts::TAU;
                rect(DOT_CX - 12, DOT_CY - 12, 24, 24, 0);
                let dx = (libm::cosf(a) * DOT_R) as i32;
                let dy = (libm::sinf(a) * DOT_R) as i32;
                rect((DOT_CX as i32 + dx - 1) as usize, (DOT_CY as i32 + dy - 1) as usize, 3, 3, BORDER);
            }
            now as u64
        }

        /// Keep this run's timeline as next run's estimate.
        pub fn end() {
            unsafe {
                let n = CALLS.min(CAP - 1);
                LAST[n] = (sceKernelGetSystemTimeWide() - START_US) as u32;
                LAST_N = n;
                rect(X, Y, W, H, FILL);
                rect(DOT_CX - 12, DOT_CY - 12, 24, 24, 0);
            }
        }
    }

    // ------------------------------------------------------------------
    // Screen: a fixed 27-row page redrawn whole. psp::dprint! keeps a
    // 27-row ring, so writing exactly ROWS newline-terminated lines
    // replaces the page rather than scrolling it.
    // ------------------------------------------------------------------

    const ROWS: usize = 27;

    struct Page {
        buf: [u8; ROWS * 90],
        len: usize,
        rows: usize,
    }

    impl Page {
        fn new() -> Self {
            Page { buf: [0; ROWS * 90], len: 0, rows: 0 }
        }

        fn line(&mut self, args: core::fmt::Arguments) {
            let _ = self.write_fmt(args);
            let _ = self.write_str("\n");
            self.rows += 1;
        }

        fn show(mut self) {
            while self.rows < ROWS {
                let _ = self.write_str("\n");
                self.rows += 1;
            }
            let s = core::str::from_utf8(&self.buf[..self.len]).unwrap_or("?");
            psp::dprint!("{}", s);
        }
    }

    impl Write for Page {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let n = s.len().min(self.buf.len() - self.len);
            self.buf[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
            self.len += n;
            Ok(())
        }
    }

    macro_rules! line {
        ($page:expr, $($arg:tt)*) => { $page.line(format_args!($($arg)*)) };
    }

    // ------------------------------------------------------------------
    // Input: edge-triggered buttons.
    // ------------------------------------------------------------------

    struct Pad {
        last: CtrlButtons,
    }

    impl Pad {
        fn new() -> Self {
            unsafe { sceCtrlSetSamplingMode(CtrlMode::Digital) };
            Pad { last: CtrlButtons::empty() }
        }

        /// Buttons that went down since the previous poll.
        fn pressed(&mut self) -> CtrlButtons {
            let mut data = SceCtrlData::default();
            unsafe { sceCtrlPeekBufferPositive(&mut data, 1) };
            let now = data.buttons;
            let edge = now & !self.last;
            self.last = now;
            edge
        }

        /// Block until one of `mask` is pressed; returns which.
        fn wait(&mut self, mask: CtrlButtons) -> CtrlButtons {
            loop {
                let p = self.pressed() & mask;
                if !p.is_empty() {
                    return p;
                }
                unsafe { sceKernelDelayThread(20_000) };
            }
        }
    }

    // ------------------------------------------------------------------
    // Audio.
    // ------------------------------------------------------------------

    fn capture() -> &'static mut [i16] {
        unsafe {
            let all: &mut [i16; MAX_SAMPLES] = &mut *core::ptr::addr_of_mut!(CAPTURE);
            &mut all[..CAPTURED]
        }
    }

    /// Record until X is pressed again or the buffer fills.
    fn record(pad: &mut Pad) {
        let r = unsafe { sceAudioInputInit(0, 1, 0) };
        if r < 0 {
            psp_rt::dprintln!("pspbird: sceAudioInputInit failed: {:#x}", r);
            let mut p = Page::new();
            line!(p, "Microphone init failed: {:#x}", r);
            line!(p, "Press X to continue");
            p.show();
            pad.wait(CtrlButtons::CROSS);
            return;
        }
        let buf = unsafe { &mut *core::ptr::addr_of_mut!(CAPTURE) };
        let mut n = 0usize;
        let mut last_shown = usize::MAX;
        while n + CHUNK <= MAX_SAMPLES {
            // Redraw once a second: each redraw clears and repaints the
            // whole framebuffer, which is not free.
            let sec = n / SAMPLE_RATE as usize;
            if sec != last_shown {
                let mut p = Page::new();
                line!(p, "RECORDING  {:2} s / {} s", sec, MAX_SECONDS);
                line!(p, "");
                line!(p, "X  stop");
                p.show();
                last_shown = sec;
            }
            unsafe {
                sceAudioInputBlocking(
                    CHUNK as i32,
                    AudioInputFrequency::Khz44_1,
                    buf[n..].as_mut_ptr() as *mut c_void,
                );
            }
            n += CHUNK;
            if pad.pressed().contains(CtrlButtons::CROSS) {
                break;
            }
        }
        unsafe { CAPTURED = n };
        psp_rt::dprintln!("pspbird: captured {} samples ({} ms)", n, n as u32 * 1000 / SAMPLE_RATE);
    }

    /// Play the capture back on a hardware channel (stereo, both sides the
    /// mono signal). X cancels.
    fn play(pad: &mut Pad) {
        let samples = capture();
        if samples.is_empty() {
            return;
        }
        let ch = unsafe { sceAudioChReserve(AUDIO_NEXT_CHANNEL, CHUNK as i32, AudioFormat::Stereo) };
        if ch < 0 {
            psp_rt::dprintln!("pspbird: sceAudioChReserve failed: {:#x}", ch);
            return;
        }
        let mut stereo = [0i16; CHUNK * 2];
        let mut last_shown = usize::MAX;
        let total = samples.len();
        for (ci, chunk) in samples.chunks(CHUNK).enumerate() {
            let sec = ci * CHUNK / SAMPLE_RATE as usize;
            if sec != last_shown {
                let mut p = Page::new();
                line!(p, "PLAYING  {:2} s / {} s", sec, total / SAMPLE_RATE as usize);
                line!(p, "");
                line!(p, "X  stop");
                p.show();
                last_shown = sec;
            }
            stereo.fill(0);
            for (i, &s) in chunk.iter().enumerate() {
                stereo[2 * i] = s;
                stereo[2 * i + 1] = s;
            }
            unsafe {
                sceAudioOutputPannedBlocking(
                    ch,
                    AUDIO_VOLUME_MAX as i32,
                    AUDIO_VOLUME_MAX as i32,
                    stereo.as_mut_ptr() as *mut c_void,
                );
            }
            if pad.pressed().contains(CtrlButtons::CROSS) {
                break;
            }
        }
        unsafe { sceAudioChRelease(ch) };
    }

    fn peak() -> i16 {
        capture().iter().map(|s| s.unsigned_abs()).max().unwrap_or(0).min(i16::MAX as u16) as i16
    }

    // ------------------------------------------------------------------
    // Screens.
    // ------------------------------------------------------------------

    fn region_menu(pad: &mut Pad) -> usize {
        let mut sel = psp_bird::loaded_region().unwrap_or(0);
        let mut note = "";
        loop {
            let mut p = Page::new();
            line!(p, "PSPBird");
            line!(p, "");
            line!(p, "Select a region (which pruned classifier to load):");
            line!(p, "");
            for (i, (r, _)) in REGIONS.iter().enumerate() {
                line!(p, "  {} {}", if i == sel { ">" } else { " " }, r);
            }
            line!(p, "");
            line!(p, "{} classes per region (TOPK={})", OUTPUT_CLASSES, psp_bird::APP_TOPK);
            if !note.is_empty() {
                line!(p, "{}", note);
            }
            line!(p, "");
            line!(p, "UP/DOWN  move    X  select");
            p.show();

            let b = pad.wait(CtrlButtons::UP | CtrlButtons::DOWN | CtrlButtons::CROSS);
            if b.contains(CtrlButtons::UP) {
                sel = (sel + REGIONS.len() - 1) % REGIONS.len();
            } else if b.contains(CtrlButtons::DOWN) {
                sel = (sel + 1) % REGIONS.len();
            } else {
                let mut p = Page::new();
                line!(p, "Loading {} ...", REGIONS[sel].0);
                p.show();
                match psp_bird::load_region(sel) {
                    Ok(()) => return sel,
                    Err(e) => {
                        psp_rt::dprintln!("pspbird: load_region({}) failed: {:?}", REGIONS[sel].0, e);
                        note = "Load failed (see host log); pick another region";
                    }
                }
            }
        }
    }

    /// First 3 s of the capture, 44.1 kHz i16 -> 48 kHz f32 in [-1, 1] by
    /// linear interpolation; zero-padded when the recording is shorter.
    fn prepare_model_input() -> &'static [f32; INPUT_SAMPLES] {
        let src = capture();
        let dst = unsafe { &mut *core::ptr::addr_of_mut!(MODEL_INPUT) };
        let ratio = SAMPLE_RATE as f32 / 48_000.0;
        for (i, d) in dst.iter_mut().enumerate() {
            let pos = i as f32 * ratio;
            let j = pos as usize;
            let frac = pos - j as f32;
            let a = src.get(j).copied().unwrap_or(0) as f32;
            let b = src.get(j + 1).copied().unwrap_or(0) as f32;
            *d = (a + (b - a) * frac) / 32768.0;
        }
        dst
    }

    fn sort_order(scores: &[f32; OUTPUT_CLASSES], order: &mut [u16; OUTPUT_CLASSES]) {
        for (i, o) in order.iter_mut().enumerate() {
            *o = i as u16;
        }
        order.sort_unstable_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
    }

    /// The one place a dense score vector becomes a screen: `title` on top,
    /// then classes from `top`, best first. Returns how many rows fit.
    fn results_page(
        title: core::fmt::Arguments,
        scores: &[f32; OUTPUT_CLASSES],
        order: &[u16; OUTPUT_CLASSES],
        top: usize,
    ) -> usize {
        const PAGE: usize = ROWS - 3;
        let cols = unsafe { TEXT_COLS };
        let mut p = Page::new();
        p.line(title);
        line!(p, "");
        for &i in order.iter().skip(top).take(PAGE) {
            let i = i as usize;
            // Clip to the text area so labels never run under the grid.
            let label = psp_bird::label(i);
            let mut cut = label.len().min(cols.saturating_sub(8));
            while !label.is_char_boundary(cut) {
                cut -= 1;
            }
            line!(p, "{:5.1}%  {}", psp_bird::sigmoid(scores[i]) * 100.0, &label[..cut]);
        }
        p.show();
        let us = images_draw_grid(order);
        if us > 0 && top == 0 {
            psp_rt::dprintln!("pspbird: images: grid drawn in {} us", us);
        }
        PAGE
    }

    fn log_top5(scores: &[f32; OUTPUT_CLASSES], order: &[u16; OUTPUT_CLASSES], ms: u32) {
        psp_rt::dprintln!("pspbird: classified in {} ms, top 5:", ms);
        for &i in order.iter().take(5) {
            psp_rt::dprintln!(
                "  {:.3}  {}",
                psp_bird::sigmoid(scores[i as usize]),
                psp_bird::label(i as usize)
            );
        }
    }

    /// Scroll input for a results page: UP/DOWN a line, L/R a page.
    /// Returns the new top, or None on any other button in `exit`.
    fn scroll(pad: &mut Pad, top: usize, page: usize, exit: CtrlButtons) -> Option<usize> {
        let b = pad.wait(
            CtrlButtons::UP | CtrlButtons::DOWN | CtrlButtons::LTRIGGER | CtrlButtons::RTRIGGER
                | exit,
        );
        let max_top = OUTPUT_CLASSES.saturating_sub(page);
        Some(if b.contains(CtrlButtons::UP) {
            top.saturating_sub(1)
        } else if b.contains(CtrlButtons::DOWN) {
            (top + 1).min(max_top)
        } else if b.contains(CtrlButtons::LTRIGGER) {
            top.saturating_sub(page)
        } else if b.contains(CtrlButtons::RTRIGGER) {
            (top + page).min(max_top)
        } else {
            return None;
        })
    }

    /// Single-step: run the model on the capture and show every class.
    fn classify(pad: &mut Pad, region: usize) {
        let mut p = Page::new();
        line!(p, "Classifying [{}] ...", REGIONS[region].0);
        line!(p, "");
        line!(p, "(about 4 s)");
        p.show();

        let input = prepare_model_input();
        let scores = unsafe { &mut *core::ptr::addr_of_mut!(SCORES) };
        let t0 = unsafe { sceKernelGetSystemTimeWide() };
        progress::begin();
        psp_bird::classify_birds_timed(input, scores, progress::tick);
        progress::end();
        let ms = ((unsafe { sceKernelGetSystemTimeWide() } - t0) / 1000) as u32;
        let order = unsafe { &mut *core::ptr::addr_of_mut!(ORDER) };
        sort_order(scores, order);
        log_top5(scores, order, ms);

        let mut top = 0usize;
        loop {
            let page = results_page(
                format_args!("Results [{}]  {} ms   X back", REGIONS[region].0, ms),
                scores,
                order,
                top,
            );
            match scroll(pad, top, page, CtrlButtons::CROSS | CtrlButtons::CIRCLE) {
                Some(t) => top = t,
                None => return,
            }
        }
    }

    // ------------------------------------------------------------------
    // Live mode.
    // ------------------------------------------------------------------

    /// Resample the newest 3 s of the ring into MODEL_INPUT. Only reads
    /// samples older than the write pointer; the ring is big enough that
    /// the writer cannot lap them during the copy.
    fn snapshot_window() -> &'static [f32; INPUT_SAMPLES] {
        let written = RING_WRITTEN.load(Ordering::Acquire);
        let start = written - WINDOW;
        let ring = unsafe { &*core::ptr::addr_of!(RING) };
        let dst = unsafe { &mut *core::ptr::addr_of_mut!(MODEL_INPUT) };
        let ratio = SAMPLE_RATE as f32 / 48_000.0;
        for (i, d) in dst.iter_mut().enumerate() {
            let pos = i as f32 * ratio;
            let j = pos as usize;
            let frac = pos - j as f32;
            let a = ring[(start + j) % RING_LEN] as f32;
            let b = ring[(start + j + 1) % RING_LEN] as f32;
            *d = (a + (b - a) * frac) / 32768.0;
        }
        dst
    }

    /// The inference thread: whenever a full new window is available and
    /// it is free, classify the newest 3 s and publish. Never queues — a
    /// slow model costs coverage, not lag.
    unsafe extern "C" fn infer_thread(_argc: usize, _argv: *mut c_void) -> i32 {
        let mut last_start = 0usize;
        while !LIVE_STOP.load(Ordering::Relaxed) {
            let written = RING_WRITTEN.load(Ordering::Acquire);
            if written < WINDOW || written - WINDOW < last_start + WINDOW {
                // Poll at ~mic-chunk rate: the UI thread owns the clock.
                sceKernelDelayThread(25_000);
                continue;
            }
            last_start = written - WINDOW;
            let input = snapshot_window();
            let scores = &mut *core::ptr::addr_of_mut!(LIVE_SCORES);
            let t0 = sceKernelGetSystemTimeWide();
            psp_bird::classify_birds(input, scores);
            let ms = ((sceKernelGetSystemTimeWide() - t0) / 1000) as u32;
            LIVE_LAST_MS.store(ms, Ordering::Relaxed);
            LIVE_SEQ.fetch_add(1, Ordering::Release);
        }
        0
    }

    /// Stream the mic into the ring on this thread while the inference
    /// thread works behind it; redraw whenever a result lands. X leaves.
    fn live(pad: &mut Pad, region: usize) {
        let r = unsafe { sceAudioInputInit(0, 1, 0) };
        if r < 0 {
            psp_rt::dprintln!("pspbird: sceAudioInputInit failed: {:#x}", r);
            return;
        }
        RING_WRITTEN.store(0, Ordering::Release);
        LIVE_STOP.store(false, Ordering::Release);
        LIVE_SEQ.store(0, Ordering::Release);
        let thid = unsafe {
            sceKernelCreateThread(
                b"pspbird_infer\0".as_ptr(),
                infer_thread,
                INFER_PRIORITY,
                64 * 1024,
                ThreadAttributes::USER | ThreadAttributes::VFPU,
                core::ptr::null_mut(),
            )
        };
        if thid.0 < 0 {
            psp_rt::dprintln!("pspbird: sceKernelCreateThread failed: {:#x}", thid.0);
            return;
        }
        unsafe { sceKernelStartThread(thid, 0, core::ptr::null_mut()) };
        psp_rt::dprintln!("pspbird: live mode on [{}]", REGIONS[region].0);

        let ring = unsafe { &mut *core::ptr::addr_of_mut!(RING) };
        let scores = unsafe { &*core::ptr::addr_of!(LIVE_SCORES) };
        let order = unsafe { &mut *core::ptr::addr_of_mut!(ORDER) };
        let mut seen = 0u32;
        let mut top = 0usize;
        let mut page = ROWS - 3;
        let mut redraw = true;
        let mut windows = 0u32;
        let t_start = unsafe { sceKernelGetSystemTimeWide() };
        loop {
            // One mic chunk (~23 ms). The write index is published after
            // the data so the reader never sees a half-written chunk.
            let w = RING_WRITTEN.load(Ordering::Relaxed);
            let at = w % RING_LEN;
            unsafe {
                sceAudioInputBlocking(
                    CHUNK as i32,
                    AudioInputFrequency::Khz44_1,
                    ring[at..].as_mut_ptr() as *mut c_void,
                );
            }
            RING_WRITTEN.store(w + CHUNK, Ordering::Release);

            let seq = LIVE_SEQ.load(Ordering::Acquire);
            if seq != seen {
                seen = seq;
                windows += 1;
                sort_order(scores, order);
                log_top5(scores, order, LIVE_LAST_MS.load(Ordering::Relaxed));
                redraw = true;
            }
            let b = pad.pressed();
            if b.contains(CtrlButtons::CROSS) {
                break;
            }
            let max_top = OUTPUT_CLASSES.saturating_sub(page);
            if b.contains(CtrlButtons::UP) {
                top = top.saturating_sub(1);
                redraw = true;
            } else if b.contains(CtrlButtons::DOWN) {
                top = (top + 1).min(max_top);
                redraw = true;
            }
            if redraw {
                redraw = false;
                let secs = ((unsafe { sceKernelGetSystemTimeWide() } - t_start) / 1_000_000) as u32;
                let ms = LIVE_LAST_MS.load(Ordering::Relaxed);
                if windows == 0 {
                    let mut p = Page::new();
                    line!(p, "LIVE [{}]  {} s   listening...", REGIONS[region].0, secs);
                    line!(p, "");
                    line!(p, "First result after ~7 s (3 s window + inference).");
                    line!(p, "");
                    line!(p, "X  stop");
                    p.show();
                } else {
                    // Coverage: fraction of audio actually analysed.
                    let cov = (windows as u64 * 3 * 100 / secs.max(1) as u64).min(100);
                    page = results_page(
                        format_args!(
                            "LIVE [{}] {}s  #{}  {}ms  cov {}%  X stop",
                            REGIONS[region].0, secs, windows, ms, cov
                        ),
                        scores,
                        order,
                        top,
                    );
                }
            }
        }

        LIVE_STOP.store(true, Ordering::Release);
        unsafe {
            sceKernelWaitThreadEnd(thid, core::ptr::null_mut());
            sceKernelDeleteThread(thid);
        }
        psp_rt::dprintln!("pspbird: live mode off after {} windows", windows);
    }

    /// Single-step review. Returns false to go back to the region menu.
    fn review(pad: &mut Pad, region: usize, note: &str) -> bool {
        let mut p = Page::new();
        let n = capture().len();
        line!(p, "PSPBird  [{}]", REGIONS[region].0);
        line!(p, "");
        line!(
            p,
            "Recording: {}.{:02} s, peak {}/32767",
            n / SAMPLE_RATE as usize,
            (n % SAMPLE_RATE as usize) * 100 / SAMPLE_RATE as usize,
            peak()
        );
        if !note.is_empty() {
            line!(p, "");
            line!(p, "{}", note);
        }
        line!(p, "");
        line!(p, "SQUARE    play back");
        line!(p, "CIRCLE    classify");
        line!(p, "TRIANGLE  record again");
        line!(p, "SELECT    change region");
        p.show();

        let b = pad.wait(
            CtrlButtons::SQUARE | CtrlButtons::CIRCLE | CtrlButtons::TRIANGLE | CtrlButtons::SELECT,
        );
        if b.contains(CtrlButtons::SQUARE) {
            play(pad);
            review(pad, region, "")
        } else if b.contains(CtrlButtons::CIRCLE) {
            classify(pad, region);
            review(pad, region, "")
        } else {
            b.contains(CtrlButtons::TRIANGLE)
        }
    }

    fn app_main() {
        psp_rt::enable_home_button();
        unsafe { psp::sys::scePowerSetClockFrequency(333, 333, 166) };
        psp_rt::dprintln!("pspbird: started ({} regions, {} classes)", REGIONS.len(), OUTPUT_CLASSES);
        psp_bird::frontend::init();
        psp_bird::backbone::init();
        psp_bird::classifier::init();
        images_init();

        let mut pad = Pad::new();
        loop {
            let region = region_menu(&mut pad);
            images_select_region(region);
            loop {
                let mut p = Page::new();
                line!(p, "PSPBird  [{}]", REGIONS[region].0);
                line!(p, "");
                line!(p, "X       single-step: record up to {} s, then classify", MAX_SECONDS);
                line!(p, "CIRCLE  live: continuous listening");
                line!(p, "SELECT  change region");
                p.show();
                let b = pad.wait(CtrlButtons::CROSS | CtrlButtons::CIRCLE | CtrlButtons::SELECT);
                if b.contains(CtrlButtons::SELECT) {
                    break;
                }
                if b.contains(CtrlButtons::CIRCLE) {
                    live(&mut pad, region);
                    continue;
                }
                record(&mut pad);
                if !review(&mut pad, region, "") {
                    break;
                }
            }
        }
    }
}
