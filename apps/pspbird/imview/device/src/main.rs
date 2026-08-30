//! pspbird-imview: de-risks image mode for the PSPBird app. Opens the
//! `PBIM` species-image pack the host runner wrote (`host0:/birds.img`),
//! lists its names, and on X seeks to the chosen image, reads it, and
//! blits it onto the debug screen, timing each step. UP/DOWN move the
//! cursor, SELECT quits.
//!
//! Uses the same fixed-page trick as pspbird: `psp::dprint!` keeps a
//! 27-row ring and clears the screen on every print, so a page is 27
//! newline-terminated lines and the image is drawn *after* the page.
//!
//! Measured (hostfs, 96x96): ~2.1 ms seek+read per 18 KB image, ~1.06 ms
//! blit -- so a top-6 grid costs ~20 ms per results update.

#![no_std]
#![no_main]

use birdnet::imfile::device::{OpenError, Pack, SCREEN_W};
use core::fmt::Write;
use psp::sys::{
    sceCtrlPeekBufferPositive, sceCtrlSetSamplingMode, sceKernelDelayThread,
    sceKernelGetSystemTimeWide, CtrlButtons, CtrlMode, SceCtrlData,
};

// Interactive: no watchdog, the user decides when it ends.
psp_rt::module!("pspbird_imview", 1, 0, timeout_secs = 0);

const PACK_PATHS: [&str; 2] = ["host0:/birds.img", "birds.img"];
/// Largest image this build can hold (bytes): 128x128 RGB565.
const IMAGE_CAP: usize = 128 * 128 * 2;
/// Name table cap: ~1150 species at ~45 bytes is 52 KiB.
const NAMES_CAP: usize = 64 * 1024;

static mut NAMES_BUF: [u8; NAMES_CAP] = [0; NAMES_CAP];
static mut IMAGE_BUF: [u8; IMAGE_CAP] = [0; IMAGE_CAP];

/// The first `len` bytes of `IMAGE_BUF`. Single-threaded, so the aliasing
/// is by construction: one live borrow at a time.
fn image_buf(len: usize) -> &'static mut [u8] {
    assert!(len <= IMAGE_CAP);
    unsafe { core::slice::from_raw_parts_mut(core::ptr::addr_of_mut!(IMAGE_BUF) as *mut u8, len) }
}

fn names_buf() -> &'static mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(core::ptr::addr_of_mut!(NAMES_BUF) as *mut u8, NAMES_CAP) }
}

/// Where the image goes: right edge, under the title row.
const IMAGE_Y: usize = 12;
const ROWS: usize = 27;
/// Menu entries shown at once.
const VISIBLE: usize = 18;
/// Quit after this long with no input (the host runner's idle timeout
/// is the same, so both sides give up together).
const IDLE_QUIT_US: i64 = 10 * 60 * 1_000_000;

// ---- screen page ------------------------------------------------------

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

// ---- input ------------------------------------------------------------

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
    /// Block until one of `mask` is pressed, or `IDLE_QUIT_US` passes
    /// with nothing pressed (returns empty): a run left alone ends
    /// instead of holding psplink's shell thread forever.
    fn wait(&mut self, mask: CtrlButtons) -> CtrlButtons {
        let start = now_us();
        loop {
            let p = self.pressed() & mask;
            if !p.is_empty() {
                return p;
            }
            if now_us() - start > IDLE_QUIT_US {
                return CtrlButtons::empty();
            }
            unsafe { sceKernelDelayThread(20_000) };
        }
    }
}

fn now_us() -> i64 {
    unsafe { sceKernelGetSystemTimeWide() }
}

/// One image load, timed: seek + read, then blit. Returns (io_us, blit_us).
fn load_and_draw(pack: &Pack, i: usize) -> Result<(i64, i64), OpenError> {
    let img = image_buf(pack.header.image_len());
    let t0 = now_us();
    pack.read_image(i, img)?;
    let t1 = now_us();
    pack.draw(img, SCREEN_W - pack.header.w as usize - 4, IMAGE_Y);
    let t2 = now_us();
    Ok((t1 - t0, t2 - t1))
}

// ---- app --------------------------------------------------------------

struct Stats {
    loads: u32,
    last_io_us: i64,
    last_blit_us: i64,
    total_io_us: i64,
    total_blit_us: i64,
}

fn app_main() {
    psp_rt::enable_home_button();
    unsafe { psp::sys::scePowerSetClockFrequency(333, 333, 166) };
    psp_rt::dprintln!("imview: started");

    let t0 = now_us();
    let pack = match Pack::open(&PACK_PATHS) {
        Ok(p) => p,
        Err(e) => {
            psp_rt::dprintln!("imview: no usable pack ({:?}); run the host with a fetched manifest", e);
            return;
        }
    };
    let header = pack.header;
    if header.image_len() > IMAGE_CAP {
        psp_rt::dprintln!("imview: {}x{} exceeds this build's image buffer", header.w, header.h);
        pack.close();
        return;
    }
    let names = match pack.read_names(names_buf()) {
        Ok(n) => n,
        Err(e) => {
            psp_rt::dprintln!("imview: name table: {:?}", e);
            pack.close();
            return;
        }
    };
    let open_us = now_us() - t0;
    psp_rt::dprintln!(
        "imview: {} images {}x{} ({} B each), {} regions, names {} B, header+names in {} us",
        header.n, header.w, header.h, header.image_len(), header.n_regions, header.names_len, open_us
    );

    let mut pad = Pad::new();
    let mut cursor = 0usize;
    let mut shown: Option<usize> = None;
    let mut stats = Stats { loads: 0, last_io_us: 0, last_blit_us: 0, total_io_us: 0, total_blit_us: 0 };
    let n = names.len();

    // One unattended load so a run that nobody touches still reports a
    // number (and the first-read cost, which includes hostfs warm-up,
    // shows up separately from the interactive ones).
    match load_and_draw(&pack, 0) {
        Ok((io, blit)) => {
            psp_rt::dprintln!("imview: cold load of image 0: io {} us, blit {} us", io, blit);
            shown = Some(0);
        }
        Err(e) => psp_rt::dprintln!("imview: cold load failed: {:?}", e),
    }

    loop {
        let mut p = Page::new();
        line!(p, "PSPBird image viewer   {} images {}x{}", n, header.w, header.h);
        line!(p, "");
        let top = cursor.saturating_sub(VISIBLE / 2).min(n.saturating_sub(VISIBLE));
        for i in top..(top + VISIBLE).min(n) {
            let mark = if Some(i) == shown { '*' } else { ' ' };
            let arrow = if i == cursor { '>' } else { ' ' };
            line!(p, "{}{} {:3}  {:.50}", arrow, mark, i, names.get(i).unwrap_or("?"));
        }
        while p.rows < 2 + VISIBLE {
            line!(p, "");
        }
        line!(p, "");
        if stats.loads > 0 {
            line!(
                p,
                "last: io {} us, blit {} us   avg over {}: io {} us, blit {} us",
                stats.last_io_us,
                stats.last_blit_us,
                stats.loads,
                stats.total_io_us / stats.loads as i64,
                stats.total_blit_us / stats.loads as i64
            );
        } else {
            line!(p, "no image loaded yet");
        }
        line!(p, "UP/DOWN select   X load   SELECT quit");
        p.show();
        if shown.is_some() {
            // The page clear wiped it; re-blit from the buffer (no I/O).
            pack.draw(image_buf(header.image_len()), SCREEN_W - header.w as usize - 4, IMAGE_Y);
        }

        let b = pad.wait(CtrlButtons::UP | CtrlButtons::DOWN | CtrlButtons::CROSS | CtrlButtons::SELECT);
        if b.contains(CtrlButtons::SELECT) {
            break;
        }
        if b.is_empty() {
            psp_rt::dprintln!("imview: idle, quitting");
            break;
        }
        if b.contains(CtrlButtons::UP) && cursor > 0 {
            cursor -= 1;
        }
        if b.contains(CtrlButtons::DOWN) && cursor + 1 < n {
            cursor += 1;
        }
        if b.contains(CtrlButtons::CROSS) {
            match load_and_draw(&pack, cursor) {
                Ok((io, blit)) => {
                    stats.loads += 1;
                    stats.last_io_us = io;
                    stats.last_blit_us = blit;
                    stats.total_io_us += io;
                    stats.total_blit_us += blit;
                    shown = Some(cursor);
                    psp_rt::dprintln!(
                        "imview: image {} ({}) io {} us ({} B), blit {} us",
                        cursor,
                        names.get(cursor).unwrap_or("?"),
                        io,
                        header.image_len(),
                        blit
                    );
                }
                Err(e) => psp_rt::dprintln!("imview: load {} failed: {:?}", cursor, e),
            }
        }
    }

    pack.close();
    psp_rt::dprintln!(
        "imview: {} loads, avg io {} us, avg blit {} us",
        stats.loads,
        if stats.loads > 0 { stats.total_io_us / stats.loads as i64 } else { 0 },
        if stats.loads > 0 { stats.total_blit_us / stats.loads as i64 } else { 0 }
    );
}
