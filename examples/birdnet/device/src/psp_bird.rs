//! BirdNET for the app: frontend -> headless backbone -> external-weight
//! classifier, with the classifier's rows chosen at runtime by loading a
//! region blob into the generated slots.
//!
//! ```ignore
//! psp_bird::load_region(0)?;                 // index into REGIONS
//! psp_bird::classify_birds(&audio, &mut out); // out[i] is the raw logit
//! psp_bird::label(i)
//! ```
//!
//! Blob format (`PBRD`, written by prune_classifier.py --write-blob): a
//! 32-byte header, then `[N, 1024]` f32 weights, `[N]` f32 bias, and the
//! newline-separated labels. N is frozen at build time (`OUTPUT_CLASSES`),
//! so every region's blob has the same width and a load is a straight read
//! into the slots — no reallocation, no per-region code.

use core::ffi::c_void;
use psp::sys::{sceIoClose, sceIoOpen, sceIoRead, IoOpenFlags};

#[allow(dead_code)]
pub mod frontend {
    include!(concat!(env!("OUT_DIR"), "/app/frontend/custom_frontend.rs"));
}
#[allow(dead_code)]
pub mod backbone {
    include!(concat!(env!("OUT_DIR"), "/app/generated.rs"));
}
#[allow(dead_code)]
pub mod classifier {
    include!(concat!(env!("OUT_DIR"), "/app/classifier.rs"));
}
include!(concat!(env!("OUT_DIR"), "/app/regions.rs"));

pub const INPUT_SAMPLES: usize = 144_000;
pub const EMBEDDING: usize = 1024;
const N_BANKS: usize = 96;
const N_WINDOWS: usize = 511;
const MEL_LEN: usize = N_BANKS * N_WINDOWS;

/// Blobs live in `blobs/` beside the module: `psp_rt::module!` chdir()s to
/// the launch directory, so that is the staged copy next to the .prx under
/// psplink and the EBOOT's folder on a memory stick. `host0:/blobs/` (the
/// runner's mount) is the fallback.
const LOCAL_PREFIX: &str = "blobs/";
const HOST_PREFIX: &str = "host0:/blobs/";

const BLOB_MAGIC: &[u8; 4] = b"PBRD";
const BLOB_VERSION: u32 = 1;
const BLOB_HEADER: usize = 32;
const LABELS_CAP: usize = 32 * 1024;
const CHUNK: usize = 64 * 1024;

static mut MEL_2048: [f32; MEL_LEN] = [0.0; MEL_LEN];
static mut MEL_1024: [f32; MEL_LEN] = [0.0; MEL_LEN];
static mut BACKBONE_INPUT: [f32; MEL_LEN * 2] = [0.0; MEL_LEN * 2];
static mut EMB: [f32; EMBEDDING] = [0.0; EMBEDDING];
static mut LABELS: [u8; LABELS_CAP] = [0; LABELS_CAP];
static mut LABELS_LEN: usize = 0;
static mut LOADED: Option<usize> = None;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    /// Neither `blobs/` nor `host0:/blobs/` had it (`sceIoOpen` error code).
    NotFound(i32),
    Truncated,
    BadMagic,
    BadVersion(u32),
    /// The blob's `[n_classes, in_features]` is not what this build compiled.
    WrongShape(u32, u32),
    LabelsTooLong(u32),
}

/// Index into `REGIONS` of the loaded blob, if any.
pub fn loaded_region() -> Option<usize> {
    unsafe { LOADED }
}

pub fn label(index: usize) -> &'static str {
    let bytes = unsafe {
        let all: &[u8; LABELS_CAP] = &*core::ptr::addr_of!(LABELS);
        &all[..LABELS_LEN]
    };
    core::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.lines().nth(index))
        .unwrap_or("?")
}

/// Fill the classifier slots from `REGIONS[region]`'s blob.
pub fn load_region(region: usize) -> Result<(), LoadError> {
    let (name, file) = REGIONS[region];
    unsafe { LOADED = None };
    let fd = open_blob(file)?;
    let result = read_blob(fd);
    unsafe { sceIoClose(fd) };
    match result {
        Ok(()) => {
            unsafe { LOADED = Some(region) };
            psp_rt::dprintln!("psp_bird: loaded region {name} ({OUTPUT_CLASSES} classes)");
            Ok(())
        }
        Err(e) => {
            // Never run on a half-filled slot.
            classifier::weights().fill(0.0);
            classifier::bias().fill(0.0);
            Err(e)
        }
    }
}

/// Frontend -> backbone -> classifier. `audio` is 48 kHz mono in [-1, 1];
/// `out` receives raw logits (sigmoid for a confidence).
pub fn classify_birds(audio: &[f32; INPUT_SAMPLES], out: &mut [f32; OUTPUT_CLASSES]) {
    let m2048 = unsafe { &mut *core::ptr::addr_of_mut!(MEL_2048) };
    let m1024 = unsafe { &mut *core::ptr::addr_of_mut!(MEL_1024) };
    frontend::forward(audio, m2048, m1024);
    // Assemble the backbone's [1, 96, 511, 2] input from the bank-major
    // outputs: each branch mel-axis reversed (the model's REVERSE_V2) and
    // channel-interleaved, 2048 first — what the severed CONCAT consumed.
    let bb = unsafe { &mut *core::ptr::addr_of_mut!(BACKBONE_INPUT) };
    for q in 0..N_BANKS {
        let src = (N_BANKS - 1 - q) * N_WINDOWS;
        for t in 0..N_WINDOWS {
            let i = (q * N_WINDOWS + t) * 2;
            bb[i] = m2048[src + t];
            bb[i + 1] = m1024[src + t];
        }
    }
    let emb = unsafe { &mut *core::ptr::addr_of_mut!(EMB) };
    backbone::forward(bb, emb);
    classifier::forward(emb, out);
}

/// Tick calls `classify_birds_timed` makes: the generated `forward_timed`s
/// call `tick` before and after every op, in every stage.
pub const TIMED_TICKS: usize = 2 * (frontend::NUM_OPS + backbone::NUM_OPS + classifier::NUM_OPS);

static mut FRONTEND_TICKS: [u64; frontend::NUM_OPS] = [0; frontend::NUM_OPS];
static mut BACKBONE_TICKS: [u64; backbone::NUM_OPS] = [0; backbone::NUM_OPS];
static mut CLASSIFIER_TICKS: [u64; classifier::NUM_OPS] = [0; classifier::NUM_OPS];

/// `classify_birds` through the instrumented forwards: `tick` is called
/// around every op of every stage (`TIMED_TICKS` times), which is what a
/// progress display hooks. Per-op tick deltas land in the *_TICKS statics.
pub fn classify_birds_timed(
    audio: &[f32; INPUT_SAMPLES],
    out: &mut [f32; OUTPUT_CLASSES],
    tick: fn() -> u64,
) {
    let m2048 = unsafe { &mut *core::ptr::addr_of_mut!(MEL_2048) };
    let m1024 = unsafe { &mut *core::ptr::addr_of_mut!(MEL_1024) };
    let ft = unsafe { &mut *core::ptr::addr_of_mut!(FRONTEND_TICKS) };
    frontend::forward_timed(audio, m2048, m1024, ft, tick);
    let bb = unsafe { &mut *core::ptr::addr_of_mut!(BACKBONE_INPUT) };
    for q in 0..N_BANKS {
        let src = (N_BANKS - 1 - q) * N_WINDOWS;
        for t in 0..N_WINDOWS {
            let i = (q * N_WINDOWS + t) * 2;
            bb[i] = m2048[src + t];
            bb[i + 1] = m1024[src + t];
        }
    }
    let emb = unsafe { &mut *core::ptr::addr_of_mut!(EMB) };
    let bt = unsafe { &mut *core::ptr::addr_of_mut!(BACKBONE_TICKS) };
    backbone::forward_timed(bb, emb, bt, tick);
    let ct = unsafe { &mut *core::ptr::addr_of_mut!(CLASSIFIER_TICKS) };
    classifier::forward_timed(emb, out, ct, tick);
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

/// Below this confidence a class is not a detection at all (BirdNET's own
/// default `min_conf`). The outputs are independent sigmoids, not a
/// softmax, so this absolute floor has to come before any relative test:
/// 5% / 2% / 2% has the same *shape* as 60% / 30% / 30% and means nothing.
pub const MIN_CONF: f32 = 0.1;
/// Most species shown at once.
/// TODO: scale with what the page can fit instead of a fixed cap.
pub const MAX_SHOWN: usize = 3;

/// How many of the leading classes to present, given `order` (best first,
/// from `sort_order`): 0 if nothing clears `MIN_CONF`, else the *effective
/// number of species* among the top `MAX_SHOWN` survivors — the inverse
/// Simpson index (Hill number of order 2) of their confidence shares,
/// rounded. 90/1/1 -> 1, 60/30/5 -> 2, 60/30/30 -> 3.
pub fn how_many(scores: &[f32; OUTPUT_CLASSES], order: &[u16]) -> usize {
    let mut p = [0.0f32; MAX_SHOWN];
    let mut n = 0;
    for &i in order.iter().take(MAX_SHOWN) {
        let c = sigmoid(scores[i as usize]);
        if c < MIN_CONF {
            break;
        }
        p[n] = c;
        n += 1;
    }
    if n == 0 {
        return 0;
    }
    let total: f32 = p[..n].iter().sum();
    let simpson: f32 = p[..n].iter().map(|c| (c / total) * (c / total)).sum();
    let effective = 1.0 / simpson;
    (libm::roundf(effective) as usize).clamp(1, n)
}

// ---------------------------------------------------------------------------

/// Path bytes for `prefix + file`, NUL-terminated.
fn path(prefix: &str, file: &str) -> [u8; 96] {
    let mut p = [0u8; 96];
    let mut n = 0;
    for b in prefix.bytes().chain(file.bytes()) {
        p[n] = b;
        n += 1;
    }
    p
}

fn open_ro(p: &[u8]) -> Result<psp::sys::SceUid, i32> {
    let fd = unsafe { sceIoOpen(p.as_ptr(), IoOpenFlags::RD_ONLY, 0) };
    if fd.0 < 0 {
        Err(fd.0)
    } else {
        Ok(fd)
    }
}

/// Open the blob beside the module, else from the host mount.
fn open_blob(file: &str) -> Result<psp::sys::SceUid, LoadError> {
    match open_ro(&path(LOCAL_PREFIX, file)) {
        Ok(fd) => Ok(fd),
        Err(_) => open_ro(&path(HOST_PREFIX, file)).map_err(LoadError::NotFound),
    }
}

/// Read exactly `dst.len()` bytes in hostfs-friendly chunks.
fn read_exact(fd: psp::sys::SceUid, dst: &mut [u8]) -> Result<(), LoadError> {
    let mut done = 0usize;
    while done < dst.len() {
        let want = (dst.len() - done).min(CHUNK);
        let n = unsafe {
            sceIoRead(fd, dst[done..].as_mut_ptr() as *mut c_void, want as u32)
        };
        if n <= 0 {
            return Err(LoadError::Truncated);
        }
        done += n as usize;
    }
    Ok(())
}

fn read_blob(fd: psp::sys::SceUid) -> Result<(), LoadError> {
    let mut h = [0u8; BLOB_HEADER];
    read_exact(fd, &mut h)?;
    if &h[0..4] != BLOB_MAGIC {
        return Err(LoadError::BadMagic);
    }
    let u = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
    if u(4) != BLOB_VERSION {
        return Err(LoadError::BadVersion(u(4)));
    }
    let (n, k, labels_len) = (u(8), u(12), u(16));
    if n as usize != OUTPUT_CLASSES || k as usize != EMBEDDING {
        return Err(LoadError::WrongShape(n, k));
    }
    if labels_len as usize > LABELS_CAP {
        return Err(LoadError::LabelsTooLong(labels_len));
    }
    let w = classifier::weights();
    let w_bytes = unsafe {
        core::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, w.len() * 4)
    };
    read_exact(fd, w_bytes)?;
    let b = classifier::bias();
    let b_bytes = unsafe {
        core::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut u8, b.len() * 4)
    };
    read_exact(fd, b_bytes)?;
    let labels = unsafe { &mut *core::ptr::addr_of_mut!(LABELS) };
    read_exact(fd, &mut labels[..labels_len as usize])?;
    unsafe { LABELS_LEN = labels_len as usize };
    Ok(())
}
