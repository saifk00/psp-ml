#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use core::ffi::c_void;
use psp::dprintln;
use psp::sys::{
    sceAudioInputBlocking, sceAudioInputInit, sceCtrlReadBufferPositive,
    sceIoClose, sceIoDclose, sceIoDopen, sceIoDread, sceIoMkdir, sceIoOpen,
    sceIoRead, sceIoRemove, sceIoWrite,
    AudioInputFrequency, CtrlButtons, CtrlMode, IoOpenFlags,
    SceCtrlData, SceIoDirent,
};

psp::module!("audrec", 1, 0);

// Storage paths
const LOCAL_CACHE_DIR: &[u8] = b"ms0:/PSP/MUSIC/AUDREC\0";

// WAV file header (44 bytes)
#[repr(C, packed)]
struct WavHeader {
    riff_id: [u8; 4],
    file_size: u32,
    wave_id: [u8; 4],
    fmt_id: [u8; 4],
    fmt_size: u32,
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    data_id: [u8; 4],
    data_size: u32,
}

impl WavHeader {
    fn new(sample_rate: u32, data_size: u32) -> Self {
        WavHeader {
            riff_id: *b"RIFF",
            file_size: data_size + 36,
            wave_id: *b"WAVE",
            fmt_id: *b"fmt ",
            fmt_size: 16,
            audio_format: 1,
            num_channels: 1,
            sample_rate,
            byte_rate: sample_rate * 2,
            block_align: 2,
            bits_per_sample: 16,
            data_id: *b"data",
            data_size,
        }
    }
}

struct AudioRecorder {
    samples: Vec<i16>,
    sample_rate: u32,
    recording: bool,
}

impl AudioRecorder {
    fn new(sample_rate: u32) -> Self {
        AudioRecorder {
            samples: Vec::new(),
            sample_rate,
            recording: false,
        }
    }

    fn get_input_freq(&self) -> AudioInputFrequency {
        match self.sample_rate {
            44100 => AudioInputFrequency::Khz44_1,
            22050 => AudioInputFrequency::Khz22_05,
            _ => AudioInputFrequency::Khz11_025,
        }
    }

    fn start(&mut self) -> i32 {
        let result = unsafe { sceAudioInputInit(0, 1, 0) };
        if result < 0 {
            dprintln!("sceAudioInputInit failed: {:#x}", result);
            return result;
        }

        self.samples.clear();
        self.recording = true;
        dprintln!("Recording started...");
        0
    }

    fn record_chunk(&mut self) {
        if !self.recording {
            return;
        }

        let mut buf = [0i16; 1024];

        unsafe {
            sceAudioInputBlocking(
                buf.len() as i32,
                self.get_input_freq(),
                buf.as_mut_ptr() as *mut c_void,
            );
        }

        self.samples.extend_from_slice(&buf);
    }

    fn stop(&mut self) {
        self.recording = false;
        dprintln!("Recording stopped. {} samples captured.", self.samples.len());
    }

    fn save_wav(&self, path: &[u8]) -> i32 {
        let fd = unsafe {
            sceIoOpen(
                path.as_ptr(),
                IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
                0o644,
            )
        };

        if fd.0 < 0 {
            dprintln!("Open failed: {:#010x} {}", fd.0 as u32, error_name(fd.0));
            return fd.0;
        }

        let data_size = (self.samples.len() * 2) as u32;
        let header = WavHeader::new(self.sample_rate, data_size);

        unsafe {
            sceIoWrite(
                fd,
                &header as *const WavHeader as *const c_void,
                core::mem::size_of::<WavHeader>(),
            );
            sceIoWrite(
                fd,
                self.samples.as_ptr() as *const c_void,
                data_size as usize,
            );
            sceIoClose(fd);
        }

        dprintln!("Saved {} bytes", data_size + 44);
        0
    }
}

// Check if host0: is available by trying to open the root directory
fn is_host0_available() -> bool {
    let fd = unsafe { sceIoDopen(b"host0:/\0".as_ptr()) };
    if fd.0 >= 0 {
        unsafe { sceIoDclose(fd) };
        true
    } else {
        false
    }
}

// Decode PSP error codes to human-readable strings
fn error_name(code: i32) -> &'static str {
    match code as u32 {
        0x80010002 => "ENOENT (no such file/dir)",
        0x80010005 => "EIO (I/O error)",
        0x80010009 => "EBADF (bad file descriptor)",
        0x8001000C => "ENOMEM (out of memory)",
        0x8001000D => "EACCES (permission denied)",
        0x80010010 => "EBUSY (device busy)",
        0x80010011 => "EEXIST (file exists)",
        0x80010014 => "ENODEV (no such device)",
        0x80010015 => "ENOTDIR (not a directory)",
        0x80010016 => "EISDIR (is a directory)",
        0x80010018 => "EINVAL (invalid argument)",
        0x8001001C => "ENOSPC (no space left)",
        0x8001001E => "EROFS (read-only filesystem)",
        0x80010024 => "EMFILE (too many open files)",
        0x80020001 => "ERROR_KERNEL_ERROR",
        _ => "unknown error",
    }
}

// Create the local cache directory (and parents if needed)
fn ensure_cache_dir() {
    // Create parent directories first (0o755 = rwxr-xr-x)
    unsafe {
        sceIoMkdir(b"ms0:/PSP/MUSIC\0".as_ptr(), 0o755);
        sceIoMkdir(LOCAL_CACHE_DIR.as_ptr(), 0o755);
    }
}

// Format a path with a number: prefix + NNN + suffix
fn format_path(buf: &mut [u8], prefix: &[u8], num: u32, suffix: &[u8]) -> usize {
    let mut pos = 0;

    // Copy prefix (without null terminator if present)
    for &b in prefix {
        if b == 0 {
            break;
        }
        buf[pos] = b;
        pos += 1;
    }

    // Format number as 3 digits
    buf[pos] = b'0' + ((num / 100) % 10) as u8;
    buf[pos + 1] = b'0' + ((num / 10) % 10) as u8;
    buf[pos + 2] = b'0' + (num % 10) as u8;
    pos += 3;

    // Copy suffix
    for &b in suffix {
        buf[pos] = b;
        pos += 1;
        if b == 0 {
            break;
        }
    }

    pos
}

// Parse a 3-digit number from a filename like "rec_NNN.wav"
fn parse_file_number(name: &[u8]) -> Option<u32> {
    // Look for pattern: rec_NNN.wav or recording_NNN.wav
    let rec_prefix = b"rec_";
    let recording_prefix = b"recording_";

    let num_start = if name.starts_with(rec_prefix) {
        rec_prefix.len()
    } else if name.starts_with(recording_prefix) {
        recording_prefix.len()
    } else {
        return None;
    };

    // Check we have at least 3 digits + .wav
    if name.len() < num_start + 7 {
        return None;
    }

    let d0 = name[num_start];
    let d1 = name[num_start + 1];
    let d2 = name[num_start + 2];

    if !(d0.is_ascii_digit() && d1.is_ascii_digit() && d2.is_ascii_digit()) {
        return None;
    }

    // Check .wav suffix
    if &name[num_start + 3..num_start + 7] != b".wav" {
        return None;
    }

    Some(((d0 - b'0') as u32) * 100 + ((d1 - b'0') as u32) * 10 + ((d2 - b'0') as u32))
}

// Get the name length (until null terminator)
fn name_len(name: &[u8]) -> usize {
    name.iter().position(|&b| b == 0).unwrap_or(name.len())
}

// Scan cache directory and return (count, highest_number)
fn scan_cache_dir() -> (u32, u32) {
    let fd = unsafe { sceIoDopen(LOCAL_CACHE_DIR.as_ptr()) };
    if fd.0 < 0 {
        return (0, 0);
    }

    let mut count = 0u32;
    let mut highest = 0u32;
    let mut dirent: SceIoDirent = unsafe { core::mem::zeroed() };

    loop {
        let result = unsafe { sceIoDread(fd, &mut dirent) };
        if result <= 0 {
            break;
        }

        let name = &dirent.d_name[..name_len(&dirent.d_name)];
        if let Some(num) = parse_file_number(name) {
            count += 1;
            if num > highest {
                highest = num;
            }
        }
    }

    unsafe { sceIoDclose(fd) };
    (count, highest)
}

// Get next file number for host0:
fn scan_host_dir() -> u32 {
    let fd = unsafe { sceIoDopen(b"host0:/\0".as_ptr()) };
    if fd.0 < 0 {
        return 1;
    }

    let mut highest = 0u32;
    let mut dirent: SceIoDirent = unsafe { core::mem::zeroed() };

    loop {
        let result = unsafe { sceIoDread(fd, &mut dirent) };
        if result <= 0 {
            break;
        }

        let name = &dirent.d_name[..name_len(&dirent.d_name)];
        if let Some(num) = parse_file_number(name) {
            if num > highest {
                highest = num;
            }
        }
    }

    unsafe { sceIoDclose(fd) };
    highest + 1
}

// Copy a file from local cache to host0:
fn upload_file(local_name: &[u8], host_num: u32) -> bool {
    // Build local path
    let mut local_path = [0u8; 64];
    let mut pos = 0;
    for &b in b"ms0:/PSP/MUSIC/AUDREC/" {
        local_path[pos] = b;
        pos += 1;
    }
    for &b in local_name {
        if b == 0 {
            break;
        }
        local_path[pos] = b;
        pos += 1;
    }
    local_path[pos] = 0;

    // Build host path
    let mut host_path = [0u8; 64];
    format_path(&mut host_path, b"host0:/recording_", host_num, b".wav\0");

    // Open source
    let src = unsafe { sceIoOpen(local_path.as_ptr(), IoOpenFlags::RD_ONLY, 0) };
    if src.0 < 0 {
        dprintln!("Open src failed: {:#010x} {}", src.0 as u32, error_name(src.0));
        return false;
    }

    // Open destination
    let dst = unsafe {
        sceIoOpen(
            host_path.as_ptr(),
            IoOpenFlags::WR_ONLY | IoOpenFlags::CREAT | IoOpenFlags::TRUNC,
            0o777,
        )
    };
    if dst.0 < 0 {
        unsafe { sceIoClose(src) };
        dprintln!("Open dst failed: {:#010x} {}", dst.0 as u32, error_name(dst.0));
        return false;
    }

    // Copy in chunks
    let mut buf = [0u8; 4096];
    loop {
        let read = unsafe { sceIoRead(src, buf.as_mut_ptr() as *mut c_void, buf.len() as u32) };
        if read <= 0 {
            break;
        }
        unsafe { sceIoWrite(dst, buf.as_ptr() as *const c_void, read as usize) };
    }

    unsafe {
        sceIoClose(src);
        sceIoClose(dst);
    }

    // Delete local file
    unsafe { sceIoRemove(local_path.as_ptr()) };

    true
}

// Upload all cached files to host0:
fn upload_cached_files() -> u32 {
    let fd = unsafe { sceIoDopen(LOCAL_CACHE_DIR.as_ptr()) };
    if fd.0 < 0 {
        return 0;
    }

    let mut host_num = scan_host_dir();
    let mut uploaded = 0u32;
    let mut dirent: SceIoDirent = unsafe { core::mem::zeroed() };

    // Collect file names first (can't modify dir while iterating)
    let mut files: Vec<[u8; 32]> = Vec::new();

    loop {
        let result = unsafe { sceIoDread(fd, &mut dirent) };
        if result <= 0 {
            break;
        }

        let name = &dirent.d_name[..name_len(&dirent.d_name)];
        if parse_file_number(name).is_some() {
            let mut fname = [0u8; 32];
            for (i, &b) in name.iter().enumerate() {
                if i >= 32 {
                    break;
                }
                fname[i] = b;
            }
            files.push(fname);
        }
    }

    unsafe { sceIoDclose(fd) };

    // Now upload each file
    for fname in files.iter() {
        dprintln!("Uploading to recording_{:03}.wav...", host_num);
        if upload_file(fname, host_num) {
            uploaded += 1;
            host_num += 1;
        }
    }

    uploaded
}

// Wait for a specific button press
fn wait_for_button(target: CtrlButtons) -> CtrlButtons {
    let mut pad = SceCtrlData::default();
    let mut last = CtrlButtons::empty();

    loop {
        unsafe { sceCtrlReadBufferPositive(&mut pad, 1) };
        let pressed = pad.buttons & !last;

        if pressed.contains(target) || pressed.contains(CtrlButtons::CROSS) {
            return pressed & (target | CtrlButtons::CROSS);
        }

        last = pad.buttons;
        unsafe { psp::sys::sceKernelDelayThread(50_000) };
    }
}

extern "C" {
    fn sceCtrlSetSamplingMode(mode: CtrlMode) -> i32;
}

fn psp_main() {
    psp::enable_home_button();
    unsafe { sceCtrlSetSamplingMode(CtrlMode::Digital) };

    dprintln!("PSP Microphone Recorder");
    dprintln!("=======================");

    // Ensure cache directory exists
    ensure_cache_dir();

    // Check connectivity and handle cached files
    let host_available = is_host0_available();
    let mut file_counter: u32;

    if host_available {
        dprintln!("host0: connected");

        let (cached_count, _) = scan_cache_dir();
        if cached_count > 0 {
            dprintln!("");
            dprintln!("Found {} cached recording(s)", cached_count);
            dprintln!("Triangle = Upload, X = Skip");

            let btn = wait_for_button(CtrlButtons::TRIANGLE);
            if btn.contains(CtrlButtons::TRIANGLE) {
                let uploaded = upload_cached_files();
                dprintln!("Uploaded {} file(s)", uploaded);
            } else {
                dprintln!("Skipped upload");
            }
        }

        file_counter = scan_host_dir();
        dprintln!("Next file: recording_{:03}.wav", file_counter);
    } else {
        dprintln!("host0: not available");
        dprintln!("Saving to memory stick");

        let (_, highest) = scan_cache_dir();
        file_counter = highest + 1;
        dprintln!("Next file: rec_{:03}.wav", file_counter);
    }

    dprintln!("");
    dprintln!("O = Start, X = Stop & Save");
    dprintln!("");

    let mut recorder = AudioRecorder::new(44100);
    let mut pad = SceCtrlData::default();
    let mut last_buttons = CtrlButtons::empty();

    loop {
        unsafe { sceCtrlReadBufferPositive(&mut pad, 1) };
        let pressed = pad.buttons & !last_buttons;

        if pressed.contains(CtrlButtons::CIRCLE) && !recorder.recording {
            recorder.start();
        }

        if pressed.contains(CtrlButtons::CROSS) && recorder.recording {
            recorder.stop();

            // Try host0: first, fall back to local
            let mut path = [0u8; 64];

            if is_host0_available() {
                format_path(&mut path, b"host0:/recording_", file_counter, b".wav\0");
                dprintln!("Saving to host0:/recording_{:03}.wav", file_counter);
            } else {
                format_path(&mut path, b"ms0:/PSP/MUSIC/AUDREC/rec_", file_counter, b".wav\0");
                dprintln!("Saving to ms0:/.../rec_{:03}.wav", file_counter);
            }

            let result = recorder.save_wav(&path);
            if result >= 0 {
                file_counter += 1;
            } else {
                dprintln!("Save failed: {:#010x} {}", result as u32, error_name(result));
            }
        }

        if recorder.recording {
            recorder.record_chunk();
        }

        last_buttons = pad.buttons;

        if !recorder.recording {
            unsafe { psp::sys::sceKernelDelayThread(50_000) };
        }
    }
}
