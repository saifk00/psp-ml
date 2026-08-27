//! vme-dump -- print a VME machine image's buffers as hex words.
//!
//!     cargo run -p psp-tc --bin vme-dump <image.bin>
//!
//! Dumps all eight ring buffers (TOP_0..3, BASE_0..3) from a 1 MB machine
//! image (vme-emu's input or output format), eight words per row with word
//! offsets, eliding all-zero runs the way hexdump does.  Also prints
//! DMA_STAT when the image carries one (output images do).

use std::process::exit;
use vme_assembler::{Buffer, MachineImage};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: vme-dump <image.bin>");
        exit(2);
    }
    let img = MachineImage::from_file(&args[1]).unwrap_or_else(|e| {
        eprintln!("vme-dump: {}: {e}", args[1]);
        exit(2);
    });

    for b in [
        Buffer::Top0,
        Buffer::Top1,
        Buffer::Top2,
        Buffer::Top3,
        Buffer::Base0,
        Buffer::Base1,
        Buffer::Base2,
        Buffer::Base3,
    ] {
        dump_buffer(b, &img.read_buffer(b));
    }

    let stat = img.word(0xFF000);
    if stat != 0 {
        println!(
            "DMA_STAT {stat:#010x}  (VD={} TD={})",
            (stat >> 11) & 1,
            (stat >> 9) & 1
        );
    }
}

fn dump_buffer(b: Buffer, words: &[i32]) {
    let nonzero = words.iter().filter(|w| **w != 0).count();
    println!("{:?}  ({} of {} words non-zero)", b, nonzero, words.len());

    let mut eliding = false;
    for (row_idx, row) in words.chunks(8).enumerate() {
        if row.iter().all(|w| *w == 0) {
            if !eliding {
                println!("  *");
                eliding = true;
            }
            continue;
        }
        eliding = false;
        let mut line = format!("  {:#06x}:", row_idx * 8);
        for w in row {
            line.push_str(&format!(" {:08x}", *w as u32));
        }
        println!("{line}");
    }
    println!();
}
