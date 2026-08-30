//! Manual hardware checkpoint for C Stage A (see the implementation plan):
//! connects to a real PSP over USB and pumps the read loop a few times,
//! confirming the HOSTFS_CMD_HELLO handshake completes. Not wired into
//! any example's `host/` crate — this is a standalone smoke test for the
//! vendored C.
//!
//! Usage: cargo run --example connect_check -p usbhostfs-sys -- <host0_dir> <host1_dir>

use std::env;
use std::ffi::CString;
use std::os::raw::c_void;

extern "C" fn on_async(
    _user: *mut c_void,
    channel: std::os::raw::c_int,
    data: *const u8,
    len: std::os::raw::c_int,
) {
    let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) };
    eprintln!("[async chan={channel}] {:?}", String::from_utf8_lossy(bytes));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let host0 = args.get(1).cloned().unwrap_or_else(|| ".".to_string());
    let host1 = args.get(2).cloned().unwrap_or_else(|| ".".to_string());

    unsafe {
        let ctx = usbhostfs_sys::uhfs_ctx_new();
        assert!(!ctx.is_null(), "uhfs_ctx_new failed");

        let host0_c = CString::new(host0.as_str()).unwrap();
        let host1_c = CString::new(host1.as_str()).unwrap();
        let r0 = usbhostfs_sys::uhfs_add_drive(ctx, 0, host0_c.as_ptr());
        let r1 = usbhostfs_sys::uhfs_add_drive(ctx, 1, host1_c.as_ptr());
        assert_eq!(r0, 0, "uhfs_add_drive(0, {host0}) failed");
        assert_eq!(r1, 0, "uhfs_add_drive(1, {host1}) failed");

        usbhostfs_sys::uhfs_set_async_callback(ctx, on_async, std::ptr::null_mut());

        eprintln!("connecting...");
        let r = usbhostfs_sys::uhfs_connect(ctx);
        assert_eq!(r, 0, "uhfs_connect failed");
        eprintln!("connected, pumping for the HOSTFS_CMD_HELLO handshake...");

        let mut processed = 0;
        for _ in 0..50 {
            let ret = usbhostfs_sys::uhfs_pump(ctx, 200);
            if ret > 0 {
                processed += 1;
                eprintln!("pump processed a command (count={processed})");
            } else if ret < 0 {
                eprintln!("pump reported disconnect");
                break;
            }
        }

        eprintln!("done, processed {processed} command(s)");
        usbhostfs_sys::uhfs_disconnect(ctx);
        usbhostfs_sys::uhfs_ctx_free(ctx);
    }
}
