//! Diagnostic loader: run a shell command and print EVERY psplink event raw
//! (stdout/stderr/shell) with timestamps, no marker interpretation.
//! Usage: dbg_run <host0_dir> <host1_dir> <cmd> [args...]

use psplink_connection::{PSPConnection, PspEvent};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (host0, host1, cmd) = (&args[0], &args[1], &args[2..]);
    let conn = PSPConnection::connect(host0.as_ref(), host1.as_ref(), Default::default())
        .expect("connect failed");
    let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    conn.send_shell_command(&cmd_refs).expect("send failed");
    let start = Instant::now();
    loop {
        match conn.recv_event(Duration::from_secs(60)) {
            Ok(PspEvent::Stdout(b)) => {
                println!("[{:7.2}s stdout] {}", start.elapsed().as_secs_f32(), String::from_utf8_lossy(&b))
            }
            Ok(PspEvent::Stderr(b)) => {
                println!("[{:7.2}s stderr] {}", start.elapsed().as_secs_f32(), String::from_utf8_lossy(&b))
            }
            Ok(PspEvent::ShellRaw(b)) => {
                println!("[{:7.2}s shell ] {:?}", start.elapsed().as_secs_f32(), String::from_utf8_lossy(&b))
            }
            Ok(PspEvent::Disconnected) => {
                println!("[{:7.2}s] DISCONNECTED", start.elapsed().as_secs_f32());
                break;
            }
            Err(e) => {
                println!("[{:7.2}s] recv error/timeout: {e}", start.elapsed().as_secs_f32());
                break;
            }
        }
    }
    conn.disconnect();
}
