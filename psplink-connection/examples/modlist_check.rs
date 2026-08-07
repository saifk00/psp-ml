//! Ad-hoc hardware check: connect, send `modlist`, print the raw shell
//! response. Used to confirm psp_ml::module!'s exit-status fix doesn't
//! leak a resident module per `ld` call. Not a permanent fixture.

use psplink_connection::PSPConnection;
use std::path::Path;
use std::time::Duration;

fn main() {
    let conn = PSPConnection::connect(
        Path::new("psp-ml"),
        Path::new("psp-ml/target/mipsel-sony-psp/debug"),
        Default::default(),
    )
    .expect("connect failed");

    conn.send_shell_command(&["modlist"]).expect("send failed");

    // load_program owns the nice framer; for this ad-hoc check just dump
    // raw shell bytes for a couple seconds and print them as text.
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut all = Vec::new();
    while std::time::Instant::now() < deadline {
        if let Ok(event) = conn.recv_event(Duration::from_millis(500)) {
            if let psplink_connection::PspEvent::ShellRaw(bytes) = event {
                all.extend_from_slice(&bytes);
            }
        }
    }

    let text = String::from_utf8_lossy(&all);
    for line in text.split(['\r', '\n', '\u{ff}', '\u{fe}']) {
        let line = line.trim();
        if line.contains("hello_psp") || line.contains("Module List") {
            println!("{line}");
        }
    }

    conn.disconnect();
}
