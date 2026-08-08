//! Send psplink's `reset` shell command — reboots psplink, reclaiming all
//! leaked partition memory. Usage: cargo run -p psplink-connection --example reset

use psplink_connection::PSPConnection;

fn main() {
    let tmp = std::env::temp_dir();
    let conn = PSPConnection::connect(&tmp, &tmp, Default::default()).expect("connect failed");
    conn.send_shell_command(&["reset"]).expect("reset failed");
    eprintln!("reset sent; waiting for psplink to reboot...");
    std::thread::sleep(std::time::Duration::from_secs(3));
    conn.disconnect();
}
