/// Build a "load and start module" shell command.
///
/// psplink shell protocol: NUL-separated args terminated by 0x01.
/// Wire format: `ld\0host0:/path/to/module.prx\0\x01`
pub fn build_load_command(prx_path: &str) -> Vec<u8> {
    let mut cmd = Vec::new();
    cmd.extend_from_slice(b"ld\0");
    cmd.extend_from_slice(prx_path.as_bytes());
    cmd.push(0x00); // NUL-terminate path arg
    cmd.push(0x01); // command terminator
    cmd
}
