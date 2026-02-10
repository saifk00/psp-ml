/// Magic numbers — first 4 bytes of every USB packet identify the protocol layer.
pub const HOSTFS_MAGIC: u32 = 0x782F_0812;
pub const ASYNC_MAGIC: u32 = 0x782F_0813;
pub const BULK_MAGIC: u32 = 0x782F_0814;

/// HostFS command IDs (from usbhostfs.h).
pub const HELLO_CMD: u32 = 0x8FFC_0000;
pub const BYE_CMD: u32 = 0x8FFC_0001;
pub const OPEN_CMD: u32 = 0x8FFC_0002;
pub const CLOSE_CMD: u32 = 0x8FFC_0003;
pub const READ_CMD: u32 = 0x8FFC_0004;
pub const WRITE_CMD: u32 = 0x8FFC_0005;
pub const LSEEK_CMD: u32 = 0x8FFC_0006;
pub const REMOVE_CMD: u32 = 0x8FFC_0007;
pub const MKDIR_CMD: u32 = 0x8FFC_0008;
pub const RMDIR_CMD: u32 = 0x8FFC_0009;
pub const DOPEN_CMD: u32 = 0x8FFC_000A;
pub const DREAD_CMD: u32 = 0x8FFC_000B;
pub const DCLOSE_CMD: u32 = 0x8FFC_000C;
pub const GETSTAT_CMD: u32 = 0x8FFC_000D;
pub const CHSTAT_CMD: u32 = 0x8FFC_000E;
pub const RENAME_CMD: u32 = 0x8FFC_000F;
pub const CHDIR_CMD: u32 = 0x8FFC_0010;
pub const IOCTL_CMD: u32 = 0x8FFC_0011;
pub const DEVCTL_CMD: u32 = 0x8FFC_0012;

/// Async channel IDs.
pub const ASYNC_SHELL: u32 = 0;
pub const ASYNC_GDB: u32 = 1;
pub const ASYNC_STDOUT: u32 = 2;
pub const ASYNC_STDERR: u32 = 3;

/// Shell response framing bytes.
pub const SHELL_BEGIN: u8 = 0xFF;
pub const SHELL_END: u8 = 0xFE;

/// PSP file open flags (from psp_fileio.h).
pub const PSP_O_RDONLY: u32 = 0x0001;
pub const PSP_O_WRONLY: u32 = 0x0002;
pub const PSP_O_RDWR: u32 = 0x0003;
pub const PSP_O_NBLOCK: u32 = 0x0004;
pub const PSP_O_DIROPEN: u32 = 0x0008;
pub const PSP_O_APPEND: u32 = 0x0100;
pub const PSP_O_CREAT: u32 = 0x0200;
pub const PSP_O_TRUNC: u32 = 0x0400;
pub const PSP_O_EXCL: u32 = 0x0800;
pub const PSP_O_NOWAIT: u32 = 0x8000;

/// Maximum bytes per HostFS read/write transfer.
pub const HOSTFS_MAX_BLOCK: usize = 64 * 1024;

// ---------------------------------------------------------------------------
// Wire serialization helpers (all little-endian)
// ---------------------------------------------------------------------------

#[inline]
pub fn read_u32_le(buf: &[u8], offset: usize) -> Option<u32> {
    let b = buf.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline]
pub fn read_i32_le(buf: &[u8], offset: usize) -> Option<i32> {
    let b = buf.get(offset..offset + 4)?;
    Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline]
pub fn read_i64_le(buf: &[u8], offset: usize) -> Option<i64> {
    let b = buf.get(offset..offset + 8)?;
    Some(i64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

#[inline]
pub fn write_u32_le(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

#[inline]
pub fn write_i32_le(buf: &mut Vec<u8>, val: i32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

#[inline]
pub fn write_i64_le(buf: &mut Vec<u8>, val: i64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Build a 12-byte HostFS header.
pub fn hostfs_header(command: u32, extra_len: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(12);
    write_u32_le(&mut buf, HOSTFS_MAGIC);
    write_u32_le(&mut buf, command);
    write_u32_le(&mut buf, extra_len);
    buf
}

// ---------------------------------------------------------------------------
// Packet classification
// ---------------------------------------------------------------------------

pub enum PacketKind {
    HostFs,
    Async,
    Bulk,
    Unknown(u32),
}

pub fn classify_packet(buf: &[u8]) -> PacketKind {
    match read_u32_le(buf, 0) {
        Some(HOSTFS_MAGIC) => PacketKind::HostFs,
        Some(ASYNC_MAGIC) => PacketKind::Async,
        Some(BULK_MAGIC) => PacketKind::Bulk,
        Some(other) => PacketKind::Unknown(other),
        None => PacketKind::Unknown(0),
    }
}

/// Return a human-readable name for a HostFS command ID.
pub fn cmd_name(cmd: u32) -> &'static str {
    match cmd {
        HELLO_CMD => "HELLO",
        BYE_CMD => "BYE",
        OPEN_CMD => "OPEN",
        CLOSE_CMD => "CLOSE",
        READ_CMD => "READ",
        WRITE_CMD => "WRITE",
        LSEEK_CMD => "LSEEK",
        REMOVE_CMD => "REMOVE",
        MKDIR_CMD => "MKDIR",
        RMDIR_CMD => "RMDIR",
        DOPEN_CMD => "DOPEN",
        DREAD_CMD => "DREAD",
        DCLOSE_CMD => "DCLOSE",
        GETSTAT_CMD => "GETSTAT",
        CHSTAT_CMD => "CHSTAT",
        RENAME_CMD => "RENAME",
        CHDIR_CMD => "CHDIR",
        IOCTL_CMD => "IOCTL",
        DEVCTL_CMD => "DEVCTL",
        _ => "UNKNOWN",
    }
}
