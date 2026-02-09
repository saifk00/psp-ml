use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::mpsc;

use crate::protocol::{self, *};

/// Virtual filename used by `psp_ml::exit()` to signal program completion.
const EXIT_SENTINEL: &str = "__psp_exit";

/// Manages open file/directory descriptors and handles HostFS commands from the PSP.
pub struct FdTable {
    root: PathBuf,
    next_fd: i32,
    files: HashMap<i32, File>,
    dirs: HashMap<i32, DirState>,
    /// Tracks which fds were opened with write flags.
    write_fds: HashSet<i32>,
    /// Maps fd -> path (for write notifications on close).
    fd_paths: HashMap<i32, PathBuf>,
    /// Sends the path of each file that was written then closed.
    written_files_tx: Option<mpsc::Sender<PathBuf>>,
}

struct DirState {
    entries: Vec<fs::DirEntry>,
    pos: usize,
}

impl FdTable {
    pub fn new(root: PathBuf, written_files_tx: Option<mpsc::Sender<PathBuf>>) -> Self {
        FdTable {
            root,
            next_fd: 1, // start at 1 so 0 is never a valid fd
            files: HashMap::new(),
            dirs: HashMap::new(),
            write_fds: HashSet::new(),
            fd_paths: HashMap::new(),
            written_files_tx,
        }
    }

    /// Resolve a PSP path (e.g. `/benchmarks.json`) relative to the host root directory.
    /// The PSP sends paths with the `host0:/` prefix already stripped by the kernel driver.
    fn resolve_path(&self, psp_path: &str) -> PathBuf {
        let stripped = psp_path
            .strip_prefix('/')
            .unwrap_or(psp_path);
        self.root.join(stripped)
    }

    fn alloc_fd(&mut self) -> i32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        fd
    }

    // -----------------------------------------------------------------------
    // Top-level dispatch
    // -----------------------------------------------------------------------

    /// Process a raw HostFS command packet and return the response to send on EP2.
    ///
    /// Returns `(header, extra)` — caller must send as **separate** USB writes since the
    /// PSP reads them as distinct bulk transfers via `sceUsbbdReqRecv`.
    ///
    /// `extra_data` is the trailing data read separately after the command struct
    /// (filenames, write payloads, etc).
    pub fn handle_command(&mut self, cmd_buf: &[u8], extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let cmd_id = protocol::read_u32_le(cmd_buf, 4);
        log::debug!(
            "HostFS {} (extra {} bytes)",
            cmd_name(cmd_id),
            extra_data.len()
        );

        match cmd_id {
            HELLO_CMD => self.handle_hello(cmd_buf),
            OPEN_CMD => self.handle_open(cmd_buf, extra_data),
            CLOSE_CMD => self.handle_close(cmd_buf),
            READ_CMD => self.handle_read(cmd_buf),
            WRITE_CMD => self.handle_write(cmd_buf, extra_data),
            LSEEK_CMD => self.handle_lseek(cmd_buf),
            REMOVE_CMD => self.handle_remove(extra_data),
            MKDIR_CMD => self.handle_mkdir(cmd_buf, extra_data),
            RMDIR_CMD => self.handle_rmdir(extra_data),
            DOPEN_CMD => self.handle_dopen(extra_data),
            DREAD_CMD => self.handle_dread(cmd_buf),
            DCLOSE_CMD => self.handle_dclose(cmd_buf),
            GETSTAT_CMD => self.handle_getstat(extra_data),
            CHDIR_CMD => self.handle_chdir(extra_data),
            _ => {
                log::warn!("unhandled HostFS command {:#010X} ({})", cmd_id, cmd_name(cmd_id));
                self.simple_response(cmd_id, -1)
            }
        }
    }

    // -----------------------------------------------------------------------
    // HELLO (0x8FFC0000)
    // -----------------------------------------------------------------------

    fn handle_hello(&self, _cmd_buf: &[u8]) -> (Vec<u8>, Vec<u8>) {
        log::info!("HELLO — handshake from PSP");
        (hostfs_header(HELLO_CMD, 0), Vec::new())
    }

    // -----------------------------------------------------------------------
    // OPEN (0x8FFC0002)
    // Cmd: header(12) + mode(4) + mask(4) + fsnum(4) = 24 bytes
    // Extra: NUL-terminated filename
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_open(&mut self, cmd_buf: &[u8], extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mode = protocol::read_u32_le(cmd_buf, 12);
        let _mask = protocol::read_u32_le(cmd_buf, 16);
        let _fsnum = protocol::read_u32_le(cmd_buf, 20);

        let filename = extract_cstring(extra_data);

        // Virtual file: exit sentinel — signal the runner without touching disk
        if filename.trim_start_matches('/') == EXIT_SENTINEL {
            log::info!("OPEN {} — exit sentinel, signaling runner", filename);
            if let Some(tx) = &self.written_files_tx {
                let _ = tx.send(PathBuf::from(EXIT_SENTINEL));
            }
            // Return a fake fd — the PSP will close it, which is a harmless no-op
            let fd = self.alloc_fd();
            return self.simple_response(OPEN_CMD, fd);
        }

        let local_path = self.resolve_path(&filename);

        log::info!("OPEN {} (mode {:#06x}) -> {}", filename, mode, local_path.display());

        let mut opts = OpenOptions::new();
        let is_write = (mode & PSP_O_WRONLY) != 0 || (mode & PSP_O_RDWR) == PSP_O_RDWR;
        let is_read = (mode & PSP_O_RDONLY) != 0 || (mode & PSP_O_RDWR) == PSP_O_RDWR || !is_write;
        if is_read {
            opts.read(true);
        }
        if is_write {
            opts.write(true);
        }
        if (mode & PSP_O_CREAT) != 0 {
            opts.create(true);
        }
        if (mode & PSP_O_TRUNC) != 0 {
            opts.truncate(true);
        }
        if (mode & PSP_O_APPEND) != 0 {
            opts.append(true);
        }

        // Ensure parent directory exists for create operations
        if (mode & PSP_O_CREAT) != 0 {
            if let Some(parent) = local_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
        }

        match opts.open(&local_path) {
            Ok(file) => {
                let fd = self.alloc_fd();
                self.files.insert(fd, file);
                self.fd_paths.insert(fd, local_path);
                if is_write {
                    self.write_fds.insert(fd);
                }
                log::debug!("  -> fd {}", fd);
                self.simple_response(OPEN_CMD, fd)
            }
            Err(e) => {
                log::warn!("  OPEN failed: {}", e);
                self.simple_response(OPEN_CMD, -1)
            }
        }
    }

    // -----------------------------------------------------------------------
    // CLOSE (0x8FFC0003)
    // Cmd: header(12) + fd(4) = 16 bytes
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_close(&mut self, cmd_buf: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let fd = protocol::read_i32_le(cmd_buf, 12);

        let result = if self.files.remove(&fd).is_some() {
            // If this fd was opened for writing, notify the runner
            if self.write_fds.remove(&fd) {
                if let Some(path) = self.fd_paths.get(&fd) {
                    log::info!("CLOSE fd {} (wrote {})", fd, path.display());
                    if let Some(tx) = &self.written_files_tx {
                        let _ = tx.send(path.clone());
                    }
                }
            } else {
                log::debug!("CLOSE fd {}", fd);
            }
            self.fd_paths.remove(&fd);
            0
        } else if self.dirs.remove(&fd).is_some() {
            log::debug!("DCLOSE fd {}", fd);
            0
        } else {
            log::warn!("CLOSE bad fd {}", fd);
            -1
        };

        self.simple_response(CLOSE_CMD, result)
    }

    // -----------------------------------------------------------------------
    // READ (0x8FFC0004)
    // Cmd: header(12) + fd(4) + len(4) = 20 bytes
    // Resp: header(12, extra_len=bytes_read) + result(4) + [data]
    // -----------------------------------------------------------------------

    fn handle_read(&mut self, cmd_buf: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let fd = protocol::read_i32_le(cmd_buf, 12);
        let requested = protocol::read_i32_le(cmd_buf, 16) as usize;
        let read_len = requested.min(HOSTFS_MAX_BLOCK);

        let mut data = vec![0u8; read_len];
        let result = match self.files.get_mut(&fd) {
            Some(file) => match file.read(&mut data) {
                Ok(n) => {
                    data.truncate(n);
                    n as i32
                }
                Err(e) => {
                    log::warn!("READ fd {} error: {}", fd, e);
                    data.clear();
                    -1
                }
            },
            None => {
                log::warn!("READ bad fd {}", fd);
                data.clear();
                -1
            }
        };

        log::debug!("READ fd {} requested {} -> {}", fd, requested, result);

        // Header (16 bytes): sent as first USB write
        let mut hdr = hostfs_header(READ_CMD, data.len() as u32);
        write_i32_le(&mut hdr, result);
        // Extra (file data): sent as second USB write
        (hdr, data)
    }

    // -----------------------------------------------------------------------
    // WRITE (0x8FFC0005)
    // Cmd: header(12, extra_len=data_len) + fd(4) = 16 bytes, then [data]
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_write(&mut self, cmd_buf: &[u8], extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let fd = protocol::read_i32_le(cmd_buf, 12);

        let result = match self.files.get_mut(&fd) {
            Some(file) => match file.write_all(extra_data) {
                Ok(()) => extra_data.len() as i32,
                Err(e) => {
                    log::warn!("WRITE fd {} error: {}", fd, e);
                    -1
                }
            },
            None => {
                log::warn!("WRITE bad fd {}", fd);
                -1
            }
        };

        log::debug!("WRITE fd {} {} bytes -> {}", fd, extra_data.len(), result);
        self.simple_response(WRITE_CMD, result)
    }

    // -----------------------------------------------------------------------
    // LSEEK (0x8FFC0006)
    // Cmd: header(12) + fd(4) + offset(8) + whence(4) = 28 bytes
    // Resp: header(12) + result(4) + pos(8) = 24 bytes
    // -----------------------------------------------------------------------

    fn handle_lseek(&mut self, cmd_buf: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let fd = protocol::read_i32_le(cmd_buf, 12);
        let offset = protocol::read_i64_le(cmd_buf, 16);
        let whence = protocol::read_i32_le(cmd_buf, 24);

        let seek_from = match whence {
            0 => SeekFrom::Start(offset as u64),
            1 => SeekFrom::Current(offset),
            2 => SeekFrom::End(offset),
            _ => {
                log::warn!("LSEEK fd {} bad whence {}", fd, whence);
                return self.lseek_response(-1, 0);
            }
        };

        match self.files.get_mut(&fd) {
            Some(file) => match file.seek(seek_from) {
                Ok(pos) => {
                    log::debug!("LSEEK fd {} -> pos {}", fd, pos);
                    self.lseek_response(0, pos as i64)
                }
                Err(e) => {
                    log::warn!("LSEEK fd {} error: {}", fd, e);
                    self.lseek_response(-1, 0)
                }
            },
            None => {
                log::warn!("LSEEK bad fd {}", fd);
                self.lseek_response(-1, 0)
            }
        }
    }

    fn lseek_response(&self, result: i32, pos: i64) -> (Vec<u8>, Vec<u8>) {
        let mut resp = Vec::with_capacity(24);
        let hdr = hostfs_header(LSEEK_CMD, 0);
        resp.extend_from_slice(&hdr);
        write_i32_le(&mut resp, result);
        write_i64_le(&mut resp, pos);
        (resp, Vec::new())
    }

    // -----------------------------------------------------------------------
    // REMOVE (0x8FFC0007)
    // Cmd: header(12) + fsnum(4) = 16 bytes, extra = NUL-terminated filename
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_remove(&mut self, extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let filename = extract_cstring(extra_data);
        let local_path = self.resolve_path(&filename);
        log::info!("REMOVE {}", local_path.display());

        let result = match fs::remove_file(&local_path) {
            Ok(()) => 0,
            Err(e) => {
                log::warn!("REMOVE failed: {}", e);
                -1
            }
        };
        self.simple_response(REMOVE_CMD, result)
    }

    // -----------------------------------------------------------------------
    // MKDIR (0x8FFC0008)
    // Cmd: header(12) + mode(4) + fsnum(4) = 20 bytes, extra = dirname
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_mkdir(&mut self, _cmd_buf: &[u8], extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let dirname = extract_cstring(extra_data);
        let local_path = self.resolve_path(&dirname);
        log::info!("MKDIR {}", local_path.display());

        let result = match fs::create_dir_all(&local_path) {
            Ok(()) => 0,
            Err(e) => {
                log::warn!("MKDIR failed: {}", e);
                -1
            }
        };
        self.simple_response(MKDIR_CMD, result)
    }

    // -----------------------------------------------------------------------
    // RMDIR (0x8FFC0009)
    // Cmd: header(12) + fsnum(4) = 16 bytes, extra = dirname
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_rmdir(&mut self, extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let dirname = extract_cstring(extra_data);
        let local_path = self.resolve_path(&dirname);
        log::info!("RMDIR {}", local_path.display());

        let result = match fs::remove_dir(&local_path) {
            Ok(()) => 0,
            Err(e) => {
                log::warn!("RMDIR failed: {}", e);
                -1
            }
        };
        self.simple_response(RMDIR_CMD, result)
    }

    // -----------------------------------------------------------------------
    // DOPEN (0x8FFC000A)
    // Cmd: header(12) + fsnum(4) = 16 bytes, extra = dirname
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_dopen(&mut self, extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let dirname = extract_cstring(extra_data);
        let local_path = self.resolve_path(&dirname);
        log::info!("DOPEN {}", local_path.display());

        match fs::read_dir(&local_path) {
            Ok(read_dir) => {
                let entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
                let fd = self.alloc_fd();
                self.dirs.insert(fd, DirState { entries, pos: 0 });
                log::debug!("  -> dfd {}", fd);
                self.simple_response(DOPEN_CMD, fd)
            }
            Err(e) => {
                log::warn!("DOPEN failed: {}", e);
                self.simple_response(DOPEN_CMD, -1)
            }
        }
    }

    // -----------------------------------------------------------------------
    // DREAD (0x8FFC000B)
    // Cmd: header(12) + fd(4) = 16 bytes
    // Resp: header(12, extra_len=dirent_size) + result(4), extra = SceIoDirent
    // -----------------------------------------------------------------------

    fn handle_dread(&mut self, cmd_buf: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let fd = protocol::read_i32_le(cmd_buf, 12);

        match self.dirs.get_mut(&fd) {
            Some(dir_state) => {
                if dir_state.pos < dir_state.entries.len() {
                    let entry = &dir_state.entries[dir_state.pos];
                    dir_state.pos += 1;

                    let dirent = build_sce_dirent(entry);
                    let mut hdr = hostfs_header(DREAD_CMD, dirent.len() as u32);
                    write_i32_le(&mut hdr, 1); // 1 = entry available
                    (hdr, dirent)
                } else {
                    // No more entries
                    self.simple_response(DREAD_CMD, 0)
                }
            }
            None => {
                log::warn!("DREAD bad fd {}", fd);
                self.simple_response(DREAD_CMD, -1)
            }
        }
    }

    // -----------------------------------------------------------------------
    // DCLOSE (0x8FFC000C)
    // Cmd: header(12) + fd(4) = 16 bytes
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_dclose(&mut self, cmd_buf: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let fd = protocol::read_i32_le(cmd_buf, 12);
        let result = if self.dirs.remove(&fd).is_some() {
            log::debug!("DCLOSE fd {}", fd);
            0
        } else {
            log::warn!("DCLOSE bad fd {}", fd);
            -1
        };
        self.simple_response(DCLOSE_CMD, result)
    }

    // -----------------------------------------------------------------------
    // GETSTAT (0x8FFC000D)
    // Cmd: header(12) + fsnum(4) = 16 bytes, extra = filename
    // Resp: header(12, extra_len=stat_size) + result(4), extra = SceIoStat
    // -----------------------------------------------------------------------

    fn handle_getstat(&mut self, extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let filename = extract_cstring(extra_data);
        let local_path = self.resolve_path(&filename);
        log::debug!("GETSTAT {}", local_path.display());

        match fs::metadata(&local_path) {
            Ok(meta) => {
                let stat = build_sce_stat(&meta);
                let mut hdr = hostfs_header(GETSTAT_CMD, stat.len() as u32);
                write_i32_le(&mut hdr, 0); // success
                (hdr, stat)
            }
            Err(e) => {
                log::warn!("GETSTAT {} failed: {}", local_path.display(), e);
                self.simple_response(GETSTAT_CMD, -1)
            }
        }
    }

    // -----------------------------------------------------------------------
    // CHDIR (0x8FFC0010)
    // Cmd: header(12) + fsnum(4) = 16 bytes, extra = path
    // Resp: header(12) + result(4) = 16 bytes
    // -----------------------------------------------------------------------

    fn handle_chdir(&mut self, extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let path = extract_cstring(extra_data);
        log::info!("CHDIR {} (ignored — virtual cwd not tracked)", path);
        // Always succeed — we resolve all paths against root anyway
        self.simple_response(CHDIR_CMD, 0)
    }

    // -----------------------------------------------------------------------
    // Response builders
    // -----------------------------------------------------------------------

    /// Build a standard 16-byte response: header(12) + result(4), no extra data.
    fn simple_response(&self, cmd: u32, result: i32) -> (Vec<u8>, Vec<u8>) {
        let mut resp = Vec::with_capacity(16);
        let hdr = hostfs_header(cmd, 0);
        resp.extend_from_slice(&hdr);
        write_i32_le(&mut resp, result);
        (resp, Vec::new())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a NUL-terminated C string from a byte slice.
fn extract_cstring(data: &[u8]) -> String {
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    String::from_utf8_lossy(&data[..end]).into_owned()
}

/// Build a minimal SceIoStat structure (96 bytes) from host file metadata.
///
/// Layout (all LE):
///   [0..4]   mode   (u32)
///   [4..8]   attr   (u32)
///   [8..16]  size   (i64)
///   [16..30] ctime  (ScePspDateTime, 14 bytes: 6×u16 + u32 unused)
///   [30..44] atime  (ScePspDateTime)
///   [44..58] mtime  (ScePspDateTime)
///   [58..82] private (6×u32)
///   Total: 82 bytes (padded to 96 with zeros for safety)
fn build_sce_stat(meta: &fs::Metadata) -> Vec<u8> {
    let mut buf = vec![0u8; 96];

    // Mode: set directory/regular bits
    let mut mode: u32 = 0;
    if meta.is_dir() {
        mode |= 0x1000; // FIO_S_IFDIR
        mode |= 0x0100; // FIO_S_IRUSR
        mode |= 0x0020; // FIO_S_IXUSR
    } else {
        mode |= 0x2000; // FIO_S_IFREG
        mode |= 0x0100; // FIO_S_IRUSR
        mode |= 0x0080; // FIO_S_IWUSR
    }
    buf[0..4].copy_from_slice(&mode.to_le_bytes());

    // Size
    let size = meta.len() as i64;
    buf[8..16].copy_from_slice(&size.to_le_bytes());

    buf
}

/// Build a minimal SceIoDirent structure (~360 bytes) for a directory entry.
///
/// Layout:
///   [0..96]    SceIoStat
///   [96..352]  name (char[256])
///   [352..356] private (u32)
///   [356..360] dummy (u32)
fn build_sce_dirent(entry: &fs::DirEntry) -> Vec<u8> {
    let mut buf = vec![0u8; 360];

    // Fill in the stat portion
    if let Ok(meta) = entry.metadata() {
        let stat = build_sce_stat(&meta);
        buf[..stat.len()].copy_from_slice(&stat);
    }

    // Fill in the name (max 255 chars + NUL)
    let name = entry.file_name();
    let name_bytes = name.to_string_lossy();
    let name_bytes = name_bytes.as_bytes();
    let copy_len = name_bytes.len().min(255);
    buf[96..96 + copy_len].copy_from_slice(&name_bytes[..copy_len]);
    // Already NUL-terminated by the zeroed buffer

    buf
}
