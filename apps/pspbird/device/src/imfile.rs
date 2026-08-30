//! `PBIM`: the species-image pack the PSPBird app draws from.
//!
//! One pack serves every region. Images are stored once, in sorted label
//! order, and the file carries a *region map* per classifier blob: a
//! `u32[n_classes]` table from that blob's class index to an image index
//! (or `NO_IMAGE`). The app loads the map for the selected region once
//! (2 KB), and from then on a top-k class index becomes pixels with one
//! seek and one read -- no name lookup on device. The index-to-name table
//! is still in the file so a viewer can tell what it is looking at.
//!
//! Layout, little-endian:
//!
//! ```text
//!   0   magic       b"PBIM"
//!   4   version     u32 = 2
//!   8   n           u32  image count
//!   12  w           u32  image width, pixels
//!   16  h           u32  image height, pixels
//!   20  names_off   u32  byte offset of the name table
//!   24  names_len   u32  byte length of the name table
//!   28  regions_off u32  byte offset of the region maps
//!   32  regions_len u32  byte length of the region maps
//!   36  n_regions   u32
//!   40  image_off   u32  byte offset of image 0, 16-byte aligned
//!   44  reserved    u32
//!   names_off:      u32 ends[n]   end offset of name i within the pool
//!                   pool          UTF-8, names back to back
//!   regions_off:    (4-aligned) n_regions × { u32 name_len, name (padded to 4),
//!                                 u32 n_classes, u32 map[n_classes] }
//!   image_off:      n × (w × h) u16 RGB565, row-major, no padding
//! ```
//!
//! Images are fixed-size raw RGB565 -- the PSP's native 16-bit display
//! format, half the bytes of 8888, and decodable with a shift and an or
//! per pixel, which matters more than compression when the alternative
//! is a JPEG decoder on a 333 MHz MIPS core. Image *i* is at
//! `image_off + i * w * h * 2`.
//!
//! The reader half is `no_std` and used on device (`device::Pack` wraps
//! the `sceIo` calls); `pack_images` (behind the `imfile-pack` feature,
//! host only) builds a pack from a `manifest.toml` of `"label" =
//! "path.jpg"` entries, as written by `examples/birdnet/fetch_images.py`,
//! plus each region's label list.

pub const MAGIC: &[u8; 4] = b"PBIM";
pub const VERSION: u32 = 2;
pub const HEADER_LEN: usize = 48;
/// Map entry for a class with no picture (non-bird classes, misses).
pub const NO_IMAGE: u32 = 0xFFFF_FFFF;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub n: u32,
    pub w: u32,
    pub h: u32,
    pub names_off: u32,
    pub names_len: u32,
    pub regions_off: u32,
    pub regions_len: u32,
    pub n_regions: u32,
    pub image_off: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Fewer than `HEADER_LEN` bytes.
    Short,
    BadMagic,
    BadVersion(u32),
    /// Header fields that cannot describe a real file (zero size, name
    /// table overlapping the images, ...).
    Malformed,
    /// A name table shorter than `names_len` or with an out-of-order end.
    BadNames,
    /// A region section that runs past its length or is not UTF-8.
    BadRegions,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

impl Header {
    pub fn parse(b: &[u8]) -> Result<Header, Error> {
        if b.len() < HEADER_LEN {
            return Err(Error::Short);
        }
        if &b[0..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        let version = u32_at(b, 4);
        if version != VERSION {
            return Err(Error::BadVersion(version));
        }
        let h = Header {
            n: u32_at(b, 8),
            w: u32_at(b, 12),
            h: u32_at(b, 16),
            names_off: u32_at(b, 20),
            names_len: u32_at(b, 24),
            regions_off: u32_at(b, 28),
            regions_len: u32_at(b, 32),
            n_regions: u32_at(b, 36),
            image_off: u32_at(b, 40),
        };
        let names_end = h.names_off as u64 + h.names_len as u64;
        let regions_end = h.regions_off as u64 + h.regions_len as u64;
        if h.w == 0
            || h.h == 0
            || (h.names_off as usize) < HEADER_LEN
            || (h.names_len as u64) < 4 * h.n as u64
            || (h.regions_off as u64) < names_end
            || regions_end > h.image_off as u64
            || h.image_off % 16 != 0
        {
            return Err(Error::Malformed);
        }
        Ok(h)
    }

    /// Bytes per image.
    pub fn image_len(&self) -> usize {
        self.w as usize * self.h as usize * 2
    }

    /// Byte offset of image `i`, if it exists.
    pub fn image_offset(&self, i: usize) -> Option<u64> {
        (i < self.n as usize).then(|| self.image_off as u64 + i as u64 * self.image_len() as u64)
    }

    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(MAGIC);
        for (i, v) in [
            VERSION,
            self.n,
            self.w,
            self.h,
            self.names_off,
            self.names_len,
            self.regions_off,
            self.regions_len,
            self.n_regions,
            self.image_off,
        ]
        .iter()
        .enumerate()
        {
            b[4 + 4 * i..8 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
}

/// The name table: a view over the `names_len` bytes at `names_off`.
pub struct Names<'a> {
    n: usize,
    ends: &'a [u8],
    pool: &'a [u8],
}

impl<'a> Names<'a> {
    /// `bytes` is the table as read from the file (at least `names_len`
    /// long).
    pub fn new(h: &Header, bytes: &'a [u8]) -> Result<Names<'a>, Error> {
        let n = h.n as usize;
        let len = h.names_len as usize;
        if bytes.len() < len || len < 4 * n {
            return Err(Error::BadNames);
        }
        let (ends, pool) = bytes[..len].split_at(4 * n);
        let names = Names { n, ends, pool };
        let mut prev = 0;
        for i in 0..n {
            let e = names.end(i);
            if e < prev || e > pool.len() {
                return Err(Error::BadNames);
            }
            prev = e;
        }
        Ok(names)
    }

    fn end(&self, i: usize) -> usize {
        u32_at(self.ends, 4 * i) as usize
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Name `i`; `None` past the end or if the bytes are not UTF-8.
    pub fn get(&self, i: usize) -> Option<&'a str> {
        if i >= self.n {
            return None;
        }
        let start = if i == 0 { 0 } else { self.end(i - 1) };
        core::str::from_utf8(&self.pool[start..self.end(i)]).ok()
    }
}

/// One region's class -> image map, a view into the regions section.
#[derive(Clone, Copy)]
pub struct RegionMap<'a> {
    pub name: &'a str,
    map: &'a [u8],
}

impl<'a> RegionMap<'a> {
    pub fn n_classes(&self) -> usize {
        self.map.len() / 4
    }

    /// Image index for `class`, or `None` when the class has no picture
    /// (or is out of range).
    pub fn image(&self, class: usize) -> Option<usize> {
        if class >= self.n_classes() {
            return None;
        }
        match u32_at(self.map, 4 * class) {
            NO_IMAGE => None,
            i => Some(i as usize),
        }
    }

    /// Raw entries, `NO_IMAGE` included, for copying into a fixed table.
    pub fn raw(&self, class: usize) -> u32 {
        u32_at(self.map, 4 * class)
    }
}

/// The regions section: `n_regions` entries, walked linearly (a handful).
pub struct Regions<'a> {
    n: usize,
    bytes: &'a [u8],
}

impl<'a> Regions<'a> {
    /// `bytes` is the section as read from the file (at least
    /// `regions_len` long).
    pub fn new(h: &Header, bytes: &'a [u8]) -> Result<Regions<'a>, Error> {
        let len = h.regions_len as usize;
        if bytes.len() < len {
            return Err(Error::BadRegions);
        }
        let r = Regions { n: h.n_regions as usize, bytes: &bytes[..len] };
        // Validate by walking once.
        let mut off = 0;
        for _ in 0..r.n {
            let (_, next) = r.entry_at(off)?;
            off = next;
        }
        Ok(r)
    }

    fn entry_at(&self, off: usize) -> Result<(RegionMap<'a>, usize), Error> {
        let b = self.bytes;
        if off + 4 > b.len() {
            return Err(Error::BadRegions);
        }
        let name_len = u32_at(b, off) as usize;
        let name_end = off + 4 + name_len;
        let padded = (name_end + 3) / 4 * 4;
        if padded + 4 > b.len() {
            return Err(Error::BadRegions);
        }
        let name = core::str::from_utf8(&b[off + 4..name_end]).map_err(|_| Error::BadRegions)?;
        let n_classes = u32_at(b, padded) as usize;
        let map_start = padded + 4;
        let map_end = map_start + 4 * n_classes;
        if map_end > b.len() {
            return Err(Error::BadRegions);
        }
        Ok((RegionMap { name, map: &b[map_start..map_end] }, map_end))
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = RegionMap<'a>> + '_ {
        let mut off = 0;
        (0..self.n).map(move |_| {
            let (r, next) = self.entry_at(off).expect("validated in new()");
            off = next;
            r
        })
    }

    pub fn find(&self, name: &str) -> Option<RegionMap<'a>> {
        self.iter().find(|r| r.name == name)
    }
}

/// RGB565 -> the PSP's 8888 framebuffer word (`0xAABBGGRR`, R in the
/// low byte). Low bits are replicated so pure white stays 0xFF, not 0xF8.
#[inline]
pub fn rgb565_to_abgr8888(p: u16) -> u32 {
    let r5 = (p >> 11) as u32 & 0x1f;
    let g6 = (p >> 5) as u32 & 0x3f;
    let b5 = p as u32 & 0x1f;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g6 << 2) | (g6 >> 4);
    let b = (b5 << 3) | (b5 >> 2);
    0xff00_0000 | (b << 16) | (g << 8) | r
}

pub fn rgb888_to_rgb565(r: u8, g: u8, b: u8) -> u16 {
    ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3)
}

/// Copy one `w`×`h` RGB565 image (`src`, as read from the file: `w*h*2`
/// little-endian bytes) into an 8888 framebuffer at `dst`, which already
/// points at the top-left destination pixel; `stride` is the buffer width
/// in pixels (512 on the PSP).
///
/// # Safety
/// `dst` must be valid for `h` rows of `stride` words starting there;
/// the caller keeps the image on screen.
pub unsafe fn blit_rgb565(src: &[u8], w: usize, h: usize, dst: *mut u32, stride: usize) {
    debug_assert!(src.len() >= w * h * 2);
    let mut row = dst;
    let mut s = 0;
    for _ in 0..h {
        let mut p = row;
        for _ in 0..w {
            let px = u16::from_le_bytes([src[s], src[s + 1]]);
            s += 2;
            *p = rgb565_to_abgr8888(px);
            p = p.add(1);
        }
        row = row.add(stride);
    }
}

/// Device side: a pack open on `sceIo`, with the seek+read per image.
/// Buffers are the caller's, so this module owns no statics.
#[cfg(target_os = "psp")]
pub mod device {
    use super::{Error, Header, Names, Regions, HEADER_LEN};
    use core::ffi::c_void;
    use psp::sys::{sceIoClose, sceIoLseek32, sceIoOpen, sceIoRead, IoOpenFlags, IoWhence, SceUid};

    /// The PSP's 8888 debug framebuffer, uncached: what `psp::dprint!`
    /// draws into, 512 words per row.
    pub const SCREEN_STRIDE: usize = 512;
    pub const SCREEN_W: usize = 480;
    pub const SCREEN_H: usize = 272;

    pub fn framebuffer() -> *mut u32 {
        (0x4000_0000u32 | unsafe { psp::sys::sceGeEdramGetAddr() } as u32) as *mut u32
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum OpenError {
        /// No path opened (`sceIoOpen` error code of the last one).
        NotFound(i32),
        /// `sceIoRead` returned this.
        Io(i32),
        /// `sceIoLseek32` returned this instead of the requested offset.
        Seek(i32),
        Format(Error),
        /// The caller's buffer is smaller than the section.
        BufferTooSmall(u32),
    }

    pub struct Pack {
        fd: SceUid,
        pub header: Header,
    }

    fn read_exact(fd: SceUid, dst: &mut [u8]) -> Result<(), OpenError> {
        let mut done = 0;
        while done < dst.len() {
            let n = unsafe { sceIoRead(fd, dst[done..].as_mut_ptr() as *mut c_void, (dst.len() - done) as u32) };
            if n <= 0 {
                return Err(OpenError::Io(n));
            }
            done += n as usize;
        }
        Ok(())
    }

    fn cpath(path: &str) -> [u8; 96] {
        let mut p = [0u8; 96];
        p[..path.len().min(95)].copy_from_slice(&path.as_bytes()[..path.len().min(95)]);
        p
    }

    impl Pack {
        /// Open the first of `paths` that exists and read its header.
        pub fn open(paths: &[&str]) -> Result<Pack, OpenError> {
            let mut last = -1;
            for path in paths {
                let fd = unsafe { sceIoOpen(cpath(path).as_ptr(), IoOpenFlags::RD_ONLY, 0) };
                if fd.0 < 0 {
                    last = fd.0;
                    continue;
                }
                let mut hb = [0u8; HEADER_LEN];
                read_exact(fd, &mut hb)?;
                let header = Header::parse(&hb).map_err(OpenError::Format)?;
                return Ok(Pack { fd, header });
            }
            Err(OpenError::NotFound(last))
        }

        fn seek(&self, off: u64) -> Result<(), OpenError> {
            // sceIoLseek32, not sceIoLseek: the 64-bit form fails over
            // hostfs (the host logs an unknown fid and returns -1).
            let pos = unsafe { sceIoLseek32(self.fd, off as i32, IoWhence::Set) };
            if pos != off as i32 {
                return Err(OpenError::Seek(pos));
            }
            Ok(())
        }

        fn read_exact(&self, dst: &mut [u8]) -> Result<(), OpenError> {
            read_exact(self.fd, dst)
        }

        /// Read the name table into `buf` and view it.
        pub fn read_names<'a>(&self, buf: &'a mut [u8]) -> Result<Names<'a>, OpenError> {
            let len = self.header.names_len as usize;
            if buf.len() < len {
                return Err(OpenError::BufferTooSmall(self.header.names_len));
            }
            self.seek(self.header.names_off as u64)?;
            self.read_exact(&mut buf[..len])?;
            Names::new(&self.header, &buf[..len]).map_err(OpenError::Format)
        }

        /// Read the region maps into `buf` and view them.
        pub fn read_regions<'a>(&self, buf: &'a mut [u8]) -> Result<Regions<'a>, OpenError> {
            let len = self.header.regions_len as usize;
            if buf.len() < len {
                return Err(OpenError::BufferTooSmall(self.header.regions_len));
            }
            self.seek(self.header.regions_off as u64)?;
            self.read_exact(&mut buf[..len])?;
            Regions::new(&self.header, &buf[..len]).map_err(OpenError::Format)
        }

        /// Seek to image `i` and read it into `dst` (at least
        /// `header.image_len()` bytes).
        pub fn read_image(&self, i: usize, dst: &mut [u8]) -> Result<(), OpenError> {
            let len = self.header.image_len();
            if dst.len() < len {
                return Err(OpenError::BufferTooSmall(len as u32));
            }
            let off = self.header.image_offset(i).ok_or(OpenError::Format(Error::Malformed))?;
            self.seek(off)?;
            self.read_exact(&mut dst[..len])
        }

        /// Draw an image already in `src` at screen position (`x`, `y`).
        pub fn draw(&self, src: &[u8], x: usize, y: usize) {
            let (w, h) = (self.header.w as usize, self.header.h as usize);
            if x + w > SCREEN_W || y + h > SCREEN_H {
                return;
            }
            unsafe {
                super::blit_rgb565(src, w, h, framebuffer().add(x + y * SCREEN_STRIDE), SCREEN_STRIDE);
            }
        }

        pub fn close(self) {
            unsafe { sceIoClose(self.fd) };
        }
    }
}

#[cfg(feature = "imfile-pack")]
pub use pack::{pack_images, PackError, Packed};

#[cfg(feature = "imfile-pack")]
mod pack {
    extern crate std;

    use super::{rgb888_to_rgb565, Header, HEADER_LEN, NO_IMAGE};
    use std::path::{Path, PathBuf};
    use std::string::{String, ToString};
    use std::vec::Vec;
    use std::{fmt, format};

    #[derive(Debug)]
    pub enum PackError {
        Io(PathBuf, std::io::Error),
        Manifest(String),
        Empty,
    }

    impl fmt::Display for PackError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                PackError::Io(p, e) => write!(f, "{}: {e}", p.display()),
                PackError::Manifest(m) => write!(f, "manifest: {m}"),
                PackError::Empty => write!(f, "manifest lists no images"),
            }
        }
    }

    impl std::error::Error for PackError {}

    /// A pack in memory plus the label order it was built in (image `i`
    /// is `labels[i]`) and, per region, how many classes got no picture.
    pub struct Packed {
        pub bytes: Vec<u8>,
        pub labels: Vec<String>,
        pub width: u32,
        pub height: u32,
        /// `(region, classes without an image)`, in the order given.
        pub unmapped: Vec<(String, usize)>,
        /// Labels whose file was missing or undecodable and got the
        /// placeholder instead, with the reason.
        pub fallbacks: Vec<(String, String)>,
    }

    /// A stand-in thumbnail: slate card with a pale bird silhouette, so a
    /// broken download shows as "no photo" rather than aborting the pack.
    fn placeholder(w: u32, h: u32) -> image::RgbImage {
        let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([52, 58, 70]));
        let (fw, fh) = (w as f32, h as f32);
        let inside = |cx: f32, cy: f32, rx: f32, ry: f32, x: f32, y: f32| {
            let dx = (x - cx) / rx;
            let dy = (y - cy) / ry;
            dx * dx + dy * dy <= 1.0
        };
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                // Body, head, and a tail wedge; all in unit coordinates.
                let body = inside(0.46 * fw, 0.62 * fh, 0.26 * fw, 0.19 * fh, px, py);
                let head = inside(0.64 * fw, 0.40 * fh, 0.13 * fw, 0.13 * fh, px, py);
                let tail = {
                    let u = px / fw;
                    let v = py / fh;
                    u < 0.30 && v > 0.58 && v < 0.78 && (v - 0.58) > (0.30 - u) * 0.5
                };
                let beak = {
                    let u = px / fw;
                    let v = py / fh;
                    u > 0.76 && u < 0.86 && (v - 0.40).abs() < (0.86 - u) * 0.5
                };
                if body || head || tail || beak {
                    img.put_pixel(x, y, image::Rgb([150, 158, 172]));
                }
            }
        }
        img
    }

    /// Build a `PBIM` pack from `manifest.toml` (`"label" = "path"`, paths
    /// relative to the manifest), every image centre-cropped square and
    /// resized to `w`×`h`. `regions` is one `(name, labels)` per classifier
    /// blob, labels in class order; each becomes a class -> image map
    /// (`NO_IMAGE` where the label has no picture). Images are packed in
    /// sorted label order.
    pub fn pack_images(
        manifest: &Path,
        w: u32,
        h: u32,
        regions: &[(String, Vec<String>)],
    ) -> Result<Packed, PackError> {
        assert!(w > 0 && h > 0, "image size must be nonzero");
        let text = std::fs::read_to_string(manifest)
            .map_err(|e| PackError::Io(manifest.to_path_buf(), e))?;
        let table: toml::Table =
            toml::from_str(&text).map_err(|e| PackError::Manifest(e.to_string()))?;
        let base = manifest.parent().unwrap_or(Path::new("."));

        let mut entries: Vec<(String, PathBuf)> = Vec::with_capacity(table.len());
        for (label, value) in table {
            let rel = value.as_str().ok_or_else(|| {
                PackError::Manifest(format!("{label:?}: value must be a path string"))
            })?;
            entries.push((label, base.join(rel)));
        }
        if entries.is_empty() {
            return Err(PackError::Empty);
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Name table: u32 end offsets, then the pool.
        let mut ends = Vec::with_capacity(entries.len() * 4);
        let mut pool = Vec::new();
        for (label, _) in &entries {
            pool.extend_from_slice(label.as_bytes());
            ends.extend_from_slice(&(pool.len() as u32).to_le_bytes());
        }
        let names_len = ends.len() + pool.len();

        // Region maps.
        let index_of = |label: &str| entries.binary_search_by(|e| e.0.as_str().cmp(label)).ok();
        let mut regions_bytes = Vec::new();
        let mut unmapped = Vec::with_capacity(regions.len());
        for (name, labels) in regions {
            regions_bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
            regions_bytes.extend_from_slice(name.as_bytes());
            while regions_bytes.len() % 4 != 0 {
                regions_bytes.push(0);
            }
            regions_bytes.extend_from_slice(&(labels.len() as u32).to_le_bytes());
            let mut misses = 0;
            for label in labels {
                let idx = match index_of(label) {
                    Some(i) => i as u32,
                    None => {
                        misses += 1;
                        NO_IMAGE
                    }
                };
                regions_bytes.extend_from_slice(&idx.to_le_bytes());
            }
            unmapped.push((name.clone(), misses));
        }

        let names_off = HEADER_LEN;
        // 4-aligned so the u32 tables inside are aligned in the file too.
        let regions_off = (names_off + names_len + 3) / 4 * 4;
        let image_off = (regions_off + regions_bytes.len() + 15) / 16 * 16;

        let header = Header {
            n: entries.len() as u32,
            w,
            h,
            names_off: names_off as u32,
            names_len: names_len as u32,
            regions_off: regions_off as u32,
            regions_len: regions_bytes.len() as u32,
            n_regions: regions.len() as u32,
            image_off: image_off as u32,
        };
        let mut bytes = Vec::with_capacity(image_off + entries.len() * header.image_len());
        bytes.extend_from_slice(&header.to_bytes());
        bytes.extend_from_slice(&ends);
        bytes.extend_from_slice(&pool);
        bytes.resize(regions_off, 0);
        bytes.extend_from_slice(&regions_bytes);
        bytes.resize(image_off, 0);

        let mut fallbacks = Vec::new();
        let stand_in = placeholder(w, h);
        for (label, path) in &entries {
            // Sniff the format from the bytes: iNat serves the odd PNG
            // under a .jpg URL, and `image::open` would trust the name.
            let decoded = std::fs::read(path)
                .map_err(|e| e.to_string())
                .and_then(|b| image::load_from_memory(&b).map_err(|e| e.to_string()));
            let rgb = match decoded {
                Ok(img) => {
                    let side = img.width().min(img.height());
                    let x = (img.width() - side) / 2;
                    let y = (img.height() - side) / 2;
                    img.crop_imm(x, y, side, side)
                        .resize_exact(w, h, image::imageops::FilterType::Lanczos3)
                        .to_rgb8()
                }
                Err(reason) => {
                    fallbacks.push((label.clone(), format!("{}: {reason}", path.display())));
                    stand_in.clone()
                }
            };
            for px in rgb.pixels() {
                bytes.extend_from_slice(&rgb888_to_rgb565(px[0], px[1], px[2]).to_le_bytes());
            }
        }

        Ok(Packed {
            bytes,
            labels: entries.into_iter().map(|(l, _)| l).collect(),
            width: w,
            height: h,
            unmapped,
            fallbacks,
        })
    }

    /// Labels of a `PBRD` classifier blob (prune_classifier.py
    /// --write-blob): header `magic, version, n, k, labels_len`, then
    /// `n*k` f32 weights, `n` f32 bias, then the newline-separated labels.
    pub fn pbrd_labels(path: &Path) -> Result<Vec<String>, PackError> {
        let b = std::fs::read(path).map_err(|e| PackError::Io(path.to_path_buf(), e))?;
        let bad = |m: &str| PackError::Manifest(format!("{}: {m}", path.display()));
        if b.len() < 32 || &b[0..4] != b"PBRD" {
            return Err(bad("not a PBRD blob"));
        }
        let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
        let (n, k, labels_len) = (u(8), u(12), u(16));
        let start = 32 + 4 * (n * k + n);
        if start + labels_len > b.len() {
            return Err(bad("labels run past the end"));
        }
        let text = std::str::from_utf8(&b[start..start + labels_len]).map_err(|_| bad("labels not UTF-8"))?;
        let labels: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        if labels.len() != n {
            return Err(bad("label count does not match n_classes"));
        }
        Ok(labels)
    }
}

#[cfg(feature = "imfile-pack")]
pub use pack::pbrd_labels;

#[cfg(all(test, feature = "imfile-pack"))]
mod tests {
    extern crate std;
    use super::*;
    use std::path::PathBuf;
    use std::string::ToString;
    use std::vec;

    fn scratch() -> PathBuf {
        let d = std::env::temp_dir().join(format!("pbim-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn pack_then_read_back() {
        let d = scratch();
        // Two flat-colour PNGs of different shapes: the crop and resize
        // must leave each image its own colour.
        let red = image::RgbImage::from_pixel(20, 10, image::Rgb([255, 0, 0]));
        let blue = image::RgbImage::from_pixel(7, 30, image::Rgb([0, 0, 255]));
        red.save(d.join("red.png")).unwrap();
        blue.save(d.join("blue.png")).unwrap();
        std::fs::write(
            d.join("manifest.toml"),
            "\"Zz zz_Late\" = \"red.png\"\n\"Aa aa_Early\" = \"blue.png\"\n",
        )
        .unwrap();

        let regions = vec![
            ("east".to_string(), vec!["Zz zz_Late".to_string(), "Engine_Engine".to_string(), "Aa aa_Early".to_string()]),
            ("west".to_string(), vec!["Aa aa_Early".to_string()]),
        ];
        let p = pack_images(&d.join("manifest.toml"), 4, 4, &regions).unwrap();
        assert_eq!(p.labels, ["Aa aa_Early", "Zz zz_Late"]);
        assert!(p.fallbacks.is_empty());
        assert_eq!(p.unmapped, [("east".to_string(), 1), ("west".to_string(), 0)]);

        let h = Header::parse(&p.bytes).unwrap();
        assert_eq!((h.n, h.w, h.h, h.n_regions), (2, 4, 4, 2));
        assert_eq!(h.image_len(), 32);
        assert_eq!(p.bytes.len(), h.image_off as usize + 64);

        let names = Names::new(&h, &p.bytes[h.names_off as usize..]).unwrap();
        assert_eq!(names.get(0), Some("Aa aa_Early"));
        assert_eq!(names.get(1), Some("Zz zz_Late"));
        assert_eq!(names.get(2), None);

        let regions = Regions::new(&h, &p.bytes[h.regions_off as usize..]).unwrap();
        assert_eq!(regions.len(), 2);
        let east = regions.find("east").unwrap();
        assert_eq!(east.n_classes(), 3);
        assert_eq!(east.image(0), Some(1));
        assert_eq!(east.image(1), None);
        assert_eq!(east.raw(1), NO_IMAGE);
        assert_eq!(east.image(2), Some(0));
        assert_eq!(east.image(3), None);
        let west = regions.find("west").unwrap();
        assert_eq!((west.n_classes(), west.image(0)), (1, Some(0)));
        assert!(regions.find("north").is_none());

        let img = |i: usize| {
            let o = h.image_offset(i).unwrap() as usize;
            &p.bytes[o..o + h.image_len()]
        };
        assert!(img(0).chunks(2).all(|c| u16::from_le_bytes([c[0], c[1]]) == 0x001f));
        assert!(img(1).chunks(2).all(|c| u16::from_le_bytes([c[0], c[1]]) == 0xf800));
        assert_eq!(h.image_offset(2), None);

        let mut fb = [0u32; 8 * 4];
        unsafe { blit_rgb565(img(1), 4, 4, fb.as_mut_ptr().add(2), 8) };
        assert_eq!(fb[2], 0xff00_00ff);
        assert_eq!(fb[1], 0);
        assert_eq!(fb[8 + 5], 0xff00_00ff);
        assert_eq!(fb[8 + 6], 0);

        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn broken_file_gets_placeholder() {
        let d = scratch().join("broken");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("junk.jpg"), b"<html>not an image</html>").unwrap();
        std::fs::write(
            d.join("manifest.toml"),
            "\"Bb bb_Junk\" = \"junk.jpg\"\n\"Cc cc_Gone\" = \"missing.jpg\"\n",
        )
        .unwrap();
        let p = pack_images(&d.join("manifest.toml"), 8, 8, &[]).unwrap();
        assert_eq!(p.fallbacks.len(), 2);
        assert_eq!(p.fallbacks[0].0, "Bb bb_Junk");
        assert_eq!(p.fallbacks[1].0, "Cc cc_Gone");
        let h = Header::parse(&p.bytes).unwrap();
        // The placeholder is not blank: silhouette pixels differ from the card.
        let o = h.image_off as usize;
        let px: std::vec::Vec<u16> =
            p.bytes[o..o + h.image_len()].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert!(px.iter().any(|&v| v != px[0]));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn header_rejections() {
        assert_eq!(Header::parse(&[0; 8]), Err(Error::Short));
        let mut b = Header {
            n: 1, w: 2, h: 2, names_off: 48, names_len: 8, regions_off: 56, regions_len: 0, n_regions: 0, image_off: 64,
        }
        .to_bytes();
        assert!(Header::parse(&b).is_ok());
        b[4] = 9;
        assert_eq!(Header::parse(&b), Err(Error::BadVersion(9)));
        b[4] = 2;
        b[40] = 65; // unaligned image_off
        assert_eq!(Header::parse(&b), Err(Error::Malformed));
        b[0] = b'X';
        assert_eq!(Header::parse(&b), Err(Error::BadMagic));
    }

    #[test]
    fn truncated_regions_rejected() {
        let h = Header {
            n: 0, w: 1, h: 1, names_off: 48, names_len: 0, regions_off: 48, regions_len: 12, n_regions: 1, image_off: 64,
        };
        // name_len 4 "east", n_classes 5 but no map follows.
        let mut b = std::vec::Vec::new();
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(b"east");
        b.extend_from_slice(&5u32.to_le_bytes());
        assert_eq!(Regions::new(&h, &b).err(), Some(Error::BadRegions));
    }

    #[test]
    fn colour_roundtrip() {
        assert_eq!(rgb565_to_abgr8888(0xffff), 0xffff_ffff);
        assert_eq!(rgb565_to_abgr8888(0), 0xff00_0000);
        assert_eq!(rgb565_to_abgr8888(rgb888_to_rgb565(255, 0, 0)), 0xff00_00ff);
    }
}
