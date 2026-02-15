//! Benchmark helpers for PSP ML inference.
//!
//! Provides [`format_results`] to serialize benchmark data as self-describing
//! JSON, and [`write_hostfs`] to write it via psplink HostFS. The JSON
//! includes op names and timing so host-side tools need no compiler coupling.

/// Benchmark results ready for serialization.
pub struct BenchmarkResult<'a> {
    /// Run identifier (e.g. "vfpu", "naive").
    pub id: &'a str,
    /// Number of inference samples run.
    pub num_samples: usize,
    /// Ticks per second (from `sceRtcGetTickResolution` or host equivalent).
    pub tick_resolution: u64,
    /// Wall-clock ticks for the entire benchmark run.
    pub total_ticks: u64,
    /// Number of correct predictions.
    pub correct: u32,
    /// Per-op names (from `generated::OP_NAMES`).
    pub op_names: &'a [&'a str],
    /// Per-op accumulated ticks (from `generated::forward_timed`).
    pub op_ticks: &'a [u64],
}

/// Return the index of the maximum element.
pub fn argmax(output: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = output[0];
    for i in 1..output.len() {
        if output[i] > best_val {
            best_val = output[i];
            best = i;
        }
    }
    best
}

/// Format benchmark results as self-describing JSON.
pub fn format_results(r: &BenchmarkResult) -> JsonBuf {
    let total_us = if r.tick_resolution > 0 {
        (r.total_ticks * 1_000_000) / r.tick_resolution
    } else {
        0
    };
    let per_sample_us = if r.num_samples > 0 {
        total_us / r.num_samples as u64
    } else {
        0
    };

    let mut j = JsonBuf::new();
    j.push_str("{\n");

    j.push_str("  \"id\": \"");
    j.push_str(r.id);
    j.push_str("\",\n");

    j.push_str("  \"num_samples\": ");
    j.push_u64(r.num_samples as u64);
    j.push_str(",\n");

    j.push_str("  \"tick_resolution\": ");
    j.push_u64(r.tick_resolution);
    j.push_str(",\n");

    j.push_str("  \"total_ticks\": ");
    j.push_u64(r.total_ticks);
    j.push_str(",\n");

    j.push_str("  \"total_us\": ");
    j.push_u64(total_us);
    j.push_str(",\n");

    j.push_str("  \"per_sample_us\": ");
    j.push_u64(per_sample_us);
    j.push_str(",\n");

    j.push_str("  \"correct\": ");
    j.push_u64(r.correct as u64);
    j.push_str(",\n");

    j.push_str("  \"ops\": [\n");
    let num_ops = r.op_names.len().min(r.op_ticks.len());
    for i in 0..num_ops {
        let us = if r.tick_resolution > 0 {
            (r.op_ticks[i] * 1_000_000) / r.tick_resolution
        } else {
            0
        };
        j.push_str("    { \"name\": \"");
        j.push_str(r.op_names[i]);
        j.push_str("\", \"ticks\": ");
        j.push_u64(r.op_ticks[i]);
        j.push_str(", \"us\": ");
        j.push_u64(us);
        j.push_str(" }");
        if i + 1 < num_ops {
            j.push_byte(b',');
        }
        j.push_byte(b'\n');
    }
    j.push_str("  ]\n");

    j.push_str("}\n");
    j
}

/// Write bytes to a HostFS path on PSP. No-op on non-PSP targets.
pub fn write_hostfs(path: &str, data: &[u8]) {
    #[cfg(target_os = "psp")]
    {
        let mut path_buf = [0u8; 256];
        let len = path.as_bytes().len().min(254);
        path_buf[..len].copy_from_slice(&path.as_bytes()[..len]);

        unsafe {
            let fd = psp::sys::sceIoOpen(
                path_buf.as_ptr(),
                psp::sys::IoOpenFlags::WR_ONLY
                    | psp::sys::IoOpenFlags::CREAT
                    | psp::sys::IoOpenFlags::TRUNC,
                0o644,
            );
            if fd.0 >= 0 {
                psp::sys::sceIoWrite(
                    fd,
                    data.as_ptr() as *const core::ffi::c_void,
                    data.len(),
                );
                psp::sys::sceIoClose(fd);
            }
        }
    }

    #[cfg(not(target_os = "psp"))]
    let _ = (path, data);
}

// ---------------------------------------------------------------------------
// Minimal no_std JSON buffer
// ---------------------------------------------------------------------------

/// Fixed-size JSON buffer. No heap allocation.
pub struct JsonBuf {
    buf: [u8; 4096],
    pos: usize,
}

impl JsonBuf {
    fn new() -> Self {
        JsonBuf { buf: [0u8; 4096], pos: 0 }
    }

    /// The formatted JSON bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.pos]
    }

    fn push_byte(&mut self, b: u8) {
        if self.pos < self.buf.len() {
            self.buf[self.pos] = b;
            self.pos += 1;
        }
    }

    fn push_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.push_byte(b);
        }
    }

    fn push_u64(&mut self, mut val: u64) {
        if val == 0 {
            self.push_byte(b'0');
            return;
        }
        let start = self.pos;
        while val > 0 {
            self.push_byte(b'0' + (val % 10) as u8);
            val /= 10;
        }
        let end = self.pos;
        let mut i = start;
        let mut j = end - 1;
        while i < j {
            self.buf.swap(i, j);
            i += 1;
            j -= 1;
        }
    }
}
