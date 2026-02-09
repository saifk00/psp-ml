use std::time::Duration;

use rusb::{Context, DeviceHandle, UsbContext};

use crate::error::Error;

const PSP_VID: u16 = 0x054C;
const PSP_PID: u16 = 0x01C9;

const EP1_IN: u8 = 0x81; // Bulk IN  (PSP -> Host)
const EP2_OUT: u8 = 0x02; // Bulk OUT (Host -> PSP) — HostFS responses
const EP3_OUT: u8 = 0x03; // Bulk OUT (Host -> PSP) — Async/shell commands

const INTERFACE: u8 = 0;

const DEFAULT_TIMEOUT: Duration = Duration::from_millis(1000);

pub struct PspUsb {
    handle: DeviceHandle<Context>,
}

impl PspUsb {
    /// Scan the USB bus for a PSP running psplink, open and claim the interface.
    pub fn open() -> Result<Self, Error> {
        let ctx = Context::new().map_err(Error::UsbOpen)?;

        let device = ctx
            .devices()
            .map_err(Error::UsbOpen)?
            .iter()
            .find(|dev| {
                dev.device_descriptor()
                    .map(|d| d.vendor_id() == PSP_VID && d.product_id() == PSP_PID)
                    .unwrap_or(false)
            })
            .ok_or(Error::DeviceNotFound)?;

        let handle = device.open().map_err(Error::UsbOpen)?;

        // Detach any kernel driver that may have claimed the interface.
        handle
            .set_auto_detach_kernel_driver(true)
            .map_err(Error::UsbOpen)?;

        handle
            .claim_interface(INTERFACE)
            .map_err(Error::UsbOpen)?;

        log::info!(
            "Opened PSP device (bus {:03} dev {:03})",
            device.bus_number(),
            device.address()
        );

        Ok(PspUsb { handle })
    }

    /// Read from EP1 (PSP -> Host). Returns number of bytes read.
    pub fn read_ep1(&self, buf: &mut [u8], timeout: Duration) -> Result<usize, Error> {
        self.handle
            .read_bulk(EP1_IN, buf, timeout)
            .map_err(Error::UsbIo)
    }

    /// Write to EP2 (Host -> PSP, HostFS responses).
    pub fn write_ep2(&self, data: &[u8]) -> Result<usize, Error> {
        self.handle
            .write_bulk(EP2_OUT, data, DEFAULT_TIMEOUT)
            .map_err(Error::UsbIo)
    }

    /// Write to EP3 (Host -> PSP, async/shell commands).
    pub fn write_ep3(&self, data: &[u8]) -> Result<usize, Error> {
        self.handle
            .write_bulk(EP3_OUT, data, DEFAULT_TIMEOUT)
            .map_err(Error::UsbIo)
    }
}

impl Drop for PspUsb {
    fn drop(&mut self) {
        let _ = self.handle.release_interface(INTERFACE);
    }
}
