/*
 * device.c - USB device lifecycle + the connect/pump entry points.
 *
 * Forked from psplinkusb/usbhostfs_pc/main.c (euid_usb_bulk_write/read,
 * configure_usb, open_device, close_device, wait_for_device). The
 * uhfs_ctx_new/free/connect/disconnect/pump functions are new — they
 * replace start_hostfs()'s outer reconnect-`while(1)` and its interactive
 * shell/CLI/daemon plumbing with an explicit API a Rust caller drives.
 *
 * See uhfs_ctx.h for the Stage A/B staging note (globals today, ctx fields
 * in Stage B).
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include "uhfs_ctx.h"

libusb_context *usbctx = NULL;
libusb_device_handle *usbhdr = NULL;

int g_verbose = 0;
int g_gdbdebug = 0;
int g_pid = HOSTFSDRIVER_PID;
int g_timeout = 0;

int euid_usb_bulk_write(libusb_device_handle *dev, int ep, char *bytes, int size,
	int timeout)
{
	int wrbytes;

	V_PRINTF(2, "Bulk Write dev %p, ep 0x%x, bytes %p, size %d, timeout %d\n",
			dev, ep, bytes, size, timeout);

	int ret = libusb_bulk_transfer(dev, ep, (unsigned char*)bytes, size, &wrbytes, timeout);
	if (!ret)
		ret = wrbytes;

	V_PRINTF(2, "Bulk Write returned %d\n", ret);

	return ret;
}

int euid_usb_bulk_read(libusb_device_handle *dev, int ep, char *bytes, int size,
	int timeout)
{
	int rdbytes;

	V_PRINTF(2, "Bulk Read dev %p, ep 0x%x, bytes %p, size %d, timeout %d\n",
			dev, ep, bytes, size, timeout);
	/* Fixed vs. upstream: this used to always read from the global `usbhdr`
	 * regardless of the `dev` argument passed in. Harmless with a single
	 * global connection (the only case that ever existed), but worth fixing
	 * now that we're touching this function anyway. */
	int ret = libusb_bulk_transfer(dev, ep, (unsigned char*)bytes, size, &rdbytes, timeout);
	if (!ret)
		ret = rdbytes;

	V_PRINTF(2, "Bulk Read returned %d\n", ret);

	return ret;
}

int configure_usb(libusb_device_handle *devh) {
	int cfgn;
	int r = libusb_get_configuration(devh, &cfgn);
	if (r) {
		fprintf(stderr, "Failed at reading selected USB configuration: %d\n", r);
		return r;
	}
	if (cfgn != USB_CONFIG_NUM)
	{
		r = libusb_set_configuration(devh, 1);
		if (r) {
			fprintf(stderr, "Failed at selecting USB configuration: %d\n", r);
			return r;
		}
	}
	r = libusb_claim_interface(devh, USB_IFACE_NUM);
	if (r) {
		fprintf(stderr, "Failed at claiming USB interface: %d\n", r);
		return r;
	}

	return 0;
}

libusb_device_handle *open_device(libusb_device *usbdev)
{
	libusb_device_handle *ret = NULL;

	int escalated = 0;
	int r = libusb_open(usbdev, &ret);
	if (r == LIBUSB_ERROR_ACCESS) {
		/* We do not seem to have permissions, try elevating privileges (if sbit is set) */
		if (seteuid(0) < 0 || setegid(0) < 0) {
			fprintf(stderr, "Permission error while opening the USB device.\n");
			fprintf(stderr, "  You might need to enable USB permissions for this program.\n");
			fprintf(stderr, "  On Linux you mgiht need to install a udev config file (50-psplink.rules)\n");
			fprintf(stderr, "  You can also set the `setuid` bit on the binary alternatively\n");

			seteuid(getuid());
			setegid(getgid());
			return NULL;
		}
		escalated = 1;

		/* Try again, see if the sbit helps. */
		r = libusb_open(usbdev, &ret);
		if (r == LIBUSB_ERROR_ACCESS) {
			fprintf(stderr, "Permission error while opening the USB device.\n");
			ret = NULL;
		}
	}
	if (!r) {
		if (configure_usb(ret))
			ret = NULL;
	}

	if (escalated) {
		seteuid(getuid());
		setegid(getgid());
	}
	return ret;
}

void close_device(libusb_device_handle *dev)
{
	if(dev)
	{
		libusb_release_interface(dev, 0);
		libusb_reset_device(dev);
		libusb_close(dev);
	}
}

libusb_device_handle *wait_for_device(void)
{
	libusb_device **devs;
	libusb_device_handle *usbdev = NULL;

	fprintf(stderr, "waiting for device...\n");

	while(!usbdev)
	{
		ssize_t devcnt = libusb_get_device_list(usbctx, &devs);

		for (ssize_t i = 0; i < devcnt; i++)
		{
			struct libusb_device_descriptor desc;
			int r = libusb_get_device_descriptor(devs[i], &desc);
			if (!r) {
				if (desc.idVendor == SONY_VID && desc.idProduct == g_pid)
				{
					printf("Found Sony PSP device (%04x:%04x) at bus: %d device: %d\n",
						desc.idVendor, desc.idProduct,
						libusb_get_bus_number(devs[i]), libusb_get_device_address(devs[i]));
					usbdev = open_device(devs[i]);
					if (usbdev)
					{
						fprintf(stderr, "Connected to device\n");
						return usbdev;
					}
				}
			}
		}

		libusb_free_device_list(devs, 1);

		/* Sleep for one second */
		sleep(1);
	}

	return NULL;
}

/* ------------------------------------------------------------------- */
/* Public FFI entry points                                             */
/* ------------------------------------------------------------------- */

struct UhfsCtx *uhfs_ctx_new(void)
{
	struct UhfsCtx *ctx = calloc(1, sizeof(struct UhfsCtx));
	if (!ctx) {
		return NULL;
	}

	if (usbctx == NULL) {
		if (libusb_init(&usbctx) != 0) {
			fprintf(stderr, "USB initialization failed\n");
			free(ctx);
			return NULL;
		}
	}

	return ctx;
}

void uhfs_ctx_free(struct UhfsCtx *ctx)
{
	free(ctx);
}

/* Blocks until a PSP is found and the HOSTFS_MAGIC handshake write
 * succeeds. The HOSTFS_CMD_HELLO reply round-trip happens naturally on the
 * first uhfs_pump() call, same as upstream start_hostfs() never special-
 * cased it either. */
int uhfs_connect(struct UhfsCtx *ctx)
{
	uint32_t magic;

	if (getcwd(g_rootdir, PATH_MAX) == NULL) {
		fprintf(stderr, "Could not get current path\n");
		return -1;
	}

	init_hostfs();

	usbhdr = wait_for_device();
	if (!usbhdr) {
		return -1;
	}

	magic = LE32(HOSTFS_MAGIC);
	if (euid_usb_bulk_write(usbhdr, 0x2, (char *) &magic, sizeof(magic), 1000) != sizeof(magic)) {
		fprintf(stderr, "Error sending HOSTFS_MAGIC handshake\n");
		close_device(usbhdr);
		usbhdr = NULL;
		return -1;
	}

	ctx->connected = 1;
	return 0;
}

void uhfs_disconnect(struct UhfsCtx *ctx)
{
	if (usbhdr) {
		close_device(usbhdr);
		usbhdr = NULL;
	}
	close_hostfs();
	ctx->connected = 0;
}

/* One iteration of the read+dispatch loop that used to be the inner
 * `while(1)` body of start_hostfs(). Returns >0 if a command was
 * processed, 0 on an idle timeout (safe to call again), <0 if the USB
 * connection is gone (caller should uhfs_disconnect()). */
int uhfs_pump(struct UhfsCtx *ctx, int timeout_ms)
{
	uint32_t data[512/sizeof(uint32_t)];
	int readlen;

	if (!ctx->connected || !usbhdr) {
		return -1;
	}

	readlen = euid_usb_bulk_read(usbhdr, 0x81, (char*) data, 512, timeout_ms);
	if (readlen == 0)
	{
		fprintf(stderr, "Read cancelled (remote disconnected)\n");
		return -1;
	}
	else if (readlen == LIBUSB_ERROR_TIMEOUT)
	{
		return 0;
	}
	else if (readlen < 0)
	{
		return -1;
	}

	if ((size_t) readlen < sizeof(uint32_t))
	{
		fprintf(stderr, "Error could not read magic\n");
		return -1;
	}

	if (LE32(data[0]) == HOSTFS_MAGIC)
	{
		if ((size_t) readlen < sizeof(struct HostFsCmd))
		{
			fprintf(stderr, "Error reading command header %d\n", readlen);
			return -1;
		}
		do_hostfs((struct HostFsCmd *) data, readlen);
	}
	else if (LE32(data[0]) == ASYNC_MAGIC)
	{
		if ((size_t) readlen < sizeof(struct AsyncCommand))
		{
			fprintf(stderr, "Error reading async header %d\n", readlen);
			return -1;
		}
		do_async((struct AsyncCommand *) data, readlen);
	}
	else if (LE32(data[0]) == BULK_MAGIC)
	{
		if ((size_t) readlen < sizeof(struct BulkCommand))
		{
			fprintf(stderr, "Error reading bulk header %d\n", readlen);
			return -1;
		}
		do_bulk((struct BulkCommand *) data, readlen);
	}
	else
	{
		fprintf(stderr, "Error, invalid magic %08X\n", LE32(data[0]));
		return -1;
	}

	return 1;
}
