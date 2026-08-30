/*
 * async.c - async channel (shell/stdout/stderr/gdb) data in and out.
 *
 * Forked from psplinkusb/usbhostfs_pc/main.c (print_gdbdebug, do_async,
 * do_bulk). Upstream's do_async/do_bulk terminated in a `write()`/
 * `fixed_write()` to a per-channel TCP client socket (g_clientsocks[chan])
 * — that's the whole thing this crate exists to remove. They now call a
 * registered callback instead. uhfs_async_write is the PC->PSP direction,
 * extracted from what used to be async_thread's write path in main.c.
 */
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "uhfs_ctx.h"

static UhfsAsyncCallback g_async_cb = NULL;
static void *g_async_cb_user = NULL;

void print_gdbdebug(int dir, const uint8_t *data, int len)
{
	int i;

	if(dir)
	{
		printf("HOST->GDB (");
	}
	else
	{
		printf("GDB->HOST (");
	}

	for(i = 0; i < len; i++)
	{
		if(data[i] >= 32)
		{
			putchar(data[i]);
		}
		else
		{
			printf("\\%02x", data[i]);
		}
	}

	printf(")\n");
}

void do_async(struct AsyncCommand *cmd, int readlen)
{
	uint8_t *data;

	V_PRINTF(2, "Async Magic: %08X\n", LE32(cmd->magic));
	V_PRINTF(2, "Async Channel: %08X\n", LE32(cmd->channel));
	V_PRINTF(2, "Async Extra Len: %d\n", readlen - (int)sizeof(struct AsyncCommand));

	if(readlen > (int) sizeof(struct AsyncCommand))
	{
		data = (uint8_t *) cmd + sizeof(struct AsyncCommand);
		unsigned int chan = LE32(cmd->channel);
		int len = readlen - (int) sizeof(struct AsyncCommand);
		if((chan < MAX_ASYNC_CHANNELS) && g_async_cb)
		{
			g_async_cb(g_async_cb_user, (int) chan, data, len);
			if((chan == ASYNC_GDB) && (g_gdbdebug))
			{
				print_gdbdebug(0, data, len);
			}
		}
	}
}

void do_bulk(struct BulkCommand *cmd, int readlen)
{
	static char block[HOSTFS_BULK_MAXWRITE];
	int  read = 0;
	int  len = 0;
	unsigned int chan = 0;
	int  ret = -1;

	(void) readlen;

	chan = LE32(cmd->channel);
	len = LE32(cmd->size);

	V_PRINTF(2, "Bulk write command length: %d channel %d\n", len, chan);

	while(read < len)
	{
		int readsize;

		readsize = (len - read) > HOSTFS_MAX_BLOCK ? HOSTFS_MAX_BLOCK : (len - read);
		ret = euid_usb_bulk_read(usbhdr, 0x81, &block[read], readsize, 10000);
		if(ret != readsize)
		{
			fprintf(stderr, "Error reading write data readsize %d, ret %d\n", readsize, ret);
			break;
		}
		read += readsize;
	}

	if(read >= len)
	{
		if((chan < MAX_ASYNC_CHANNELS) && g_async_cb)
		{
			g_async_cb(g_async_cb_user, (int) chan, (uint8_t *) block, len);
		}
	}
}

void uhfs_set_async_callback(struct UhfsCtx *ctx, UhfsAsyncCallback cb, void *user)
{
	(void) ctx;
	g_async_cb = cb;
	g_async_cb_user = user;
}

/* PC->PSP direction. One USB packet per call (max ~508 bytes of payload,
 * matching upstream async_thread's own single-packet-per-socket-read
 * behavior — it never chunked either). Callers with small, framed payloads
 * (e.g. psplink-connection's shell-command framing) are always well under
 * this. */
int uhfs_async_write(struct UhfsCtx *ctx, int channel, const uint8_t *data, int len)
{
	char buf[512];
	struct AsyncCommand *cmd = (struct AsyncCommand *) buf;
	int size;

	(void) ctx;

	if (!usbhdr) {
		return -1;
	}

	size = len > (int)(sizeof(buf) - sizeof(struct AsyncCommand))
		? (int)(sizeof(buf) - sizeof(struct AsyncCommand))
		: len;

	cmd->magic = LE32(ASYNC_MAGIC);
	cmd->channel = LE32(channel);
	memcpy(buf + sizeof(struct AsyncCommand), data, size);

	return euid_usb_bulk_write(usbhdr, 0x3, buf, size + (int) sizeof(struct AsyncCommand), 10000);
}
