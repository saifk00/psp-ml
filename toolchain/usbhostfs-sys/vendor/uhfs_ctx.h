/*
 * uhfs_ctx.h - internal header shared by device.c / hostfs.c / async.c.
 *
 * Forked from psplinkusb/usbhostfs_pc/main.c. Not part of the public FFI
 * surface (see ../src/lib.rs for that) — this just lets the three vendored
 * translation units share the constants/macros/prototypes that used to all
 * live in one file.
 *
 * Stage A note: state (usbctx/usbhdr/g_drives/open_files/open_dirs/etc.) is
 * still held in file-scope globals here, exactly as in the original
 * usbhostfs_pc, on the assumption of a single connection per process. Stage
 * B moves all of this into `struct UhfsCtx`'s fields and threads an
 * explicit ctx pointer through every function below. Don't "fix" that here.
 */
#ifndef UHFS_CTX_H
#define UHFS_CTX_H

#include <stdint.h>
#include <libusb.h>
#include <pthread.h>
#include <limits.h>
#include "usbhostfs.h"
#include "psp_fileio.h"

#define MAX_FILES 256
#define MAX_DIRS  256
#define MAX_TOKENS 256
#define MAX_HOSTDRIVES 8

#define USB_CONFIG_NUM 1
#define USB_IFACE_NUM  0

#define LE16(x) (x)
#define LE32(x) (x)
#define LE64(x) (x)

#define V_PRINTF(level, fmt, ...) { if(g_verbose >= level) { fprintf(stderr, fmt, ## __VA_ARGS__); } }
#define GETERROR(x) (0x80010000 | (x))

/* Opaque handle returned to Rust. Stage A: effectively a marker — real
 * state lives in the globals declared below until Stage B. */
struct UhfsCtx
{
	int connected;
};

/* Contains the paths for a single host drive */
struct HostDrive
{
	char rootdir[PATH_MAX];
	char currdir[PATH_MAX];
};

struct FileHandle
{
	int opened;
	int mode;
	char *name;
};

struct DirHandle
{
	int opened;
	/* Current count of entries left */
	int count;
	/* Current position in the directory entries */
	int pos;
	/* Head of list, each entry will be freed when read */
	SceIoDirent *pDir;
};

extern struct FileHandle open_files[MAX_FILES];
extern struct DirHandle  open_dirs[MAX_DIRS];

extern libusb_context *usbctx;
extern libusb_device_handle *usbhdr;

extern struct HostDrive g_drives[MAX_HOSTDRIVES];
extern char g_rootdir[PATH_MAX];
extern pthread_mutex_t g_drivemtx;

extern int g_verbose;
extern int g_gdbdebug;
extern int g_nocase;
extern int g_msslash;
extern int g_pid;
extern int g_timeout;

/* device.c */
int euid_usb_bulk_write(libusb_device_handle *dev, int ep, char *bytes, int size, int timeout);
int euid_usb_bulk_read(libusb_device_handle *dev, int ep, char *bytes, int size, int timeout);
int configure_usb(libusb_device_handle *devh);
libusb_device_handle *open_device(libusb_device *usbdev);
void close_device(libusb_device_handle *dev);
libusb_device_handle *wait_for_device(void);

/* hostfs.c */
int gen_path(char *path, int dir);
int make_path(unsigned int drive, const char *path, char *retpath, int dir);
int open_file(int drive, const char *path, unsigned int mode, unsigned int mask);
int dir_open(int drive, const char *dirname);
int dir_close(int did);
int init_hostfs(void);
void close_hostfs(void);
void do_hostfs(struct HostFsCmd *cmd, int readlen);
int add_drive(int num, const char *dir);

/* Public FFI entry points (mirrored in ../src/lib.rs's extern "C" block) */
struct UhfsCtx *uhfs_ctx_new(void);
void uhfs_ctx_free(struct UhfsCtx *ctx);
int uhfs_add_drive(struct UhfsCtx *ctx, int num, const char *dir);
int uhfs_connect(struct UhfsCtx *ctx);
void uhfs_disconnect(struct UhfsCtx *ctx);
int uhfs_pump(struct UhfsCtx *ctx, int timeout_ms);

/* async.c */
void print_gdbdebug(int dir, const uint8_t *data, int len);
void do_async(struct AsyncCommand *cmd, int readlen);
void do_bulk(struct BulkCommand *cmd, int readlen);

typedef void (*UhfsAsyncCallback)(void *user, int channel, const uint8_t *data, int len);
void uhfs_set_async_callback(struct UhfsCtx *ctx, UhfsAsyncCallback cb, void *user);
int uhfs_async_write(struct UhfsCtx *ctx, int channel, const uint8_t *data, int len);

#endif
