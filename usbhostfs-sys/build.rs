fn main() {
    let usb = pkg_config::probe_library("libusb-1.0")
        .expect("libusb-1.0 not found via pkg-config (usbhostfs-sys needs libusb dev headers)");

    let mut build = cc::Build::new();
    build.include("vendor");
    for inc in &usb.include_paths {
        build.include(inc);
    }
    build
        .file("vendor/device.c")
        .file("vendor/hostfs.c")
        .file("vendor/async.c");
    build.flag_if_supported("-Wno-unused-parameter");
    build.compile("usbhostfs_core");

    println!("cargo:rerun-if-changed=vendor");
}
