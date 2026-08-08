fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("psp") {
        return;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let exports_exp = std::path::Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .join("kernel-plugin/exports.exp");

    println!("cargo:rerun-if-changed={}", exports_exp.display());
}
