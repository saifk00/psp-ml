#![no_std]
#![no_main]

use psp_ml::dprintln;

psp_ml::module!("hello_psp", 1, 0);

fn app_main() {
    psp::enable_home_button();

    dprintln!("=========================");
    dprintln!("Hello from PSP!");
    dprintln!("If you can read this, cargo psp-ml run is working.");
    dprintln!("2 + 2 = {}", 2 + 2);
    dprintln!("=========================");
}
