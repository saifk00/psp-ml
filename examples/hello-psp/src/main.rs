#![no_std]
#![no_main]

use psp_ml::dprintln;

psp_ml::module!("hello_psp", 1, 0);

fn app_main() {
    psp::enable_home_button();

    dprintln!("Hello from PSP!\nIf you can read this, cargo psp-ml run is working.\n");
}
