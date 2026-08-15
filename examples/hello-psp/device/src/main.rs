#![no_std]
#![no_main]

use psp_rt::dprintln;

psp_rt::module!("hello_psp", 1, 0);

fn app_main() {
    psp_rt::enable_home_button();

    dprintln!("Hello from PSP!\nIf you can read this, cargo run -p hello-psp-host is working.\n");
}
