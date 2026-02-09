#![no_std]
#![no_main]

extern crate psp_ml;

use psp_ml::println;

psp_ml::module!("hello_psp", 1, 0);

fn app_main() {
    psp::enable_home_button();

    println!("=========================");
    println!("Hello from PSP!");
    println!("If you can read this, psp-runner is working.");
    println!("2 + 2 = {}", 2 + 2);
    println!("=========================");
}
