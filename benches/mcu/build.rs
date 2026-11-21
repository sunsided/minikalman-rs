//! Writes the memory layout of the emulated chip for the target being built.
use std::{env, fs, path::PathBuf};

fn main() {
    let target = env::var("TARGET").unwrap();
    // thumbv6m runs on Renode's STM32L071 (Cortex-M0+, flash and RAM enlarged in
    // renode/stm32l071-large.repl), everything else on its STM32F412
    // (Cortex-M4F). The sizes are the regions of those platform descriptions.
    let memory = if target.starts_with("thumbv6m") {
        "MEMORY {\n  FLASH : ORIGIN = 0x08000000, LENGTH = 384K\n  RAM : ORIGIN = 0x20000000, LENGTH = 64K\n}\n"
    } else {
        "MEMORY {\n  FLASH : ORIGIN = 0x08000000, LENGTH = 1024K\n  RAM : ORIGIN = 0x20000000, LENGTH = 256K\n}\n"
    };
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out.join("memory.x"), memory).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    // Here rather than in .cargo/config.toml: run.sh builds a copy nested in this directory,
    // which would pick up both configs and pass the script twice.
    println!("cargo:rustc-link-arg=-Tlink.x");
    println!("cargo:rerun-if-changed=build.rs");
}
