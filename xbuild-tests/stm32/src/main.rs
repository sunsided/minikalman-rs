#![no_std]
#![no_main]

mod kalman_fixed_example;
mod kalman_float_example;

use cortex_m_rt::entry;
use panic_halt as _; // When a panic occurs, halt the microcontroller
use stm32f3xx_hal as hal;

use hal::{pac, prelude::*};

#[entry]
fn main() -> ! {
    let _gravity = kalman_fixed_example::predict_gravity();
    let _gravity = kalman_float_example::predict_gravity();

    let dp = pac::Peripherals::take().unwrap();

    let _flash = dp.FLASH.constrain();
    let mut rcc = dp.RCC.constrain();
    let mut gpioe = dp.GPIOE.split(&mut rcc.ahb);

    let mut led = gpioe
        .pe9
        .into_push_pull_output(&mut gpioe.moder, &mut gpioe.otyper);

    loop {
        led.set_high().unwrap();
        cortex_m::asm::delay(8_000_000);
        led.set_low().unwrap();
        cortex_m::asm::delay(8_000_000);
    }
}
