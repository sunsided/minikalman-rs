fn main() {
    #[cfg(all(feature = "std", feature = "no_std"))]
    compile_error!("feature \"std\" and feature \"no_std\" cannot be enabled at the same time");

    #[cfg(not(any(feature = "std", feature = "no_std")))]
    compile_error!("either feature \"std\" or feature \"no_std\" must be enabled");
}
