mod data;
mod data_type;
mod traits;
pub(crate) mod triangular;

#[cfg(feature = "nalgebra")]
mod nalgebra_support;
mod row_major;

pub use data::*;
pub use data_type::*;
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
pub use nalgebra_support::*;
pub use row_major::*;
pub use traits::*;
