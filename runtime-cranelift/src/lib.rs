#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

pub mod atom;
pub mod header;
pub mod heap;
pub mod layout;
pub mod value;

pub use atom::AtomTable;
pub use header::{Header, Tag};
pub use heap::{AllocationError, Heap};
pub use value::Value;
