#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

pub mod atom;
pub mod binary;
pub mod ffi;
pub mod gc;
pub mod header;
pub mod heap;
pub mod layout;
pub mod value;

pub use atom::AtomTable;
pub use binary::{
    ref_count as binary_ref_count, release as release_binary, retain as retain_binary,
};
pub use gc::{
    collect as gc_collect, disable as gc_disable, enable as gc_enable,
    ensure_initialised as ensure_gc,
};
pub use header::{Header, Tag};
pub use heap::{AllocationError, Heap};
pub use value::Value;
