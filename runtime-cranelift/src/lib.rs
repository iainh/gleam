//! Crate entry point exposing the public surface of the native runtime used by
//! the Cranelift backend.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

pub mod atom;
pub mod binary;
pub mod ffi;
pub mod ffi_helpers;
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

pub use ffi_helpers::{
    ListDecodeError, ResourceDecodeError, StringDecodeError, UnsignedIntDecodeError,
    decode_unsigned_int, encode_unsigned_int, list_from_values, list_to_values, resource_from_ptr,
    resource_to_ptr, string_bytes, string_from_bytes, string_from_rust, string_to_rust,
};
