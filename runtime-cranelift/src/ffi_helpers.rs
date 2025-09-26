//!
//! Helper primitives for native FFI shims that need to inspect or construct
//! Gleam values. These functions keep the layout knowledge in one place so user
//! shims can stay small and avoid `unsafe` code wherever possible.

use core::ffi::c_void;
use core::ptr::{self, NonNull};

use crate::{
    gc,
    header::{Header, Tag},
    layout::{Binary as BinaryLayout, BinarySlice, ConsCell, ResourceHandle},
    value::Value,
};

/// Error type returned when a value cannot be interpreted as a UTF-8 string.
#[derive(Debug, PartialEq, Eq)]
pub enum StringDecodeError {
    NotAString,
    MissingData,
    InvalidUtf8,
}

/// Error type returned when a value cannot be interpreted as an unsigned int.
#[derive(Debug, PartialEq, Eq)]
pub enum UnsignedIntDecodeError {
    NotAnInt,
    Negative,
}

/// Error type returned when a value cannot be interpreted as a resource pointer.
#[derive(Debug, PartialEq, Eq)]
pub enum ResourceDecodeError {
    NotAResource,
    NullPointer,
}

/// Error type returned when a value cannot be interpreted as a List.
#[derive(Debug, PartialEq, Eq)]
pub enum ListDecodeError {
    NotAList,
}

/// Reads the bytes out of a Gleam string (binary or binary slice) into an owned
/// `Vec<u8>`.
pub fn string_bytes(value: Value) -> Result<Vec<u8>, StringDecodeError> {
    if value == Value::nil() {
        return Ok(Vec::new());
    }

    let header_ptr = value
        .as_boxed::<Header>()
        .ok_or(StringDecodeError::NotAString)?;
    let header = unsafe { *header_ptr.as_ref() };

    match header.tag() {
        Tag::Binary => {
            let binary_ptr = value
                .as_boxed::<BinaryLayout>()
                .ok_or(StringDecodeError::NotAString)?;
            let binary = unsafe { binary_ptr.as_ref() };
            let data_ptr = NonNull::new(binary.data).ok_or(StringDecodeError::MissingData)?;
            let data = unsafe { data_ptr.as_ref() };
            let bytes = unsafe { core::slice::from_raw_parts(data.as_ptr(), binary.len) };
            Ok(bytes.to_vec())
        }
        Tag::BinarySlice => {
            let slice_ptr = value
                .as_boxed::<BinarySlice>()
                .ok_or(StringDecodeError::NotAString)?;
            let slice = unsafe { slice_ptr.as_ref() };
            let data_ptr = NonNull::new(slice.data).ok_or(StringDecodeError::MissingData)?;
            let data = unsafe { data_ptr.as_ref() };
            let start = unsafe { data.as_ptr().add(slice.offset) };
            let bytes = unsafe { core::slice::from_raw_parts(start, slice.len) };
            Ok(bytes.to_vec())
        }
        _ => Err(StringDecodeError::NotAString),
    }
}

/// Converts a Gleam string value into a Rust `String`.
pub fn string_to_rust(value: Value) -> Result<String, StringDecodeError> {
    let bytes = string_bytes(value)?;
    String::from_utf8(bytes).map_err(|_| StringDecodeError::InvalidUtf8)
}

/// Creates a Gleam string from the provided UTF-8 bytes.
pub fn string_from_bytes(bytes: &[u8]) -> Value {
    if bytes.is_empty() {
        return Value::nil();
    }

    gc::ensure_initialised();
    let heap = crate::heap::Heap::new();
    let data = heap
        .alloc_binary_data(bytes.len())
        .expect("alloc binary data");
    unsafe {
        let buffer = &mut *data.as_ptr();
        ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr(), bytes.len());
    }
    heap.alloc_binary(data, bytes.len(), bytes.len())
        .expect("alloc binary")
}

/// Converts a Rust string slice into a Gleam string value.
pub fn string_from_rust(text: &str) -> Value {
    string_from_bytes(text.as_bytes())
}

/// Encodes a non-negative integer into Gleam's tagged small-int format.
pub fn encode_unsigned_int(value: u64) -> Value {
    let signed = i64::try_from(value).expect("unsigned int out of range");
    Value::from_i63(signed)
}

/// Decodes a Gleam small-int into a Rust `u64`.
pub fn decode_unsigned_int(value: Value) -> Result<u64, UnsignedIntDecodeError> {
    let signed = value.to_i63().ok_or(UnsignedIntDecodeError::NotAnInt)?;
    if signed < 0 {
        return Err(UnsignedIntDecodeError::Negative);
    }
    Ok(signed as u64)
}

/// Wraps an opaque pointer in a Gleam `Resource` value.
pub fn resource_from_ptr(pointer: *mut c_void) -> Value {
    gc::ensure_initialised();
    let heap = crate::heap::Heap::new();
    heap.alloc_resource(pointer)
        .expect("allocate resource handle")
}

/// Extracts an opaque pointer from a Gleam `Resource` value.
pub fn resource_to_ptr(value: Value) -> Result<*mut c_void, ResourceDecodeError> {
    let header_ptr = value
        .as_boxed::<Header>()
        .ok_or(ResourceDecodeError::NotAResource)?;
    let header = unsafe { *header_ptr.as_ref() };
    if header.tag() != Tag::Resource {
        return Err(ResourceDecodeError::NotAResource);
    }
    let handle_ptr = value
        .as_boxed::<ResourceHandle>()
        .ok_or(ResourceDecodeError::NotAResource)?;
    let handle = unsafe { handle_ptr.as_ref() };
    if handle.pointer.is_null() {
        return Err(ResourceDecodeError::NullPointer);
    }
    Ok(handle.pointer)
}

/// Builds a Gleam list from a slice of values.
pub fn list_from_values(values: &[Value]) -> Value {
    gc::ensure_initialised();
    let heap = crate::heap::Heap::new();
    let mut list = Value::nil();
    for value in values.iter().rev() {
        list = heap
            .alloc_cons(*value, list)
            .expect("allocate list cons cell");
    }
    list
}

/// Converts a Gleam list into a vector of values.
pub fn list_to_values(mut list: Value) -> Result<Vec<Value>, ListDecodeError> {
    let mut result = Vec::new();
    while list != Value::nil() {
        let ptr = list
            .as_boxed::<ConsCell>()
            .ok_or(ListDecodeError::NotAList)?;
        unsafe {
            let cons = ptr.as_ref();
            result.push(cons.head);
            list = cons.tail;
        }
    }
    Ok(result)
}
