use std::{
    io::{self, Write},
    ptr::NonNull,
    slice,
};

use crate::{
    binary, gc,
    heap::AllocationError,
    layout::{Binary, BinarySlice},
    Header, Heap, Tag, Value,
};

fn unwrap_allocation<T>(result: Result<T, AllocationError>, context: &'static str) -> T {
    result.unwrap_or_else(|_| panic!("runtime allocation failed: {context}"))
}

#[no_mangle]
pub extern "C" fn gleam_runtime_init() {
    gc::ensure_initialised();
}

#[no_mangle]
pub extern "C" fn gleam_runtime_collect() {
    gc::collect();
}

#[no_mangle]
pub extern "C" fn gleam_list_nil() -> u64 {
    Value::nil().to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_bool_true() -> u64 {
    Value::from_bool(true).to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_bool_false() -> u64 {
    Value::from_bool(false).to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_alloc_tuple(values_ptr: *const u64, len: usize) -> u64 {
    gc::ensure_initialised();
    if len == 0 {
        return Value::nil().to_raw();
    }

    let slice = unsafe { slice::from_raw_parts(values_ptr, len) };
    let values: Vec<Value> = slice.iter().copied().map(Value::from_raw).collect();
    let heap = Heap::new();
    unwrap_allocation(heap.alloc_tuple(&values), "tuple").to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_list_cons(head_raw: u64, tail_raw: u64) -> u64 {
    gc::ensure_initialised();
    let heap = Heap::new();
    let head = Value::from_raw(head_raw);
    let tail = Value::from_raw(tail_raw);
    unwrap_allocation(heap.alloc_cons(head, tail), "list cons").to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_binary_from_slice(bytes_ptr: *const u8, len: usize) -> u64 {
    gc::ensure_initialised();
    if len == 0 {
        return Value::nil().to_raw();
    }

    let heap = Heap::new();
    let data = unwrap_allocation(heap.alloc_binary_data(len), "binary data");
    unsafe {
        let buffer = &mut *data.as_ptr();
        std::ptr::copy_nonoverlapping(bytes_ptr, buffer.as_mut_ptr(), len);
    }
    let value = unwrap_allocation(heap.alloc_binary(data, len, len), "binary");
    value.to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_binary_retain(raw: u64) {
    let value = Value::from_raw(raw);
    if !value.is_boxed() {
        return;
    }
    let ptr = value.as_boxed::<crate::layout::Binary>();
    if let Some(ptr) = ptr {
        unsafe {
            let data = (*ptr.as_ptr()).data;
            if let Some(non_null) = std::ptr::NonNull::new(data) {
                binary::retain(non_null);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn gleam_binary_release(raw: u64) {
    let value = Value::from_raw(raw);
    if !value.is_boxed() {
        return;
    }
    let ptr = value.as_boxed::<crate::layout::Binary>();
    if let Some(ptr) = ptr {
        unsafe {
            let data = (*ptr.as_ptr()).data;
            if let Some(non_null) = std::ptr::NonNull::new(data) {
                if binary::release(non_null) {
                    // Buffer will be reclaimed with the GC; nothing else to do.
                }
            }
        }
    }
}

#[derive(Debug)]
enum StringAccessError {
    NotAString,
    MissingData,
}

fn value_to_bytes(value: Value) -> Result<Vec<u8>, StringAccessError> {
    if value == Value::nil() {
        return Ok(Vec::new());
    }

    let header_ptr = value
        .as_boxed::<Header>()
        .ok_or(StringAccessError::NotAString)?;
    let header = unsafe { *header_ptr.as_ref() };

    match header.tag() {
        Tag::Binary => {
            let binary_ptr = value
                .as_boxed::<Binary>()
                .ok_or(StringAccessError::NotAString)?;
            let binary = unsafe { binary_ptr.as_ref() };
            let data_ptr = NonNull::new(binary.data).ok_or(StringAccessError::MissingData)?;
            let data = unsafe { data_ptr.as_ref() };
            let bytes = unsafe { slice::from_raw_parts(data.as_ptr(), binary.len) };
            Ok(bytes.to_vec())
        }
        Tag::BinarySlice => {
            let slice_ptr = value
                .as_boxed::<BinarySlice>()
                .ok_or(StringAccessError::NotAString)?;
            let binary_slice = unsafe { slice_ptr.as_ref() };
            let data_ptr = NonNull::new(binary_slice.data).ok_or(StringAccessError::MissingData)?;
            let data = unsafe { data_ptr.as_ref() };
            let start = unsafe { data.as_ptr().add(binary_slice.offset) };
            let bytes = unsafe { slice::from_raw_parts(start, binary_slice.len) };
            Ok(bytes.to_vec())
        }
        _ => Err(StringAccessError::NotAString),
    }
}

enum OutputStream {
    Stdout,
    Stderr,
}

fn runtime_print(raw: u64, newline: bool, stream: OutputStream) -> u64 {
    let value = Value::from_raw(raw);
    let mut bytes = value_to_bytes(value).unwrap_or_else(|_| panic!("expected String value"));
    if newline {
        bytes.push(b'\n');
    }

    let result = match stream {
        OutputStream::Stdout => {
            let mut handle = io::stdout().lock();
            handle.write_all(&bytes).and_then(|_| handle.flush())
        }
        OutputStream::Stderr => {
            let mut handle = io::stderr().lock();
            handle.write_all(&bytes).and_then(|_| handle.flush())
        }
    };

    result.unwrap_or_else(|err| panic!("runtime print failed: {err}"));
    Value::nil().to_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_value_to_bytes_returns_original_data() {
        gc::ensure_initialised();
        let heap = Heap::new();
        let data = heap.alloc_binary_data(3).unwrap();
        unsafe {
            let buffer = data.as_ptr();
            std::ptr::copy_nonoverlapping(b"hey".as_ptr(), (*buffer).as_mut_ptr(), 3);
        }
        let value = heap.alloc_binary(data, 3, 3).unwrap();
        let bytes = value_to_bytes(value).expect("binary to bytes");
        assert_eq!(bytes, b"hey");
    }

    #[test]
    fn binary_slice_to_bytes_returns_slice() {
        gc::ensure_initialised();
        let heap = Heap::new();
        let data = heap.alloc_binary_data(5).unwrap();
        unsafe {
            let buffer = data.as_ptr();
            std::ptr::copy_nonoverlapping(b"hello".as_ptr(), (*buffer).as_mut_ptr(), 5);
        }
        let slice_value = heap.alloc_binary_slice(data, 3, 5, 1).unwrap();
        let bytes = value_to_bytes(slice_value).expect("binary slice to bytes");
        assert_eq!(bytes, b"ell");
    }

    #[test]
    fn runtime_print_returns_nil() {
        let result = runtime_print(Value::nil().to_raw(), false, OutputStream::Stdout);
        assert_eq!(result, Value::nil().to_raw());
    }
}

#[no_mangle]
pub extern "C" fn gleam_io_print(raw: u64) -> u64 {
    runtime_print(raw, false, OutputStream::Stdout)
}

#[no_mangle]
pub extern "C" fn gleam_io_println(raw: u64) -> u64 {
    runtime_print(raw, true, OutputStream::Stdout)
}

#[no_mangle]
pub extern "C" fn gleam_io_print_error(raw: u64) -> u64 {
    runtime_print(raw, false, OutputStream::Stderr)
}

#[no_mangle]
pub extern "C" fn gleam_io_println_error(raw: u64) -> u64 {
    runtime_print(raw, true, OutputStream::Stderr)
}
