use std::slice;

use crate::{binary, gc, heap::AllocationError, Heap, Value};

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
