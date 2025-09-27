use runtime_cranelift::ffi::{
    GleamFfiStatus, gleam_ffi_bytes_free, gleam_ffi_decode_uint, gleam_ffi_encode_uint,
    gleam_ffi_list_from_array, gleam_ffi_list_to_array, gleam_ffi_resource_from_ptr,
    gleam_ffi_resource_to_ptr, gleam_ffi_string_from_utf8, gleam_ffi_string_to_utf8,
    gleam_ffi_values_free,
};
use runtime_cranelift::{
    Value, list_from_values, list_to_values, string_from_rust, string_to_rust,
};

#[test]
fn string_to_utf8_roundtrip() {
    runtime_cranelift::ensure_gc();
    let value = string_from_rust("hi");
    let bytes = gleam_ffi_string_to_utf8(value.to_raw());
    assert_eq!(bytes.status, GleamFfiStatus::Ok);
    assert_eq!(bytes.len, 2);
    unsafe {
        let slice = std::slice::from_raw_parts(bytes.ptr, bytes.len);
        assert_eq!(slice, b"hi");
    }
    gleam_ffi_bytes_free(bytes);
}

#[test]
fn string_from_utf8_handles_empty() {
    runtime_cranelift::ensure_gc();
    let result = gleam_ffi_string_from_utf8(std::ptr::null(), 0);
    assert_eq!(result.status, GleamFfiStatus::Ok);
    let value = Value::from_raw(result.value);
    assert_eq!(string_to_rust(value).unwrap(), "");
}

#[test]
fn encode_decode_uint_roundtrip() {
    runtime_cranelift::ensure_gc();
    let raw = gleam_ffi_encode_uint(123);
    let decoded = gleam_ffi_decode_uint(raw);
    assert_eq!(decoded.status, GleamFfiStatus::Ok);
    assert_eq!(decoded.value, 123);
}

#[test]
fn decode_uint_rejects_negative() {
    let negative = Value::from_i63(-5);
    let decoded = gleam_ffi_decode_uint(negative.to_raw());
    assert_eq!(decoded.status, GleamFfiStatus::NegativeInt);
}

#[test]
fn decode_uint_rejects_non_int() {
    runtime_cranelift::ensure_gc();
    let value = string_from_rust("not an int");
    let decoded = gleam_ffi_decode_uint(value.to_raw());
    assert_eq!(decoded.status, GleamFfiStatus::NotAnInt);
}

#[test]
fn resource_roundtrip() {
    runtime_cranelift::ensure_gc();
    let pointer = 0x2345 as *mut core::ffi::c_void;
    let value = gleam_ffi_resource_from_ptr(pointer);
    assert_eq!(value.status, GleamFfiStatus::Ok);
    let decoded = gleam_ffi_resource_to_ptr(value.value);
    assert_eq!(decoded.status, GleamFfiStatus::Ok);
    assert_eq!(decoded.ptr, pointer);
}

#[test]
fn resource_to_ptr_rejects_non_resource() {
    runtime_cranelift::ensure_gc();
    let value = string_from_rust("not a resource").to_raw();
    let decoded = gleam_ffi_resource_to_ptr(value);
    assert_eq!(decoded.status, GleamFfiStatus::NotAResource);
}

#[test]
fn list_from_array_roundtrip() {
    runtime_cranelift::ensure_gc();
    let raw_values = vec![Value::from_i63(5).to_raw(), Value::from_i63(6).to_raw()];
    let result = gleam_ffi_list_from_array(raw_values.as_ptr(), raw_values.len());
    assert_eq!(result.status, GleamFfiStatus::Ok);
    let list = Value::from_raw(result.value);
    let decoded = list_to_values(list).unwrap();
    assert_eq!(decoded, vec![Value::from_i63(5), Value::from_i63(6)]);
}

#[test]
fn list_from_array_null_pointer_rejected() {
    let result = gleam_ffi_list_from_array(std::ptr::null(), 2);
    assert_eq!(result.status, GleamFfiStatus::InvalidArgument);
}

#[test]
fn list_to_array_roundtrip() {
    runtime_cranelift::ensure_gc();
    let list = list_from_values(&[Value::from_i63(10), Value::from_i63(20)]);
    let array = gleam_ffi_list_to_array(list.to_raw());
    assert_eq!(array.status, GleamFfiStatus::Ok);
    assert_eq!(array.len, 2);
    unsafe {
        let slice = std::slice::from_raw_parts(array.ptr, array.len);
        assert_eq!(slice[0], Value::from_i63(10).to_raw());
        assert_eq!(slice[1], Value::from_i63(20).to_raw());
    }
    gleam_ffi_values_free(array);
}

#[test]
fn list_to_array_rejects_non_list() {
    runtime_cranelift::ensure_gc();
    let value = string_from_rust("no list");
    let array = gleam_ffi_list_to_array(value.to_raw());
    assert_eq!(array.status, GleamFfiStatus::NotAList);
}
