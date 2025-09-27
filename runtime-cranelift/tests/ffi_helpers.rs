use runtime_cranelift::{
    ListDecodeError, ResourceDecodeError, UnsignedIntDecodeError, Value, decode_unsigned_int,
    encode_unsigned_int, list_from_values, list_to_values, resource_from_ptr, resource_to_ptr,
    string_from_rust, string_to_rust,
};

#[test]
fn roundtrip_int() {
    let value = encode_unsigned_int(42);
    assert_eq!(decode_unsigned_int(value), Ok(42));
}

#[test]
fn reject_negative_int() {
    let value = encode_unsigned_int(42);
    let negative = Value::from_i63(-1);
    assert_eq!(
        decode_unsigned_int(negative),
        Err(UnsignedIntDecodeError::Negative)
    );
    assert_eq!(
        decode_unsigned_int(Value::nil()),
        Err(UnsignedIntDecodeError::NotAnInt)
    );
    assert_eq!(decode_unsigned_int(value), Ok(42));
}

#[test]
fn string_roundtrip() {
    let value = string_from_rust("hello");
    assert_eq!(string_to_rust(value), Ok("hello".to_string()));
}

#[test]
fn resource_roundtrip() {
    let pointer = 0x1234 as *mut core::ffi::c_void;
    let value = resource_from_ptr(pointer);
    assert_eq!(resource_to_ptr(value), Ok(pointer));
}

#[test]
fn resource_null_pointer_rejected() {
    let value = resource_from_ptr(core::ptr::null_mut());
    assert_eq!(
        resource_to_ptr(value),
        Err(ResourceDecodeError::NullPointer)
    );
}

#[test]
fn list_roundtrip() {
    let elements = vec![Value::from_i63(1), Value::from_i63(2), Value::from_i63(3)];
    let list = list_from_values(&elements);
    let decoded = list_to_values(list).unwrap();
    assert_eq!(decoded, elements);
}

#[test]
fn list_decode_rejects_non_list() {
    let value = string_from_rust("not a list");
    assert_eq!(list_to_values(value), Err(ListDecodeError::NotAList));
}
