use runtime_cranelift::{Value, list_from_values, string_from_rust, string_to_rust};

#[test]
fn inspect_integer() {
    let output = runtime_cranelift::ffi::inspect(Value::from_i63(42).to_raw());
    let value = Value::from_raw(output);
    let text = string_to_rust(value).expect("inspect returned utf8 string");
    assert_eq!(text, "42");
}

#[test]
fn inspect_string() {
    let original = string_from_rust("hello");
    let output = runtime_cranelift::ffi::inspect(original.to_raw());
    let value = Value::from_raw(output);
    let text = string_to_rust(value).expect("inspect returned utf8 string");
    assert_eq!(text, "\"hello\"");
}

#[test]
fn inspect_charlist_list() {
    let list = list_from_values(&[Value::from_i63(65), Value::from_i63(66)]);
    let output = runtime_cranelift::ffi::inspect(list.to_raw());
    let value = Value::from_raw(output);
    let text = string_to_rust(value).expect("inspect returned utf8 string");
    assert_eq!(text, "charlist.from_string(\"AB\")");
}
