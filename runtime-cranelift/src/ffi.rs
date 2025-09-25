//! `extern "C"` entry points and helpers invoked from native Gleam code.

use std::{
    convert::TryFrom,
    fs,
    io::{self, Write},
    mem,
    ptr::NonNull,
    slice,
    sync::OnceLock,
};

use crate::{
    atom::AtomTable,
    binary, gc,
    heap::AllocationError,
    layout::{
        Binary, BinaryData, BinarySlice, BitArray as BitArrayLayout, Closure, ClosureFn, ConsCell,
        FloatBox, Map, MapEntry, MapTable,
    },
    Header, Heap, Tag, Value,
};

use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD};
use base64::Engine;
use hex::{decode as hex_decode, encode_upper};
use rand::Rng;
use unicode_segmentation::UnicodeSegmentation;

const HEX_DIGITS: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
];

fn unwrap_allocation<T>(result: Result<T, AllocationError>, context: &'static str) -> T {
    result.unwrap_or_else(|_| panic!("runtime allocation failed: {context}"))
}

static ATOM_TABLE: OnceLock<AtomTable> = OnceLock::new();

fn atom_table() -> &'static AtomTable {
    ATOM_TABLE.get_or_init(AtomTable::new)
}

fn atom(name: &str) -> Value {
    atom_table().intern(name)
}

fn atom_ok() -> Value {
    atom("ok")
}

fn atom_error() -> Value {
    atom("error")
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
pub extern "C" fn gleam_float_from_f64(number: f64) -> u64 {
    float_to_value(number).to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_int_negate(raw: u64) -> u64 {
    let int = value_to_i63(Value::from_raw(raw), "int_negate");
    Value::from_i63(-int).to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_panic(message_raw: u64) -> u64 {
    let default_message = "`panic` expression evaluated.".to_string();
    let message = value_to_string(Value::from_raw(message_raw)).unwrap_or(default_message);
    panic!("{message}");
}

#[no_mangle]
/// # Safety
/// `env_ptr` must reference `env_len` valid `Value` words captured by the closure.
pub unsafe extern "C" fn gleam_alloc_closure(
    code_ptr: u64,
    env_ptr: *const u64,
    env_len: usize,
) -> u64 {
    let code = unsafe { std::mem::transmute::<usize, ClosureFn>(code_ptr as usize) };
    let env_values = env_ptr as *const Value;
    let heap = Heap::new();
    let value = unwrap_allocation(unsafe { heap.alloc_closure(code, env_values, env_len) }, "closure");
    value.to_raw()
}

#[no_mangle]
pub extern "C" fn gleam_apply_closure(closure_raw: u64, args_ptr: *const u64, argc: usize) -> u64 {
    let closure_value = Value::from_raw(closure_raw);
    let closure_ptr = closure_value
        .as_boxed::<Closure>()
        .expect("closure value must be boxed");

    unsafe {
        let closure = closure_ptr.as_ref();
        let env_ptr = closure.env.as_ptr();
        let args = args_ptr as *const Value;
        let result = (closure.code_ptr)(env_ptr, args, argc);
        result.to_raw()
    }
}

#[no_mangle]
/// # Safety
/// `values_ptr` must point to `len` valid `Value` words.
pub unsafe extern "C" fn gleam_alloc_tuple(values_ptr: *const u64, len: usize) -> u64 {
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
/// # Safety
/// `fields_ptr` must point to `len` valid `Value` words.
pub unsafe extern "C" fn gleam_alloc_record(
    constructor_index: u64,
    fields_ptr: *const u64,
    len: usize,
) -> u64 {
    gc::ensure_initialised();
    let index = u32::try_from(constructor_index)
        .unwrap_or_else(|_| panic!("record constructor index out of range"));
    let values = unsafe { slice::from_raw_parts(fields_ptr as *const Value, len) };
    let heap = Heap::new();
    unwrap_allocation(heap.alloc_record(index, values), "record").to_raw()
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
/// # Safety
/// `bytes_ptr` must reference `len` initialised bytes.
pub unsafe extern "C" fn gleam_binary_from_slice(bytes_ptr: *const u8, len: usize) -> u64 {
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

fn bytes_to_value(bytes: &[u8]) -> Value {
    if bytes.is_empty() {
        return Value::nil();
    }

    gc::ensure_initialised();
    let heap = Heap::new();
    let data = unwrap_allocation(heap.alloc_binary_data(bytes.len()), "binary data");
    unsafe {
        let buffer = &mut *data.as_ptr();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr(), bytes.len());
    }
    unwrap_allocation(heap.alloc_binary(data, bytes.len(), bytes.len()), "binary")
}

fn value_to_string(value: Value) -> Result<String, StringAccessError> {
    let bytes = value_to_bytes(value)?;
    String::from_utf8(bytes).map_err(|_| StringAccessError::NotAString)
}

fn string_to_value(string: &str) -> Value {
    bytes_to_value(string.as_bytes())
}

fn list_to_vec(mut list: Value) -> Vec<Value> {
    let mut result = Vec::new();
    while list != Value::nil() {
        let ptr = list
            .as_boxed::<ConsCell>()
            .unwrap_or_else(|| panic!("expected List value"));
        unsafe {
            let cons = ptr.as_ref();
            result.push(cons.head);
            list = cons.tail;
        }
    }
    result
}

fn list_from_vec(values: Vec<Value>) -> Value {
    gc::ensure_initialised();
    let heap = Heap::new();
    let mut list = Value::nil();
    for value in values.into_iter().rev() {
        list = unwrap_allocation(heap.alloc_cons(value, list), "list from vec");
    }
    list
}

struct BitArrayView {
    data: NonNull<BinaryData>,
    bit_offset: usize,
    bit_len: usize,
    capacity_bits: usize,
}

fn bit_array_view(value: Value, context: &'static str) -> BitArrayView {
    let ptr = value
        .as_boxed::<BitArrayLayout>()
        .unwrap_or_else(|| panic!("runtime {context} expected BitArray value"));
    unsafe {
        let bit_array = ptr.as_ref();
        let data = NonNull::new(bit_array.data)
            .unwrap_or_else(|| panic!("runtime {context} missing bit array data"));
        BitArrayView {
            data,
            bit_offset: bit_array.bit_offset,
            bit_len: bit_array.bit_len,
            capacity_bits: bit_array.capacity_bits,
        }
    }
}

fn bit_array_data_slice(view: &BitArrayView) -> &[u8] {
    unsafe {
        let data = view.data.as_ref();
        let len = (view.capacity_bits + 7) / 8;
        slice::from_raw_parts(data.as_ptr(), len)
    }
}

fn read_bit(view: &BitArrayView, index: usize) -> u8 {
    let bit_index = view.bit_offset + index;
    let byte_index = bit_index / 8;
    let bit_position = 7 - (bit_index % 8);
    let data = bit_array_data_slice(view);
    if byte_index >= data.len() {
        0
    } else {
        (data[byte_index] >> bit_position) & 1
    }
}

fn copy_bits(view: &BitArrayView, start: usize, len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    let mut bytes = vec![0u8; (len + 7) / 8];
    copy_bits_into(view, start, len, &mut bytes, 0);
    bytes
}

fn copy_bits_into(
    view: &BitArrayView,
    start: usize,
    len: usize,
    dest: &mut [u8],
    dest_offset: usize,
) {
    for i in 0..len {
        let bit = read_bit(view, start + i);
        if bit == 0 {
            continue;
        }
        let bit_index = dest_offset + i;
        let byte_index = bit_index / 8;
        let bit_position = 7 - (bit_index % 8);
        if let Some(byte) = dest.get_mut(byte_index) {
            *byte |= 1 << bit_position;
        }
    }
}

struct BitArrayBuilder {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl BitArrayBuilder {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_len: 0,
        }
    }

    fn push_bit(&mut self, bit: u8) {
        let byte_index = self.bit_len / 8;
        if byte_index == self.bytes.len() {
            self.bytes.push(0);
        }
        if bit != 0 {
            let offset = 7 - (self.bit_len % 8);
            if let Some(byte) = self.bytes.get_mut(byte_index) {
                *byte |= 1 << offset;
            }
        }
        self.bit_len += 1;
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if self.bit_len % 8 == 0 {
            self.bytes.extend_from_slice(bytes);
            self.bit_len += bytes.len() * 8;
        } else {
            for byte in bytes {
                for shift in (0..8).rev() {
                    let bit = (byte >> shift) & 1;
                    self.push_bit(bit);
                }
            }
        }
    }

    fn append_view_bits(&mut self, view: &BitArrayView, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let new_len = self.bit_len + len;
        let required_bytes = (new_len + 7) / 8;
        if self.bytes.len() < required_bytes {
            self.bytes.resize(required_bytes, 0);
        }
        copy_bits_into(view, start, len, &mut self.bytes, self.bit_len);
        self.bit_len = new_len;
    }

    fn append_bit_array_value(&mut self, value: Value, take_bits: Option<usize>) {
        if value == Value::nil() {
            return;
        }
        let view = bit_array_view(value, "bit array builder append");
        let available = view.bit_len;
        let len = match take_bits {
            Some(bits) => {
                if bits > available {
                    panic!("runtime bit_array builder segment size exceeds available bits");
                }
                bits
            }
            None => available,
        };
        self.append_view_bits(&view, 0, len);
    }

    fn append_int(
        &mut self,
        value: i64,
        size_bits: usize,
        signed: bool,
        endianness: BuilderEndianness,
    ) {
        if size_bits == 0 {
            return;
        }
        match endianness {
            BuilderEndianness::Big => {
                for index in (0..size_bits).rev() {
                    let bit = extract_int_bit(value, index, signed);
                    self.push_bit(bit);
                }
            }
            BuilderEndianness::Little => {
                for index in 0..size_bits {
                    let bit = extract_int_bit(value, index, signed);
                    self.push_bit(bit);
                }
            }
        }
    }
}

fn extract_int_bit(value: i64, index: usize, signed: bool) -> u8 {
    if index < 63 {
        if signed {
            ((value >> index) & 1) as u8
        } else {
            (((value as u64) >> index) & 1) as u8
        }
    } else if signed && value < 0 {
        1
    } else {
        0
    }
}

enum BuilderEndianness {
    Big,
    Little,
}

fn builder_from_raw(raw: u64) -> *mut BitArrayBuilder {
    raw as *mut BitArrayBuilder
}

fn builder_ref(raw: u64) -> &'static mut BitArrayBuilder {
    unsafe { &mut *builder_from_raw(raw) }
}

fn compute_size_bits(size: i64, unit: u64) -> usize {
    let unit_i64 =
        i64::try_from(unit).unwrap_or_else(|_| panic!("bit array segment unit overflow"));
    let product = (size as i128) * (unit_i64 as i128);
    if product <= 0 {
        0
    } else {
        usize::try_from(product)
            .unwrap_or_else(|_| panic!("bit array segment size exceeds native backend limits"))
    }
}

fn bit_array_from_bytes(bytes: &[u8], bit_len: usize) -> Value {
    gc::ensure_initialised();
    let heap = Heap::new();
    let data = unwrap_allocation(heap.alloc_binary_data(bytes.len()), "bit array data");
    unsafe {
        let buffer = &mut *data.as_ptr();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr(), bytes.len());
    }
    unwrap_allocation(
        heap.alloc_bit_array(data, 0, bit_len, bytes.len() * 8),
        "bit array",
    )
}

fn bit_array_bytes(view: &BitArrayView) -> Vec<u8> {
    copy_bits(view, 0, view.bit_len)
}

fn read_byte(view: &BitArrayView, index: usize) -> u8 {
    let start = index * 8;
    let bits_to_read = if start + 8 <= view.bit_len {
        8
    } else {
        view.bit_len.saturating_sub(start)
    };
    let mut value = 0u8;
    for i in 0..bits_to_read {
        let bit = read_bit(view, start + i);
        value |= bit << (7 - i);
    }
    value
}

fn padded_bytes(view: &BitArrayView) -> (Vec<u8>, usize) {
    let padding = (8 - (view.bit_len % 8)) % 8;
    let mut bytes = bit_array_bytes(view);
    let total_bits = view.bit_len + padding;
    if padding != 0 {
        let required_len = (total_bits + 7) / 8;
        if bytes.len() < required_len {
            bytes.resize(required_len, 0);
        }
    }
    (bytes, total_bits)
}

fn map_table_from_value(value: Value) -> NonNull<MapTable> {
    let map_ptr = value
        .as_boxed::<Map>()
        .unwrap_or_else(|| panic!("expected Dict value"));
    unsafe {
        let table = (*map_ptr.as_ptr()).table;
        NonNull::new(table).unwrap_or_else(|| panic!("dict missing table"))
    }
}

fn map_entries_vec(value: Value) -> Vec<(Value, Value)> {
    let table_ptr = map_table_from_value(value);
    unsafe {
        let table = table_ptr.as_ref();
        table
            .entries_slice()
            .iter()
            .map(|entry| (entry.key, entry.value))
            .collect()
    }
}

fn deduplicate_entries(entries: &mut Vec<(Value, Value)>) {
    let mut index = 0;
    while index < entries.len() {
        let key = entries[index].0;
        let mut cursor = index + 1;
        let mut final_value = entries[index].1;
        while cursor < entries.len() {
            if entries[cursor].0 == key {
                final_value = entries[cursor].1;
                entries.remove(cursor);
            } else {
                cursor += 1;
            }
        }
        entries[index].1 = final_value;
        index += 1;
    }
}

fn map_from_vec(mut entries: Vec<(Value, Value)>) -> Value {
    deduplicate_entries(&mut entries);
    let heap = Heap::new();
    let table = unwrap_allocation(heap.alloc_map_table(entries.len()), "map table");
    unsafe {
        let table_ref = table.as_ptr();
        let slice = (*table_ref).entries_slice_mut();
        for (slot, (key, value)) in slice.iter_mut().zip(entries.iter()) {
            *slot = MapEntry {
                key: *key,
                value: *value,
            };
        }
    }
    unwrap_allocation(heap.alloc_map(table), "map")
}

fn call_function(function: Value, args: &[Value]) -> Value {
    let closure_ptr = function
        .as_boxed::<Closure>()
        .unwrap_or_else(|| panic!("expected function value"));
    unsafe {
        let closure = closure_ptr.as_ptr();
        let func = (*closure).code_ptr;
        let env_ptr = (*closure).env.as_ptr();
        func(env_ptr, args.as_ptr(), args.len())
    }
}

fn tuple_to_vec(value: Value) -> Vec<Value> {
    let header_ptr = value
        .as_boxed::<Header>()
        .unwrap_or_else(|| panic!("expected tuple value"));
    let header = unsafe { header_ptr.as_ref() };
    if header.tag() != Tag::Tuple {
        panic!("expected tuple value");
    }
    let len = header.arity() as usize;
    let payload_ptr =
        unsafe { (header_ptr.as_ptr() as *const u8).add(mem::size_of::<Header>()) as *const Value };
    unsafe { slice::from_raw_parts(payload_ptr, len) }.to_vec()
}

fn header_tag(value: Value) -> Option<Tag> {
    value
        .as_boxed::<Header>()
        .map(|ptr| unsafe { ptr.as_ref().tag() })
}

fn is_bool_value(value: Value) -> bool {
    value == Value::from_bool(true) || value == Value::from_bool(false)
}

#[allow(unreachable_patterns)]
fn classify_value(value: Value) -> &'static str {
    if value == Value::nil() {
        "Nil"
    } else if is_bool_value(value) {
        "Bool"
    } else if value.is_atom() {
        "Atom"
    } else if value.is_i63() {
        "Int"
    } else if let Some(tag) = header_tag(value) {
        match tag {
            Tag::Binary | Tag::BinarySlice => "String",
            Tag::BitArray => "BitArray",
            Tag::Float => "Float",
            Tag::List => "List",
            Tag::Map => "Dict",
            Tag::Tuple => "Array",
            Tag::Record => "Record",
            Tag::Closure => "Function",
            Tag::Resource => "Resource",
            Tag::Mailbox => "Mailbox",
            Tag::Boolean => "Bool",
            Tag::Nil => "Nil",
            _ => "Unknown",
        }
    } else {
        "Unknown"
    }
}

fn is_list_value(value: Value) -> bool {
    if value == Value::nil() {
        true
    } else {
        matches!(header_tag(value), Some(Tag::List))
    }
}

fn is_empty_list(value: Value) -> bool {
    value == Value::nil()
}

fn decode_error_record(expected: &str, found: &str, path: Value) -> Value {
    let expected_value = string_to_value(expected);
    let found_value = string_to_value(found);
    let elements = [expected_value, found_value, path];
    tuple_from(&elements, "decode error tuple")
}

fn decode_error_list(expected: &str, data: Value) -> Value {
    let found = classify_value(data);
    let error = decode_error_record(expected, found, Value::nil());
    list_from_vec(vec![error])
}

#[no_mangle]
pub extern "C" fn classify_dynamic(raw: u64) -> u64 {
    let value = Value::from_raw(raw);
    string_to_value(classify_value(value)).to_raw()
}

#[no_mangle]
pub extern "C" fn list_to_array(list_raw: u64) -> u64 {
    let elements = list_to_vec(Value::from_raw(list_raw));
    tuple_from(&elements, "list_to_array tuple").to_raw()
}

#[no_mangle]
pub extern "C" fn is_null(raw: u64) -> u64 {
    let value = Value::from_raw(raw);
    Value::from_bool(value == Value::nil()).to_raw()
}

#[derive(Debug)]
enum FloatAccessError {
    NotAFloat,
}

fn value_to_f64(value: Value) -> Result<f64, FloatAccessError> {
    let ptr = value
        .as_boxed::<FloatBox>()
        .ok_or(FloatAccessError::NotAFloat)?;
    let float = unsafe { ptr.as_ref() };
    Ok(float.value)
}

const MIN_I63: i64 = -(1i64 << 61);
const MAX_I63: i64 = (1i64 << 61) - 1;

fn is_pattern_whitespace(ch: char) -> bool {
    matches!(
        ch,
        '\u{0020}'
            | '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0085}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

fn float_to_value(number: f64) -> Value {
    gc::ensure_initialised();
    let heap = Heap::new();
    unwrap_allocation(heap.alloc_float(number), "float")
}

fn float_to_i63_value(number: f64, context: &'static str) -> Value {
    if !number.is_finite() {
        panic!("runtime {context} produced non-finite value");
    }

    let truncated = number.trunc();
    if truncated < (MIN_I63 as f64) || truncated > (MAX_I63 as f64) {
        panic!("runtime {context} result out of i63 range");
    }

    Value::from_i63(truncated as i64)
}

fn value_to_i63(value: Value, context: &'static str) -> i64 {
    value
        .to_i63()
        .unwrap_or_else(|| panic!("runtime {context} expected small int"))
}

fn ensure_i63_range(value: i128, context: &'static str) -> Value {
    if value < MIN_I63 as i128 || value > MAX_I63 as i128 {
        panic!("runtime {context} result out of i63 range");
    }
    Value::from_i63(value as i64)
}

fn tuple_from(elements: &[Value], context: &'static str) -> Value {
    gc::ensure_initialised();
    let heap = Heap::new();
    unwrap_allocation(heap.alloc_tuple(elements), context)
}

fn result_with(tag: Value, payload: Value) -> Value {
    tuple_from(&[tag, payload], "result tuple")
}

fn result_ok(payload: Value) -> u64 {
    result_with(atom_ok(), payload).to_raw()
}

fn result_error(payload: Value) -> u64 {
    result_with(atom_error(), payload).to_raw()
}

fn is_percent_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'$' | b'\'' | b'(' | b')' | b'*' | b'+' | b'-' | b'.' | b'_' | b'~'
        )
}

fn percent_encode_string(input: &str) -> String {
    let mut buffer = String::with_capacity(input.len());
    for byte in input.bytes() {
        if is_percent_unreserved(byte) {
            buffer.push(byte as char);
        } else {
            buffer.push('%');
            buffer.push(HEX_DIGITS[(byte >> 4) as usize]);
            buffer.push(HEX_DIGITS[(byte & 0x0F) as usize]);
        }
    }
    buffer
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn percent_decode_string(input: &str, plus_to_space: bool) -> Result<String, ()> {
    let mut bytes = Vec::with_capacity(input.len());
    let input_bytes = input.as_bytes();
    let mut index = 0;
    while index < input_bytes.len() {
        let byte = input_bytes[index];
        if byte == b'%' {
            if index + 2 >= input_bytes.len() {
                return Err(());
            }
            let hi = hex_value(input_bytes[index + 1]).ok_or(())?;
            let lo = hex_value(input_bytes[index + 2]).ok_or(())?;
            bytes.push((hi << 4) | lo);
            index += 3;
        } else {
            if plus_to_space && byte == b'+' {
                bytes.push(b' ');
            } else {
                bytes.push(byte);
            }
            index += 1;
        }
    }
    String::from_utf8(bytes).map_err(|_| ())
}

fn value_to_text(value: Value) -> Option<String> {
    if let Ok(text) = value_to_string(value) {
        Some(text)
    } else if let Some(index) = value.atom_index() {
        atom_table()
            .resolve(index)
            .map(|atom| atom.as_ref().to_string())
    } else {
        None
    }
}

fn map_get_field(entries: &[(Value, Value)], name: &str) -> Option<Value> {
    let atom_key = atom(name);
    for (key, value) in entries {
        if *key == atom_key {
            return Some(*value);
        }
        if let Ok(text) = value_to_string(*key) {
            if text == name {
                return Some(*value);
            }
        } else if let Some(index) = key.atom_index() {
            if let Some(atom_name) = atom_table().resolve(index) {
                if atom_name.as_ref() == name {
                    return Some(*value);
                }
            }
        }
    }
    None
}

fn map_get_text(entries: &[(Value, Value)], name: &str) -> Result<String, ()> {
    let value = map_get_field(entries, name).ok_or(())?;
    value_to_text(value).ok_or(())
}

fn map_get_int(entries: &[(Value, Value)], name: &str) -> Result<i64, ()> {
    let value = map_get_field(entries, name).ok_or(())?;
    value.to_i63().ok_or(())
}

fn map_get_list(entries: &[(Value, Value)], name: &str) -> Result<Vec<Value>, ()> {
    let value = map_get_field(entries, name).ok_or(())?;
    if !is_list_value(value) {
        return Err(());
    }
    Ok(list_to_vec(value))
}

fn build_expression(value: Value) -> Result<Value, ()> {
    if !matches!(header_tag(value), Some(Tag::Map)) {
        return Err(());
    }
    let entries = map_entries_vec(value);
    let start = map_get_int(&entries, "start")?;
    let end = map_get_int(&entries, "end")?;
    let kind_name = map_get_text(&entries, "kind")?;
    let kind_value = match kind_name.as_str() {
        "literal" => {
            let literal = map_get_field(&entries, "value").ok_or(())?;
            tuple_from(&[atom("literal"), literal], "assert expression literal")
        }
        "expression" => {
            let expr = map_get_field(&entries, "value").ok_or(())?;
            tuple_from(&[atom("expression"), expr], "assert expression value")
        }
        "unevaluated" => atom("unevaluated"),
        _ => return Err(()),
    };
    Ok(tuple_from(
        &[
            atom("asserted_expression"),
            Value::from_i63(start),
            Value::from_i63(end),
            kind_value,
        ],
        "asserted expression",
    ))
}

fn build_assert_kind(entries: &[(Value, Value)]) -> Result<Value, ()> {
    let kind_name = map_get_text(entries, "kind")?;
    match kind_name.as_str() {
        "binary_operator" => {
            let operator = map_get_text(entries, "operator")?;
            let left = map_get_field(entries, "left").ok_or(())?;
            let right = map_get_field(entries, "right").ok_or(())?;
            let left_expr = build_expression(left)?;
            let right_expr = build_expression(right)?;
            Ok(tuple_from(
                &[
                    atom("binary_operator"),
                    string_to_value(&operator),
                    left_expr,
                    right_expr,
                ],
                "assert binary operator",
            ))
        }
        "function_call" => {
            let arguments = map_get_list(entries, "arguments")?;
            let mut expressions = Vec::with_capacity(arguments.len());
            for argument in arguments {
                expressions.push(build_expression(argument)?);
            }
            let list_value = list_from_vec(expressions);
            Ok(tuple_from(
                &[atom("function_call"), list_value],
                "assert function call",
            ))
        }
        _ => {
            let expression_value = map_get_field(entries, "expression").ok_or(())?;
            let expression = build_expression(expression_value)?;
            Ok(tuple_from(
                &[atom("other_expression"), expression],
                "assert other expression",
            ))
        }
    }
}

fn wrap_gleam_panic(entries: &[(Value, Value)], kind: Value) -> Result<Value, ()> {
    let message = map_get_text(entries, "message")?;
    let file = map_get_text(entries, "file")?;
    let module = map_get_text(entries, "module")?;
    let function_name = map_get_text(entries, "function")?;
    let line = map_get_int(entries, "line")?;
    let elements = [
        atom("gleam_panic"),
        string_to_value(&message),
        string_to_value(&file),
        string_to_value(&module),
        string_to_value(&function_name),
        Value::from_i63(line),
        kind,
    ];
    Ok(tuple_from(&elements, "gleeunit panic"))
}

fn option_some(value: Value) -> Value {
    tuple_from(&[atom("some"), value], "option some")
}

fn option_none() -> Value {
    atom("none")
}

#[derive(Clone, Copy)]
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

#[no_mangle]
pub extern "C" fn print(raw: u64) -> u64 {
    runtime_print(raw, false, OutputStream::Stdout)
}

#[no_mangle]
pub extern "C" fn println(raw: u64) -> u64 {
    runtime_print(raw, true, OutputStream::Stdout)
}

#[no_mangle]
pub extern "C" fn print_error(raw: u64) -> u64 {
    runtime_print(raw, false, OutputStream::Stderr)
}

#[no_mangle]
pub extern "C" fn println_error(raw: u64) -> u64 {
    runtime_print(raw, true, OutputStream::Stderr)
}

// TODO: Remove legacy `io_*` exports once lowering is updated to call the new names.
#[no_mangle]
pub extern "C" fn io_print(raw: u64) -> u64 {
    print(raw)
}

#[no_mangle]
pub extern "C" fn io_println(raw: u64) -> u64 {
    println(raw)
}

#[no_mangle]
pub extern "C" fn io_print_error(raw: u64) -> u64 {
    print_error(raw)
}

#[no_mangle]
pub extern "C" fn io_println_error(raw: u64) -> u64 {
    println_error(raw)
}

#[no_mangle]
pub extern "C" fn parse_float(raw: u64) -> u64 {
    let input = Value::from_raw(raw);
    let parsed = value_to_string(input)
        .unwrap_or_else(|_| panic!("expected String value"))
        .trim()
        .parse::<f64>()
        .ok();

    match parsed {
        Some(number) => {
            let heap = Heap::new();
            let float_value = unwrap_allocation(heap.alloc_float(number), "float parse value");
            result_ok(float_value)
        }
        None => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn float_to_string(raw: u64) -> u64 {
    let number =
        value_to_f64(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected Float value"));
    let string = number.to_string();
    bytes_to_value(string.as_bytes()).to_raw()
}

#[no_mangle]
pub extern "C" fn ceiling(raw: u64) -> u64 {
    let number =
        value_to_f64(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected Float value"));
    float_to_value(number.ceil()).to_raw()
}

#[no_mangle]
pub extern "C" fn floor(raw: u64) -> u64 {
    let number =
        value_to_f64(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected Float value"));
    float_to_value(number.floor()).to_raw()
}

#[no_mangle]
pub extern "C" fn round(raw: u64) -> u64 {
    let number =
        value_to_f64(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected Float value"));
    float_to_i63_value(number.round(), "round").to_raw()
}

#[no_mangle]
pub extern "C" fn truncate(raw: u64) -> u64 {
    let number =
        value_to_f64(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected Float value"));
    float_to_i63_value(number, "truncate").to_raw()
}

#[no_mangle]
pub extern "C" fn float(raw: u64) -> u64 {
    let int_value = value_to_i63(Value::from_raw(raw), "float_from_int");
    float_to_value(int_value as f64).to_raw()
}

#[no_mangle]
pub extern "C" fn power(base_raw: u64, exponent_raw: u64) -> u64 {
    let base =
        value_to_f64(Value::from_raw(base_raw)).unwrap_or_else(|_| panic!("expected Float value"));
    let exponent = value_to_f64(Value::from_raw(exponent_raw))
        .unwrap_or_else(|_| panic!("expected Float value"));
    float_to_value(base.powf(exponent)).to_raw()
}

#[no_mangle]
pub extern "C" fn random_uniform() -> u64 {
    let mut rng = rand::thread_rng();
    let value: f64 = rng.r#gen::<f64>();
    float_to_value(value).to_raw()
}

#[no_mangle]
pub extern "C" fn log(raw: u64) -> u64 {
    let number =
        value_to_f64(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected Float value"));
    float_to_value(number.ln()).to_raw()
}

#[no_mangle]
pub extern "C" fn exp(raw: u64) -> u64 {
    let number =
        value_to_f64(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected Float value"));
    float_to_value(number.exp()).to_raw()
}

#[no_mangle]
pub extern "C" fn string_length(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let count = UnicodeSegmentation::graphemes(string.as_str(), true).count();
    let count = i64::try_from(count).unwrap_or_else(|_| panic!("runtime string_length overflow"));
    Value::from_i63(count).to_raw()
}

#[no_mangle]
pub extern "C" fn lowercase(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let lower = string.to_lowercase();
    string_to_value(&lower).to_raw()
}

#[no_mangle]
pub extern "C" fn uppercase(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let upper = string.to_uppercase();
    string_to_value(&upper).to_raw()
}

#[no_mangle]
pub extern "C" fn less_than(left_raw: u64, right_raw: u64) -> u64 {
    let left = value_to_string(Value::from_raw(left_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let right = value_to_string(Value::from_raw(right_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    Value::from_bool(left < right).to_raw()
}

#[no_mangle]
pub extern "C" fn string_slice(string_raw: u64, idx_raw: u64, len_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let idx = value_to_i63(Value::from_raw(idx_raw), "string_slice index");
    let len = value_to_i63(Value::from_raw(len_raw), "string_slice length");

    if len <= 0 {
        return Value::nil().to_raw();
    }

    let start = usize::try_from(idx.max(0))
        .unwrap_or_else(|_| panic!("runtime string_slice index overflow"));
    let len =
        usize::try_from(len).unwrap_or_else(|_| panic!("runtime string_slice length overflow"));
    let slice = UnicodeSegmentation::graphemes(string.as_str(), true)
        .skip(start)
        .take(len)
        .collect::<String>();
    string_to_value(&slice).to_raw()
}

#[no_mangle]
pub extern "C" fn crop_string(string_raw: u64, prefix_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let prefix = value_to_string(Value::from_raw(prefix_raw))
        .unwrap_or_else(|_| panic!("expected String value"));

    match string.find(prefix.as_str()) {
        Some(index) => string_to_value(&string[index..]).to_raw(),
        None => string_to_value(string.as_str()).to_raw(),
    }
}

#[no_mangle]
pub extern "C" fn contains_string(haystack_raw: u64, needle_raw: u64) -> u64 {
    let haystack = value_to_string(Value::from_raw(haystack_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let needle = value_to_string(Value::from_raw(needle_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    Value::from_bool(haystack.contains(needle.as_str())).to_raw()
}

#[no_mangle]
pub extern "C" fn string_starts_with(string_raw: u64, prefix_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let prefix = value_to_string(Value::from_raw(prefix_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    Value::from_bool(string.starts_with(prefix.as_str())).to_raw()
}

#[no_mangle]
pub extern "C" fn string_prefix_split(string_raw: u64, prefix_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let prefix = value_to_string(Value::from_raw(prefix_raw))
        .unwrap_or_else(|_| panic!("expected String value"));

    if let Some(rest) = string.strip_prefix(prefix.as_str()) {
        let prefix_value = string_to_value(prefix.as_str());
        let rest_value = string_to_value(rest);
        let tuple = tuple_from(
            &[Value::from_bool(true), prefix_value, rest_value],
            "string prefix split tuple",
        );
        tuple.to_raw()
    } else {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "string prefix split failure",
        );
        tuple.to_raw()
    }
}

#[no_mangle]
pub extern "C" fn string_ends_with(string_raw: u64, suffix_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let suffix = value_to_string(Value::from_raw(suffix_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    Value::from_bool(string.ends_with(suffix.as_str())).to_raw()
}

#[no_mangle]
pub extern "C" fn split_once(string_raw: u64, needle_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let needle = value_to_string(Value::from_raw(needle_raw))
        .unwrap_or_else(|_| panic!("expected String value"));

    if let Some(index) = string.find(needle.as_str()) {
        let before = &string[..index];
        let after = &string[index + needle.len()..];
        let pair = tuple_from(
            &[string_to_value(before), string_to_value(after)],
            "split_once pair",
        );
        result_ok(pair)
    } else {
        result_error(Value::nil())
    }
}

#[no_mangle]
pub extern "C" fn trim_start(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let trimmed = string.trim_start_matches(is_pattern_whitespace);
    string_to_value(trimmed).to_raw()
}

#[no_mangle]
pub extern "C" fn trim_end(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let trimmed = string.trim_end_matches(is_pattern_whitespace);
    string_to_value(trimmed).to_raw()
}

#[no_mangle]
pub extern "C" fn pop_grapheme(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let mut iter = UnicodeSegmentation::grapheme_indices(string.as_str(), true);
    if let Some((start, first)) = iter.next() {
        let end = start + first.len();
        let rest = &string[end..];
        let pair = tuple_from(
            &[string_to_value(first), string_to_value(rest)],
            "string pop grapheme tuple",
        );
        result_ok(pair)
    } else {
        result_error(Value::nil())
    }
}

#[no_mangle]
pub extern "C" fn byte_size(raw: u64) -> u64 {
    let bytes =
        value_to_bytes(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let len = i64::try_from(bytes.len()).unwrap_or_else(|_| panic!("runtime byte_size overflow"));
    Value::from_i63(len).to_raw()
}

#[no_mangle]
pub extern "C" fn utf_codepoint_to_int(raw: u64) -> u64 {
    raw
}

#[no_mangle]
pub extern "C" fn bit_array_bit_size(raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(raw), "bit_array_bit_size");
    let size = i64::try_from(view.bit_len)
        .unwrap_or_else(|_| panic!("runtime bit_array_bit_size overflow"));
    Value::from_i63(size).to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_byte_size(raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(raw), "bit_array_byte_size");
    let bytes = (view.bit_len + 7) / 8;
    let size =
        i64::try_from(bytes).unwrap_or_else(|_| panic!("runtime bit_array_byte_size overflow"));
    Value::from_i63(size).to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_pad_to_bytes(raw: u64) -> u64 {
    let value = Value::from_raw(raw);
    let view = bit_array_view(value, "bit_array_pad_to_bytes");
    let padding = (8 - (view.bit_len % 8)) % 8;
    if padding == 0 && view.bit_offset == 0 {
        return raw;
    }
    let (bytes, total_bits) = padded_bytes(&view);
    bit_array_from_bytes(&bytes, total_bits).to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_slice(bits_raw: u64, pos_raw: u64, len_raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(bits_raw), "bit_array_slice bits");
    let pos = value_to_i63(Value::from_raw(pos_raw), "bit_array_slice position");
    let len = value_to_i63(Value::from_raw(len_raw), "bit_array_slice length");
    let start = pos.min(pos + len);
    let end = pos.max(pos + len);
    if start < 0 || end < 0 {
        return result_error(Value::nil());
    }
    let start_usize =
        usize::try_from(start).unwrap_or_else(|_| panic!("runtime bit_array_slice start overflow"));
    let end_usize =
        usize::try_from(end).unwrap_or_else(|_| panic!("runtime bit_array_slice end overflow"));
    if end_usize.saturating_mul(8) > view.bit_len {
        return result_error(Value::nil());
    }
    let start_bits = start_usize * 8;
    let len_bits = (end_usize - start_usize) * 8;
    let bytes = copy_bits(&view, start_bits, len_bits);
    let slice_value = bit_array_from_bytes(&bytes, len_bits);
    result_ok(slice_value)
}

#[no_mangle]
pub extern "C" fn bit_array_to_string(raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(raw), "bit_array_to_string");
    if view.bit_len % 8 != 0 {
        return result_error(Value::nil());
    }
    let bytes = bit_array_bytes(&view);
    match String::from_utf8(bytes) {
        Ok(string) => result_ok(string_to_value(&string)),
        Err(_) => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn bit_array_unsafe_to_string(raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(raw), "bit_array_unsafe_to_string");
    if view.bit_len % 8 != 0 {
        panic!("runtime bit_array_unsafe_to_string expected byte-aligned bit array");
    }
    let bytes = bit_array_bytes(&view);
    let string = String::from_utf8(bytes)
        .unwrap_or_else(|_| panic!("runtime bit_array_unsafe_to_string invalid utf-8"));
    string_to_value(&string).to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_concat(list_raw: u64) -> u64 {
    let values = list_to_vec(Value::from_raw(list_raw));
    let mut total_bits: usize = 0;
    for value in &values {
        let view = bit_array_view(*value, "bit_array_concat input");
        total_bits = total_bits
            .checked_add(view.bit_len)
            .unwrap_or_else(|| panic!("runtime bit_array_concat overflow"));
    }
    if total_bits == 0 {
        return bit_array_from_bytes(&[], 0).to_raw();
    }
    let mut bytes = vec![0u8; (total_bits + 7) / 8];
    let mut offset = 0usize;
    for value in values {
        let view = bit_array_view(value, "bit_array_concat entry");
        copy_bits_into(&view, 0, view.bit_len, &mut bytes, offset);
        offset += view.bit_len;
    }
    bit_array_from_bytes(&bytes, total_bits).to_raw()
}

#[no_mangle]
pub extern "C" fn base64_encode(bits_raw: u64, padding_raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(bits_raw), "base64_encode bits");
    let padding = Value::from_raw(padding_raw) == Value::from_bool(true);
    let (bytes, _) = padded_bytes(&view);
    let engine = if padding { &STANDARD } else { &STANDARD_NO_PAD };
    let encoded = engine.encode(&bytes);
    string_to_value(&encoded).to_raw()
}

#[no_mangle]
pub extern "C" fn base64_decode(string_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("runtime base64_decode expected String value"));
    let decoded = STANDARD
        .decode(string.as_bytes())
        .or_else(|_| STANDARD_NO_PAD.decode(string.as_bytes()));
    match decoded {
        Ok(bytes) => {
            let value = bit_array_from_bytes(&bytes, bytes.len() * 8);
            result_ok(value)
        }
        Err(_) => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn base16_encode(bits_raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(bits_raw), "base16_encode bits");
    let (bytes, _) = padded_bytes(&view);
    let encoded = encode_upper(bytes);
    string_to_value(&encoded).to_raw()
}

#[no_mangle]
pub extern "C" fn base16_decode(string_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("runtime base16_decode expected String value"));
    if string.len() % 2 != 0 {
        return result_error(Value::nil());
    }
    match hex_decode(string.as_bytes()) {
        Ok(bytes) => {
            let value = bit_array_from_bytes(&bytes, bytes.len() * 8);
            result_ok(value)
        }
        Err(_) => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn bit_array_to_int_and_size(raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(raw), "bit_array_to_int_and_size");
    let first_byte = if view.bit_len == 0 {
        0
    } else {
        read_byte(&view, 0)
    };
    let trailing_bits = view.bit_len % 8;
    let unused_bits = if trailing_bits == 0 {
        0
    } else {
        8 - trailing_bits
    };
    let value = (first_byte >> unused_bits) as i64;
    let size = i64::try_from(view.bit_len)
        .unwrap_or_else(|_| panic!("runtime bit_array_to_int_and_size overflow"));
    let tuple = tuple_from(
        &[Value::from_i63(value), Value::from_i63(size)],
        "bit array to int tuple",
    );
    tuple.to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_starts_with(bits_raw: u64, prefix_raw: u64) -> u64 {
    let bits_view = bit_array_view(Value::from_raw(bits_raw), "bit_array_starts_with bits");
    let prefix_view = bit_array_view(Value::from_raw(prefix_raw), "bit_array_starts_with prefix");
    if prefix_view.bit_len > bits_view.bit_len {
        return Value::from_bool(false).to_raw();
    }
    for index in 0..prefix_view.bit_len {
        if read_bit(&prefix_view, index) != read_bit(&bits_view, index) {
            return Value::from_bool(false).to_raw();
        }
    }
    Value::from_bool(true).to_raw()
}

#[no_mangle]
pub extern "C" fn index(data_raw: u64, key_raw: u64) -> u64 {
    let data = Value::from_raw(data_raw);
    let key = Value::from_raw(key_raw);

    if matches!(header_tag(data), Some(Tag::Map)) {
        let entries = map_entries_vec(data);
        for (entry_key, entry_value) in entries {
            if entry_key == key {
                return result_ok(option_some(entry_value));
            }
        }
        return result_ok(option_none());
    }

    if key.is_i63() {
        let index = value_to_i63(key, "index key");
        if index < 0 {
            return result_error(string_to_value("Indexable"));
        }
        let target = index as usize;

        if is_list_value(data) {
            let mut current = data;
            let mut current_index = 0usize;
            while current != Value::nil() {
                let cons_ptr = current
                    .as_boxed::<ConsCell>()
                    .unwrap_or_else(|| panic!("expected List value"));
                let cons = unsafe { cons_ptr.as_ref() };
                if current_index == target {
                    return result_ok(option_some(cons.head));
                }
                current = cons.tail;
                current_index += 1;
            }
            return result_ok(option_none());
        }

        if matches!(header_tag(data), Some(Tag::Tuple)) {
            let elements = tuple_to_vec(data);
            if let Some(value) = elements.get(target) {
                return result_ok(option_some(*value));
            } else {
                return result_ok(option_none());
            }
        }

        return result_error(string_to_value("Indexable"));
    }

    result_error(string_to_value("Dict"))
}

#[no_mangle]
pub extern "C" fn dynamic_string(data_raw: u64) -> u64 {
    let value = Value::from_raw(data_raw);
    match header_tag(value) {
        Some(Tag::Binary) | Some(Tag::BinarySlice) => result_ok(value),
        Some(Tag::BitArray) => {
            let view = bit_array_view(value, "string bit array view");
            if view.bit_len % 8 != 0 {
                return result_error(string_to_value(""));
            }
            let bytes = bit_array_bytes(&view);
            match String::from_utf8(bytes) {
                Ok(text) => result_ok(string_to_value(&text)),
                Err(_) => result_error(string_to_value("")),
            }
        }
        _ => result_error(string_to_value("")),
    }
}

#[no_mangle]
pub extern "C" fn dynamic_int(data_raw: u64) -> u64 {
    let value = Value::from_raw(data_raw);
    if value.is_i63() {
        result_ok(value)
    } else {
        result_error(Value::from_i63(0))
    }
}

#[no_mangle]
pub extern "C" fn dynamic_float(data_raw: u64) -> u64 {
    let value = Value::from_raw(data_raw);
    if matches!(header_tag(value), Some(Tag::Float)) {
        result_ok(value)
    } else {
        result_error(float_to_value(0.0))
    }
}

#[no_mangle]
pub extern "C" fn dynamic_bit_array(data_raw: u64) -> u64 {
    let value = Value::from_raw(data_raw);
    if matches!(header_tag(value), Some(Tag::BitArray)) {
        result_ok(value)
    } else {
        let empty = bit_array_from_bytes(&[], 0);
        result_error(empty)
    }
}

#[no_mangle]
pub extern "C" fn decode_list(
    data_raw: u64,
    item_raw: u64,
    push_path_raw: u64,
    index_raw: u64,
    acc_raw: u64,
) -> u64 {
    let data = Value::from_raw(data_raw);
    let item_fn = Value::from_raw(item_raw);
    let push_path_fn = Value::from_raw(push_path_raw);
    let mut index = value_to_i63(Value::from_raw(index_raw), "list index");
    if index < 0 {
        index = 0;
    }

    let mut acc_vec = list_to_vec(Value::from_raw(acc_raw));

    let elements_opt = if data == Value::nil() {
        Some(Vec::new())
    } else if matches!(header_tag(data), Some(Tag::Tuple)) {
        Some(tuple_to_vec(data))
    } else if is_list_value(data) {
        Some(list_to_vec(data))
    } else {
        None
    };

    if let Some(elements) = elements_opt {
        let mut current_index = index as usize;
        for element in elements {
            let decoded = call_function(item_fn, &[element]);
            let parts = tuple_to_vec(decoded);
            if parts.len() != 2 {
                panic!("decoder tuple expected two elements");
            }
            let value = parts[0];
            let errors = parts[1];
            if is_empty_list(errors) {
                acc_vec.push(value);
                current_index += 1;
            } else {
                let tuple_arg = tuple_from(&[Value::nil(), errors], "list push tuple");
                let index_text = current_index.to_string();
                let index_string = string_to_value(index_text.as_str());
                let result = call_function(push_path_fn, &[tuple_arg, index_string]);
                return result.to_raw();
            }
        }
        let result_list = list_from_vec(acc_vec);
        let result = tuple_from(&[result_list, Value::nil()], "list result tuple");
        return result.to_raw();
    }

    if acc_vec.is_empty() {
        let errors = decode_error_list("List", data);
        let tuple = tuple_from(&[list_from_vec(Vec::new()), errors], "list invalid tuple");
        tuple.to_raw()
    } else {
        let result_list = list_from_vec(acc_vec);
        let tuple = tuple_from(&[result_list, Value::nil()], "list partial tuple");
        tuple.to_raw()
    }
}

#[no_mangle]
pub extern "C" fn dynamic_dict(data_raw: u64) -> u64 {
    let value = Value::from_raw(data_raw);
    if matches!(header_tag(value), Some(Tag::Map)) {
        result_ok(value)
    } else {
        result_error(Value::nil())
    }
}

#[no_mangle]
pub extern "C" fn utf_codepoint_list_to_string(raw: u64) -> u64 {
    let elements = list_to_vec(Value::from_raw(raw));
    let mut buffer = String::new();
    for value in elements {
        let codepoint = value_to_i63(value, "utf_codepoint_list_to_string");
        if let Some(ch) = char::from_u32(codepoint as u32) {
            buffer.push(ch);
        } else {
            panic!("runtime utf_codepoint_list_to_string invalid codepoint");
        }
    }
    string_to_value(&buffer).to_raw()
}

#[no_mangle]
pub extern "C" fn add(left_raw: u64, right_raw: u64) -> u64 {
    let left = value_to_string(Value::from_raw(left_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let right = value_to_string(Value::from_raw(right_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let mut buffer = left;
    buffer.push_str(right.as_str());
    string_to_value(&buffer).to_raw()
}

#[no_mangle]
pub extern "C" fn concat(list_raw: u64) -> u64 {
    let elements = list_to_vec(Value::from_raw(list_raw));
    let mut buffer = String::new();
    for element in elements {
        let string = value_to_string(element).unwrap_or_else(|_| panic!("expected String value"));
        buffer.push_str(string.as_str());
    }
    string_to_value(&buffer).to_raw()
}

#[no_mangle]
pub extern "C" fn string_replace(string_raw: u64, pattern_raw: u64, substitute_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let pattern = value_to_string(Value::from_raw(pattern_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let substitute = value_to_string(Value::from_raw(substitute_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let replaced = string.replace(pattern.as_str(), substitute.as_str());
    string_to_value(&replaced).to_raw()
}

#[no_mangle]
pub extern "C" fn string_to_utf8_bits(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    bit_array_from_bytes(string.as_bytes(), string.len() * 8).to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_pop_utf8_codepoint(raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(raw), "bit_array_pop_utf8_codepoint");

    if view.bit_offset % 8 != 0 {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "bit array utf8 split failure",
        );
        return tuple.to_raw();
    }

    let bytes = bit_array_bytes(&view);
    if bytes.is_empty() {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "bit array utf8 split failure",
        );
        return tuple.to_raw();
    }

    let string = match core::str::from_utf8(&bytes) {
        Ok(value) => value,
        Err(_) => {
            let tuple = tuple_from(
                &[Value::from_bool(false), Value::nil(), Value::nil()],
                "bit array utf8 split failure",
            );
            return tuple.to_raw();
        }
    };

    let mut chars = string.chars();
    let Some(first) = chars.next() else {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "bit array utf8 split failure",
        );
        return tuple.to_raw();
    };

    let first_len = first.len_utf8();
    let codepoint = Value::from_i63(first as u32 as i64);
    let rest_bytes = &bytes[first_len..];
    let rest_value = bit_array_from_bytes(rest_bytes, rest_bytes.len() * 8);

    let tuple = tuple_from(
        &[Value::from_bool(true), codepoint, rest_value],
        "bit array utf8 split tuple",
    );
    tuple.to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_pop_byte(raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(raw), "bit_array_pop_byte");

    if view.bit_len < 8 {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "bit array byte split failure",
        );
        return tuple.to_raw();
    }

    let byte = read_byte(&view, 0);
    let remaining_bits = view.bit_len - 8;
    let rest_bytes = copy_bits(&view, 8, remaining_bits);
    let rest_value = bit_array_from_bytes(&rest_bytes, remaining_bits);

    let tuple = tuple_from(
        &[
            Value::from_bool(true),
            Value::from_i63(byte as i64),
            rest_value,
        ],
        "bit array byte split tuple",
    );
    tuple.to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_split_bits(bits_raw: u64, size_raw: u64) -> u64 {
    let view = bit_array_view(Value::from_raw(bits_raw), "bit_array_split_bits bits");
    let size_value = value_to_i63(Value::from_raw(size_raw), "bit_array_split_bits size");

    if size_value < 0 {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "bit array split bits failure",
        );
        return tuple.to_raw();
    }

    let Ok(prefix_len) = usize::try_from(size_value) else {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "bit array split bits failure",
        );
        return tuple.to_raw();
    };

    if prefix_len > view.bit_len {
        let tuple = tuple_from(
            &[Value::from_bool(false), Value::nil(), Value::nil()],
            "bit array split bits failure",
        );
        return tuple.to_raw();
    }

    let prefix_bytes = copy_bits(&view, 0, prefix_len);
    let prefix_value = bit_array_from_bytes(&prefix_bytes, prefix_len);

    let rest_len = view.bit_len - prefix_len;
    let rest_bytes = copy_bits(&view, prefix_len, rest_len);
    let rest_value = bit_array_from_bytes(&rest_bytes, rest_len);

    let tuple = tuple_from(
        &[Value::from_bool(true), prefix_value, rest_value],
        "bit array split bits tuple",
    );
    tuple.to_raw()
}

#[no_mangle]
pub extern "C" fn bit_array_builder_new() -> u64 {
    let builder = Box::new(BitArrayBuilder::new());
    Box::into_raw(builder) as u64
}

#[no_mangle]
pub extern "C" fn bit_array_builder_append_bit_array(
    builder_raw: u64,
    bits_raw: u64,
    size_raw: u64,
    has_size_raw: u64,
    unit_raw: u64,
) -> u64 {
    let builder = builder_ref(builder_raw);
    let has_size = has_size_raw != 0;
    let unit = unit_raw as u64;
    let take_bits = if has_size {
        let size_value = value_to_i63(Value::from_raw(size_raw), "bit array segment size");
        let bits = compute_size_bits(size_value, unit);
        Some(bits)
    } else {
        None
    };
    let value = Value::from_raw(bits_raw);
    builder.append_bit_array_value(value, take_bits);
    builder_raw
}

#[no_mangle]
pub extern "C" fn bit_array_builder_append_int(
    builder_raw: u64,
    value_raw: u64,
    size_raw: u64,
    has_size_raw: u64,
    unit_raw: u64,
    default_size_raw: u64,
    signed_raw: u64,
    endianness_raw: u64,
) -> u64 {
    let builder = builder_ref(builder_raw);
    let unit = unit_raw as u64;
    let default_size_bits = default_size_raw as i64;
    let base_size = if has_size_raw != 0 {
        value_to_i63(Value::from_raw(size_raw), "bit array segment size")
    } else {
        default_size_bits
    };
    let size_bits = compute_size_bits(base_size, unit);
    if size_bits == 0 {
        return builder_raw;
    }
    let value = value_to_i63(Value::from_raw(value_raw), "bit array int segment");
    let signed = signed_raw != 0;
    let endianness = match endianness_raw {
        0 => BuilderEndianness::Big,
        1 => BuilderEndianness::Little,
        _ => panic!("invalid endianness flag"),
    };
    builder.append_int(value, size_bits, signed, endianness);
    builder_raw
}

#[no_mangle]
pub extern "C" fn bit_array_builder_append_string_utf8(builder_raw: u64, string_raw: u64) -> u64 {
    let builder = builder_ref(builder_raw);
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    builder.append_bytes(string.as_bytes());
    builder_raw
}

#[no_mangle]
pub extern "C" fn bit_array_builder_append_utf8_codepoint(
    builder_raw: u64,
    codepoint_raw: u64,
) -> u64 {
    let builder = builder_ref(builder_raw);
    let codepoint = value_to_i63(
        Value::from_raw(codepoint_raw),
        "bit array codepoint segment value",
    );
    let Some(character) = char::from_u32(codepoint as u32) else {
        panic!("runtime bit array builder received invalid codepoint");
    };

    let mut buffer = [0u8; 4];
    let encoded = character.encode_utf8(&mut buffer);
    builder.append_bytes(encoded.as_bytes());
    builder_raw
}

#[no_mangle]
pub extern "C" fn bit_array_builder_finish(builder_raw: u64) -> u64 {
    let builder_ptr = builder_from_raw(builder_raw);
    let builder = unsafe { Box::from_raw(builder_ptr) };
    let value = bit_array_from_bytes(&builder.bytes, builder.bit_len);
    value.to_raw()
}

#[no_mangle]
pub extern "C" fn string_eq(left_raw: u64, right_raw: u64) -> u64 {
    let left = value_to_string(Value::from_raw(left_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let right = value_to_string(Value::from_raw(right_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    Value::from_bool(left == right).to_raw()
}

#[no_mangle]
pub extern "C" fn string_pop_codeunit(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let mut bytes = string.into_bytes();
    if bytes.is_empty() {
        let tuple = tuple_from(
            &[Value::from_i63(0), string_to_value("")],
            "string pop codeunit tuple",
        );
        return tuple.to_raw();
    }
    let first = bytes[0];
    let rest_bytes = bytes.split_off(1);
    let rest = String::from_utf8(rest_bytes)
        .unwrap_or_else(|_| panic!("runtime string_pop_codeunit produced invalid UTF-8"));
    let tuple = tuple_from(
        &[Value::from_i63(first as i64), string_to_value(&rest)],
        "string pop codeunit tuple",
    );
    tuple.to_raw()
}

#[no_mangle]
pub extern "C" fn string_codeunit_slice(string_raw: u64, from_raw: u64, length_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(string_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let len = string.len() as i64;
    let from = value_to_i63(Value::from_raw(from_raw), "string codeunit slice from");
    let length = value_to_i63(Value::from_raw(length_raw), "string codeunit slice length");

    if length <= 0 {
        return string_to_value("").to_raw();
    }

    let mut start = if from < 0 { len + from } else { from };
    if start < 0 {
        start = 0;
    }
    if start > len {
        start = len;
    }

    let mut end = start.saturating_add(length);
    if end > len {
        end = len;
    }

    let start_usize = start as usize;
    let end_usize = end as usize;
    let slice = string.as_bytes()[start_usize..end_usize].to_vec();
    let result = String::from_utf8(slice)
        .unwrap_or_else(|_| panic!("runtime string_codeunit_slice produced invalid UTF-8"));
    string_to_value(&result).to_raw()
}

#[no_mangle]
pub extern "C" fn percent_encode(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let encoded = percent_encode_string(&string);
    string_to_value(&encoded).to_raw()
}

#[no_mangle]
pub extern "C" fn percent_decode(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    match percent_decode_string(&string, false) {
        Ok(decoded) => result_ok(string_to_value(&decoded)),
        Err(()) => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn parse_query(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let mut pairs = Vec::new();
    for section in string.split('&') {
        if section.is_empty() {
            continue;
        }
        let mut parts = section.splitn(2, '=');
        let key = parts.next().unwrap_or("");
        if key.is_empty() {
            continue;
        }
        let value = parts.next().unwrap_or("");
        let decoded_key = match percent_decode_string(key, true) {
            Ok(key) => key,
            Err(()) => return result_error(Value::nil()),
        };
        let decoded_value = match percent_decode_string(value, true) {
            Ok(value) => value,
            Err(()) => return result_error(Value::nil()),
        };
        let tuple = tuple_from(
            &[
                string_to_value(&decoded_key),
                string_to_value(&decoded_value),
            ],
            "parse_query pair",
        );
        pairs.push(tuple);
    }
    let list = list_from_vec(pairs);
    result_ok(list)
}

#[no_mangle]
pub extern "C" fn from_dynamic(raw: u64) -> u64 {
    let value = Value::from_raw(raw);
    if !matches!(header_tag(value), Some(Tag::Map)) {
        return result_error(Value::nil());
    }
    let entries = map_entries_vec(value);
    let gleam_error = match map_get_text(&entries, "gleam_error") {
        Ok(kind) => kind,
        Err(_) => return result_error(Value::nil()),
    };

    let kind_value = match gleam_error.as_str() {
        "todo" => atom("todo"),
        "panic" => atom("panic"),
        "let_assert" => {
            let start = match map_get_int(&entries, "start") {
                Ok(value) => value,
                Err(_) => return result_error(Value::nil()),
            };
            let end = match map_get_int(&entries, "end") {
                Ok(value) => value,
                Err(_) => return result_error(Value::nil()),
            };
            let pattern_start = match map_get_int(&entries, "pattern_start") {
                Ok(value) => value,
                Err(_) => return result_error(Value::nil()),
            };
            let pattern_end = match map_get_int(&entries, "pattern_end") {
                Ok(value) => value,
                Err(_) => return result_error(Value::nil()),
            };
            let value_field = match map_get_field(&entries, "value") {
                Some(field) => field,
                None => return result_error(Value::nil()),
            };
            tuple_from(
                &[
                    atom("let_assert"),
                    Value::from_i63(start),
                    Value::from_i63(end),
                    Value::from_i63(pattern_start),
                    Value::from_i63(pattern_end),
                    value_field,
                ],
                "panic let_assert",
            )
        }
        "assert" => {
            let start = match map_get_int(&entries, "start") {
                Ok(value) => value,
                Err(_) => return result_error(Value::nil()),
            };
            let end = match map_get_int(&entries, "end") {
                Ok(value) => value,
                Err(_) => return result_error(Value::nil()),
            };
            let expression_start = match map_get_int(&entries, "expression_start") {
                Ok(value) => value,
                Err(_) => return result_error(Value::nil()),
            };
            let assert_kind = match build_assert_kind(&entries) {
                Ok(kind) => kind,
                Err(_) => return result_error(Value::nil()),
            };
            tuple_from(
                &[
                    atom("assert"),
                    Value::from_i63(start),
                    Value::from_i63(end),
                    Value::from_i63(expression_start),
                    assert_kind,
                ],
                "panic assert",
            )
        }
        _ => return result_error(Value::nil()),
    };

    let panic_value = match wrap_gleam_panic(&entries, kind_value) {
        Ok(value) => value,
        Err(_) => return result_error(Value::nil()),
    };
    result_ok(panic_value)
}

#[no_mangle]
pub extern "C" fn read_file(path_raw: u64) -> u64 {
    let path = value_to_string(Value::from_raw(path_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    match fs::read(&path) {
        Ok(bytes) => {
            let bit_array = bit_array_from_bytes(&bytes, bytes.len() * 8);
            result_ok(bit_array)
        }
        Err(_) => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn read_file_text(path_raw: u64) -> u64 {
    let path = value_to_string(Value::from_raw(path_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    match fs::read_to_string(&path) {
        Ok(contents) => result_ok(string_to_value(&contents)),
        Err(_) => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn gleeunit_main() -> u64 {
    Value::nil().to_raw()
}

#[no_mangle]
pub extern "C" fn gleeunit_do_main() -> u64 {
    Value::nil().to_raw()
}

#[no_mangle]
pub extern "C" fn dict_new() -> u64 {
    map_from_vec(Vec::new()).to_raw()
}

#[no_mangle]
pub extern "C" fn dict_size(map_raw: u64) -> u64 {
    let table_ptr = map_table_from_value(Value::from_raw(map_raw));
    let len = unsafe { table_ptr.as_ref().len };
    Value::from_i63(len as i64).to_raw()
}

#[no_mangle]
pub extern "C" fn dict_to_list(map_raw: u64) -> u64 {
    let entries = map_entries_vec(Value::from_raw(map_raw));
    let mut values = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        let tuple = tuple_from(&[key, value], "dict to_list tuple");
        values.push(tuple);
    }
    list_from_vec(values).to_raw()
}

#[no_mangle]
pub extern "C" fn dict_get(map_raw: u64, key_raw: u64) -> u64 {
    let entries = map_entries_vec(Value::from_raw(map_raw));
    let key = Value::from_raw(key_raw);
    for (entry_key, entry_value) in entries.into_iter().rev() {
        if entry_key == key {
            return result_ok(entry_value);
        }
    }
    result_error(Value::nil())
}

#[no_mangle]
pub extern "C" fn dict_insert(map_raw: u64, key_raw: u64, value_raw: u64) -> u64 {
    let mut entries = map_entries_vec(Value::from_raw(map_raw));
    let key = Value::from_raw(key_raw);
    let value = Value::from_raw(value_raw);
    entries.retain(|(entry_key, _)| *entry_key != key);
    entries.push((key, value));
    map_from_vec(entries).to_raw()
}

#[no_mangle]
pub extern "C" fn dict_remove(map_raw: u64, key_raw: u64) -> u64 {
    let mut entries = map_entries_vec(Value::from_raw(map_raw));
    let key = Value::from_raw(key_raw);
    entries.retain(|(entry_key, _)| *entry_key != key);
    map_from_vec(entries).to_raw()
}

fn map_from_key_value_list(list_raw: u64, context: &str) -> Value {
    let pairs = list_to_vec(Value::from_raw(list_raw));
    let mut entries = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let elements = tuple_to_vec(pair);
        if elements.len() != 2 {
            panic!("runtime {context} expected key/value tuple");
        }
        let key = elements[0];
        value_to_string(key).unwrap_or_else(|_| panic!("runtime {context} expected String key"));
        let value = elements[1];
        entries.push((key, value));
    }
    map_from_vec(entries)
}

#[no_mangle]
pub extern "C" fn make_object(items_raw: u64) -> u64 {
    map_from_key_value_list(items_raw, "make_object").to_raw()
}

#[no_mangle]
pub extern "C" fn make_map(items_raw: u64) -> u64 {
    map_from_key_value_list(items_raw, "make_map").to_raw()
}

#[no_mangle]
pub extern "C" fn unsupported_zero_arity() -> u64 {
    panic!("erlang-only helper invoked on Cranelift runtime")
}

#[no_mangle]
pub extern "C" fn unsupported_one_arity(_arg0: u64) -> u64 {
    panic!("erlang-only helper invoked on Cranelift runtime")
}

#[no_mangle]
pub extern "C" fn unsupported_two_arity(_arg0: u64, _arg1: u64) -> u64 {
    panic!("erlang-only helper invoked on Cranelift runtime")
}

#[no_mangle]
pub extern "C" fn unsupported_three_arity(_arg0: u64, _arg1: u64, _arg2: u64) -> u64 {
    panic!("erlang-only helper invoked on Cranelift runtime")
}

#[no_mangle]
pub extern "C" fn graphemes(raw: u64) -> u64 {
    let string =
        value_to_string(Value::from_raw(raw)).unwrap_or_else(|_| panic!("expected String value"));
    let values = UnicodeSegmentation::graphemes(string.as_str(), true)
        .map(|grapheme| string_to_value(grapheme))
        .collect::<Vec<_>>();
    list_from_vec(values).to_raw()
}

#[no_mangle]
pub extern "C" fn split_string_tree(tree_raw: u64, pattern_raw: u64, _direction_raw: u64) -> u64 {
    let string = value_to_string(Value::from_raw(tree_raw))
        .unwrap_or_else(|_| panic!("expected String value"));
    let pattern = value_to_string(Value::from_raw(pattern_raw))
        .unwrap_or_else(|_| panic!("expected String value"));

    let values = if pattern.is_empty() {
        UnicodeSegmentation::graphemes(string.as_str(), true)
            .map(|grapheme| string_to_value(grapheme))
            .collect::<Vec<_>>()
    } else {
        string
            .split(pattern.as_str())
            .map(|segment| string_to_value(segment))
            .collect::<Vec<_>>()
    };

    list_from_vec(values).to_raw()
}

#[no_mangle]
pub extern "C" fn parse_int(raw: u64) -> u64 {
    let input = Value::from_raw(raw);
    let parsed = value_to_string(input)
        .unwrap_or_else(|_| panic!("expected String value"))
        .trim()
        .parse::<i64>()
        .ok();

    match parsed {
        Some(value) => result_ok(Value::from_i63(value)),
        None => result_error(Value::nil()),
    }
}

#[no_mangle]
pub extern "C" fn int_from_base_string(string_raw: u64, base_raw: u64) -> u64 {
    let string = Value::from_raw(string_raw);
    let base = value_to_i63(Value::from_raw(base_raw), "int_from_base_string");
    let parsed = value_to_string(string).unwrap_or_else(|_| panic!("expected String value"));
    match i64::from_str_radix(parsed.trim(), base as u32).ok() {
        Some(value) => result_ok(Value::from_i63(value)),
        None => result_error(Value::nil()),
    }
}

fn int_to_base_string_impl(number: i64, base: i64) -> Option<String> {
    if base < 2 || base > 36 {
        return None;
    }

    if number == 0 {
        return Some("0".into());
    }

    let negative = number < 0;
    let mut n = if negative {
        -(number as i128)
    } else {
        number as i128
    };
    let base = base as i128;
    let mut digits = Vec::new();
    while n > 0 {
        let digit = (n % base) as usize;
        digits.push(b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[digit] as char);
        n /= base;
    }
    if negative {
        digits.push('-');
    }
    digits.reverse();
    Some(digits.into_iter().collect())
}

#[no_mangle]
pub extern "C" fn to_string(int_raw: u64) -> u64 {
    let value = value_to_i63(Value::from_raw(int_raw), "int_to_string");
    bytes_to_value(value.to_string().as_bytes()).to_raw()
}

#[no_mangle]
pub extern "C" fn int_to_base_string(int_raw: u64, base_raw: u64) -> u64 {
    let value = value_to_i63(Value::from_raw(int_raw), "int_to_base_string");
    let base = value_to_i63(Value::from_raw(base_raw), "int_to_base_string base");
    let string = int_to_base_string_impl(value, base)
        .unwrap_or_else(|| panic!("runtime int_to_base_string received invalid base {base}"));
    bytes_to_value(string.as_bytes()).to_raw()
}

fn bitwise_binary_op(
    a_raw: u64,
    b_raw: u64,
    op: impl Fn(i64, i64) -> i64,
    context: &'static str,
) -> u64 {
    let a = value_to_i63(Value::from_raw(a_raw), context);
    let b = value_to_i63(Value::from_raw(b_raw), context);
    Value::from_i63(op(a, b)).to_raw()
}

fn bitwise_unary_op(a_raw: u64, op: impl Fn(i64) -> i64, context: &'static str) -> u64 {
    let a = value_to_i63(Value::from_raw(a_raw), context);
    Value::from_i63(op(a)).to_raw()
}

#[no_mangle]
pub extern "C" fn bitwise_and(a_raw: u64, b_raw: u64) -> u64 {
    bitwise_binary_op(a_raw, b_raw, |a, b| a & b, "bitwise_and")
}

#[no_mangle]
pub extern "C" fn bitwise_or(a_raw: u64, b_raw: u64) -> u64 {
    bitwise_binary_op(a_raw, b_raw, |a, b| a | b, "bitwise_or")
}

#[no_mangle]
pub extern "C" fn bitwise_exclusive_or(a_raw: u64, b_raw: u64) -> u64 {
    bitwise_binary_op(a_raw, b_raw, |a, b| a ^ b, "bitwise_xor")
}

#[no_mangle]
pub extern "C" fn bitwise_not(a_raw: u64) -> u64 {
    bitwise_unary_op(a_raw, |a| !a, "bitwise_not")
}

fn shift_left(a: i64, b: i64) -> Option<i64> {
    if b < 0 {
        shift_right(a, -b)
    } else {
        let shifted = (a as i128) << b;
        if shifted < MIN_I63 as i128 || shifted > MAX_I63 as i128 {
            None
        } else {
            Some(shifted as i64)
        }
    }
}

fn shift_right(a: i64, b: i64) -> Option<i64> {
    if b < 0 {
        shift_left(a, -b)
    } else {
        Some(a >> b)
    }
}

#[no_mangle]
pub extern "C" fn bitwise_shift_left(a_raw: u64, b_raw: u64) -> u64 {
    let a = value_to_i63(Value::from_raw(a_raw), "bitwise_shift_left");
    let b = value_to_i63(Value::from_raw(b_raw), "bitwise_shift_left");
    let result = shift_left(a, b).unwrap_or_else(|| panic!("runtime bitwise_shift_left overflow"));
    Value::from_i63(result).to_raw()
}

#[no_mangle]
pub extern "C" fn bitwise_shift_right(a_raw: u64, b_raw: u64) -> u64 {
    let a = value_to_i63(Value::from_raw(a_raw), "bitwise_shift_right");
    let b = value_to_i63(Value::from_raw(b_raw), "bitwise_shift_right");
    let result =
        shift_right(a, b).unwrap_or_else(|| panic!("runtime bitwise_shift_right overflow"));
    Value::from_i63(result).to_raw()
}

#[no_mangle]
pub extern "C" fn identity(raw: u64) -> u64 {
    raw
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
    fn runtime_print_returns_nil() {
        let result = runtime_print(Value::nil().to_raw(), false, OutputStream::Stdout);
        assert_eq!(result, Value::nil().to_raw());
    }

    #[test]
    fn float_parse_success_tuple() {
        let value = bytes_to_value(b"12.5");
        let tuple = Value::from_raw(parse_float(value.to_raw()));
        let ptr = tuple
            .as_boxed::<crate::layout::Tuple>()
            .expect("tuple pointer");
        let header = unsafe { &*ptr.as_ptr() };
        assert_eq!(header.header.arity(), 2);
    }

    #[test]
    fn float_round_returns_int_value() {
        let value = float_to_value(1.8);
        let result = Value::from_raw(round(value.to_raw()));
        assert_eq!(result.to_i63(), Some(2));
    }
}
