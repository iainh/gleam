use std::alloc::{Layout, LayoutError};
use std::sync::atomic::AtomicUsize;

use crate::{
    header::{Header, Tag},
    value::Value,
};

#[repr(C)]
#[derive(Debug)]
pub struct FloatBox {
    pub header: Header,
    pub value: f64,
}

impl FloatBox {
    pub const HEADER: Header = Header::new(Tag::Float, 0, 1);

    pub const fn new(value: f64) -> Self {
        Self {
            header: Self::HEADER,
            value,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct Binary {
    pub header: Header,
    pub data: *mut BinaryData,
    pub len: usize,
    pub capacity: usize,
}

impl Binary {
    pub const HEADER: Header = Header::new(Tag::Binary, 0, 3);
}

#[repr(C)]
#[derive(Debug)]
pub struct BinarySlice {
    pub header: Header,
    pub data: *mut BinaryData,
    pub len: usize,
    pub capacity: usize,
    pub offset: usize,
}

impl BinarySlice {
    pub const HEADER: Header = Header::new(Tag::BinarySlice, 0, 4);
}

#[repr(C)]
#[derive(Debug)]
pub struct BinaryData {
    pub ref_count: AtomicUsize,
    pub capacity: usize,
    pub bytes: [u8; 0],
}

impl BinaryData {
    pub fn layout_for(capacity: usize) -> Result<Layout, LayoutError> {
        let (layout, _) = Layout::new::<BinaryData>().extend(Layout::array::<u8>(capacity)?)?;
        Ok(layout)
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr()
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct ConsCell {
    pub header: Header,
    pub head: Value,
    pub tail: Value,
}

impl ConsCell {
    pub const HEADER: Header = Header::new(Tag::List, 2, 2);

    pub const fn new(head: Value, tail: Value) -> Self {
        Self {
            header: Self::HEADER,
            head,
            tail,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct Tuple {
    pub header: Header,
    pub elements: [Value; 0],
}

impl Tuple {
    pub const fn header_for_arity(arity: u16) -> Header {
        Header::new(Tag::Tuple, arity, arity as u32)
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct Record {
    pub header: Header,
    pub constructor_index: u32,
    pub flags: u32,
    pub fields: [Value; 0],
}

impl Record {
    pub const fn header_for_field_count(field_count: u16) -> Header {
        Header::new(Tag::Record, field_count, field_count as u32 + 1)
    }
}

pub type ClosureFn = unsafe extern "C" fn(*const Closure, *const Value, usize) -> Value;

#[repr(C)]
#[derive(Debug)]
pub struct Closure {
    pub header: Header,
    pub code_ptr: ClosureFn,
    pub env_size: usize,
    pub env: [Value; 0],
}

impl Closure {
    pub const fn header_for_env(env_size: u16) -> Header {
        Header::new(Tag::Closure, env_size + 2, env_size as u32 + 2)
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct Map {
    pub header: Header,
    pub table: *mut MapTable,
}

impl Map {
    pub const HEADER: Header = Header::new(Tag::Map, 0, 1);
}

#[repr(C)]
#[derive(Debug)]
pub struct MapTable;

#[repr(C)]
#[derive(Debug)]
pub struct BitArray {
    pub header: Header,
    pub data: *mut BinaryData,
    pub bit_offset: usize,
    pub bit_len: usize,
    pub capacity_bits: usize,
}

impl BitArray {
    pub const HEADER: Header = Header::new(Tag::BitArray, 0, 4);
}

#[repr(C)]
#[derive(Debug)]
pub struct Mailbox {
    pub header: Header,
    pub queue_head: *mut Message,
    pub queue_tail: *mut Message,
}

impl Mailbox {
    pub const HEADER: Header = Header::new(Tag::Mailbox, 0, 2);
}

#[repr(C)]
#[derive(Debug)]
pub struct Message {
    pub next: *mut Message,
    pub payload: Value,
}

#[repr(C)]
#[derive(Debug)]
pub struct ResourceHandle {
    pub header: Header,
    pub pointer: *mut core::ffi::c_void,
}

impl ResourceHandle {
    pub const HEADER: Header = Header::new(Tag::Resource, 0, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_payload_words() {
        assert_eq!(ConsCell::HEADER.payload_words(), 2);
    }

    #[test]
    fn binary_payload_words() {
        assert_eq!(Binary::HEADER.payload_words(), 3);
        assert_eq!(BinarySlice::HEADER.payload_words(), 4);
    }

    #[test]
    fn tuple_header_matches_arity() {
        let header = Tuple::header_for_arity(3);
        assert_eq!(header.arity(), 3);
        assert_eq!(header.payload_words(), 3);
    }

    #[test]
    fn record_header_accounts_for_ctor_word() {
        let header = Record::header_for_field_count(2);
        assert_eq!(header.payload_words(), 3);
    }

    #[test]
    fn binary_data_layout_scales_with_capacity() {
        let small = BinaryData::layout_for(4).unwrap();
        let big = BinaryData::layout_for(128).unwrap();
        assert!(big.size() > small.size());
    }
}
