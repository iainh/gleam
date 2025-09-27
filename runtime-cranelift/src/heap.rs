//! Safe allocation facade that constructs runtime values using the GC.

use std::alloc::{Layout, LayoutError};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;

use crate::gc;
use crate::header::{Header, Tag};
use crate::layout::{
    Binary, BinaryData, BinarySlice, BitArray, Closure, ClosureFn, ConsCell, FloatBox, Map,
    MapTable, Record, ResourceHandle,
};
use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationError {
    OutOfMemory,
    InvalidLayout,
}

impl From<LayoutError> for AllocationError {
    fn from(_: LayoutError) -> Self {
        AllocationError::InvalidLayout
    }
}

/// Simple heap facade that will later be backed by a real garbage collector.
#[derive(Debug, Default)]
pub struct Heap;

impl Heap {
    pub const fn new() -> Self {
        Self
    }

    /// Allocate a boxed value using the provided header metadata and return a pointer to it.
    pub fn allocate_box(&self, header: Header) -> Result<NonNull<Header>, AllocationError> {
        gc::ensure_initialised();
        let layout = header.allocation_layout().map_err(AllocationError::from)?;
        // SAFETY: `malloc` returns suitably aligned memory or null on failure.
        let ptr = unsafe { gc::malloc(layout.size()) } as *mut u8;
        let ptr = NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)?;
        // SAFETY: the pointer is valid for the header size; write initial header contents.
        unsafe {
            ptr.cast::<Header>().as_ptr().write(header);
        }
        Ok(ptr.cast())
    }

    /// Allocate `words` 64-bit slots without writing a header. Intended for DST payloads.
    pub fn allocate_words(&self, words: usize) -> Result<NonNull<u8>, AllocationError> {
        gc::ensure_initialised();
        let bytes = words
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(AllocationError::InvalidLayout)?;
        let _ = Layout::from_size_align(bytes, core::mem::align_of::<u64>())
            .map_err(AllocationError::from)?;
        let ptr = unsafe { gc::malloc(bytes) } as *mut u8;
        NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)
    }

    pub fn alloc_binary_data(
        &self,
        capacity: usize,
    ) -> Result<NonNull<BinaryData>, AllocationError> {
        gc::ensure_initialised();
        let layout = BinaryData::layout_for(capacity).map_err(AllocationError::from)?;
        // Binary payloads never store GC pointers, so skip scanning by using atomic allocation.
        let ptr = unsafe { gc::malloc_atomic(layout.size()) } as *mut u8;
        let ptr = NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)?;
        unsafe {
            let data_ptr = ptr.cast::<BinaryData>().as_ptr();
            core::ptr::addr_of_mut!((*data_ptr).ref_count).write(AtomicUsize::new(1));
            core::ptr::addr_of_mut!((*data_ptr).capacity).write(capacity);
            let bytes_ptr = data_ptr
                .cast::<u8>()
                .add(core::mem::size_of::<BinaryData>());
            core::ptr::write_bytes(bytes_ptr, 0, capacity);
        }
        Ok(ptr.cast())
    }

    /// Allocate a floating point box.
    pub fn alloc_float(&self, value: f64) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(FloatBox::HEADER)?;
        unsafe {
            ptr.cast::<FloatBox>().as_ptr().write(FloatBox::new(value));
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    /// Allocate a cons cell.
    pub fn alloc_cons(&self, head: Value, tail: Value) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(ConsCell::HEADER)?;
        unsafe {
            let cell = ptr.cast::<ConsCell>().as_ptr();
            (*cell).head = head;
            (*cell).tail = tail;
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    /// Allocate a tuple from the provided elements.
    pub fn alloc_tuple(&self, elements: &[Value]) -> Result<Value, AllocationError> {
        let arity = elements.len();
        let header = Header::new(Tag::Tuple, arity as u16, arity as u32);
        let ptr = self.allocate_box(header)?;
        unsafe {
            let payload = ptr
                .as_ptr()
                .cast::<u8>()
                .add(core::mem::size_of::<Header>()) as *mut Value;
            core::ptr::copy_nonoverlapping(elements.as_ptr(), payload, arity);
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    pub fn alloc_record(
        &self,
        constructor_index: u32,
        fields: &[Value],
    ) -> Result<Value, AllocationError> {
        let field_count = fields.len();
        let arity = u16::try_from(field_count).map_err(|_| AllocationError::InvalidLayout)?;
        let header = Record::header_for_field_count(arity);
        let ptr = self.allocate_box(header)?;
        unsafe {
            let record = ptr.cast::<Record>().as_ptr();
            core::ptr::addr_of_mut!((*record).constructor_index).write(constructor_index);
            core::ptr::addr_of_mut!((*record).flags).write(0);
            let fields_ptr = (record as *mut u8).add(core::mem::size_of::<Record>()) as *mut Value;
            core::ptr::copy_nonoverlapping(fields.as_ptr(), fields_ptr, field_count);
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    pub fn alloc_binary(
        &self,
        data: NonNull<BinaryData>,
        len: usize,
        capacity: usize,
    ) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(Binary::HEADER)?;
        unsafe {
            let binary = ptr.cast::<Binary>().as_ptr();
            (*binary).data = data.as_ptr();
            (*binary).len = len;
            (*binary).capacity = capacity;
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    pub fn alloc_binary_slice(
        &self,
        data: NonNull<BinaryData>,
        len: usize,
        capacity: usize,
        offset: usize,
    ) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(BinarySlice::HEADER)?;
        unsafe {
            let binary = ptr.cast::<BinarySlice>().as_ptr();
            (*binary).data = data.as_ptr();
            (*binary).len = len;
            (*binary).capacity = capacity;
            (*binary).offset = offset;
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    pub fn alloc_bit_array(
        &self,
        data: NonNull<BinaryData>,
        bit_offset: usize,
        bit_len: usize,
        capacity_bits: usize,
    ) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(BitArray::HEADER)?;
        unsafe {
            let bit_array = ptr.cast::<BitArray>().as_ptr();
            (*bit_array).data = data.as_ptr();
            (*bit_array).bit_offset = bit_offset;
            (*bit_array).bit_len = bit_len;
            (*bit_array).capacity_bits = capacity_bits;
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    pub fn alloc_map_table(&self, len: usize) -> Result<NonNull<MapTable>, AllocationError> {
        gc::ensure_initialised();
        let layout = MapTable::layout_for(len).map_err(AllocationError::from)?;
        let ptr = unsafe { gc::malloc(layout.size()) } as *mut u8;
        let ptr = NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)?;
        unsafe {
            let table = ptr.cast::<MapTable>().as_ptr();
            core::ptr::addr_of_mut!((*table).len).write(len);
            let entries = (*table).entries_slice_mut();
            for entry in entries {
                core::ptr::addr_of_mut!(entry.key).write(Value::nil());
                core::ptr::addr_of_mut!(entry.value).write(Value::nil());
            }
        }
        Ok(ptr.cast())
    }

    pub fn alloc_map(&self, table: NonNull<MapTable>) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(Map::HEADER)?;
        unsafe {
            let map = ptr.cast::<Map>().as_ptr();
            (*map).table = table.as_ptr();
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    pub fn alloc_resource(&self, pointer: *mut c_void) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(ResourceHandle::HEADER)?;
        unsafe {
            ptr.cast::<ResourceHandle>()
                .as_ptr()
                .write(ResourceHandle::new(pointer));
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }

    /// # Safety
    /// `env_ptr` must point to `env_len` consecutive, initialised `Value`s.
    pub unsafe fn alloc_closure(
        &self,
        code_ptr: ClosureFn,
        env_ptr: *const Value,
        env_len: usize,
    ) -> Result<Value, AllocationError> {
        let header = Closure::header_for_env(env_len as u16);
        let ptr = self.allocate_box(header)?;
        unsafe {
            let closure = ptr.cast::<Closure>().as_ptr();
            (*closure).code_ptr = code_ptr;
            (*closure).env_size = env_len;
            if env_len > 0 {
                let dst = (*closure).env.as_ptr() as *mut Value;
                std::ptr::copy_nonoverlapping(env_ptr, dst, env_len);
            }
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_float_boxes_value() {
        let heap = Heap::new();
        let value = heap.alloc_float(1.5).unwrap();
        assert!(value.is_boxed());
        let float_box = unsafe { value.as_boxed::<FloatBox>().unwrap().as_ref() };
        assert_eq!(float_box.value, 1.5);
    }

    #[test]
    fn alloc_binary_data_zeroes_buffer() {
        let heap = Heap::new();
        let data = heap.alloc_binary_data(4).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts((*data.as_ptr()).as_ptr(), 4) };
        assert_eq!(bytes, &[0, 0, 0, 0]);
    }

    #[test]
    fn alloc_resource_stores_pointer() {
        let heap = Heap::new();
        let pointer = 0x1234 as *mut core::ffi::c_void;
        let value = heap.alloc_resource(pointer).unwrap();
        let handle = unsafe { value.as_boxed::<ResourceHandle>().unwrap().as_ref() };
        assert_eq!(handle.pointer, pointer);
    }
}
