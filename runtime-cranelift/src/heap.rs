use std::alloc::{alloc, dealloc, Layout, LayoutError};
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;

use crate::header::{Header, Tag};
use crate::layout::{
    Binary, BinaryData, BinarySlice, BitArray, ConsCell, FloatBox, Map, MapTable, Tuple,
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
        let layout = header.allocation_layout().map_err(AllocationError::from)?;
        // SAFETY: `alloc` returns suitably aligned memory for the layout or null on failure.
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)?;
        // SAFETY: the pointer is valid for the header size; write initial header contents.
        unsafe {
            ptr.cast::<Header>().as_ptr().write(header);
        }
        Ok(ptr.cast())
    }

    /// Deallocate a previously allocated boxed value.
    ///
    /// # Safety
    ///
    /// `ptr` must be a pointer returned by [`Heap::allocate_box`] and must not be used afterwards.
    pub unsafe fn deallocate_box(&self, ptr: NonNull<Header>) {
        let layout = {
            let header = unsafe { ptr.as_ref() };
            header
                .allocation_layout()
                .expect("header layout should be valid for deallocation")
        };
        unsafe {
            dealloc(ptr.cast().as_ptr(), layout);
        }
    }

    /// Allocate `words` 64-bit slots without writing a header. Intended for DST payloads.
    pub fn allocate_words(&self, words: usize) -> Result<NonNull<u8>, AllocationError> {
        let bytes = words
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(AllocationError::InvalidLayout)?;
        let layout = Layout::from_size_align(bytes, core::mem::align_of::<u64>())
            .map_err(AllocationError::from)?;
        let ptr = unsafe { alloc(layout) };
        NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)
    }

    pub fn alloc_binary_data(
        &self,
        capacity: usize,
    ) -> Result<NonNull<BinaryData>, AllocationError> {
        let layout = BinaryData::layout_for(capacity).map_err(AllocationError::from)?;
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocationError::OutOfMemory)?;
        unsafe {
            let data_ptr = ptr.cast::<BinaryData>().as_ptr();
            core::ptr::addr_of_mut!((*data_ptr).ref_count).write(AtomicUsize::new(1));
            core::ptr::addr_of_mut!((*data_ptr).capacity).write(capacity);
        }
        Ok(ptr.cast())
    }

    pub unsafe fn dealloc_binary_data(&self, ptr: NonNull<BinaryData>) {
        let layout = {
            let capacity = unsafe { ptr.as_ref() }.capacity;
            BinaryData::layout_for(capacity)
                .expect("binary data layout should be valid for stored capacity")
        };
        unsafe {
            dealloc(ptr.cast().as_ptr(), layout);
        }
    }

    /// Allocate a floating point box.
    pub fn alloc_float(&self, value: f64) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(FloatBox::HEADER)?;
        unsafe {
            let float_box = ptr.cast::<FloatBox>().as_ptr();
            (*float_box).value = value;
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
            let payload =
                ptr.as_ptr().cast::<u8>().add(core::mem::size_of::<Tuple>()) as *mut Value;
            core::ptr::copy_nonoverlapping(elements.as_ptr(), payload, arity);
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

    pub fn alloc_map(&self, table: NonNull<MapTable>) -> Result<Value, AllocationError> {
        let ptr = self.allocate_box(Map::HEADER)?;
        unsafe {
            let map = ptr.cast::<Map>().as_ptr();
            (*map).table = table.as_ptr();
        }
        Ok(Value::from_raw(ptr.as_ptr() as u64))
    }
}
