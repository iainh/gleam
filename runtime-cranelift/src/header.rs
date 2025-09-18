use std::alloc::{Layout, LayoutError};

/// Primary tag describing a boxed runtime value.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Float = 1,
    Binary = 2,
    BinarySlice = 3,
    List = 4,
    Tuple = 5,
    Record = 6,
    Closure = 7,
    Map = 8,
    BitArray = 9,
    Mailbox = 10,
    Resource = 11,
}

impl Tag {
    pub const fn to_u16(self) -> u16 {
        self as u16
    }
}

/// Common 64-bit header stored at the start of every boxed value.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    tag: Tag,
    arity: u16,
    payload_words: u32,
}

impl Header {
    pub const fn new(tag: Tag, arity: u16, payload_words: u32) -> Self {
        Self {
            tag,
            arity,
            payload_words,
        }
    }

    pub const fn tag(&self) -> Tag {
        self.tag
    }

    pub const fn arity(&self) -> u16 {
        self.arity
    }

    pub const fn payload_words(&self) -> u32 {
        self.payload_words
    }

    pub const fn total_words(&self) -> usize {
        1 + self.payload_words as usize
    }

    pub const fn total_bytes(&self) -> usize {
        self.total_words() * core::mem::size_of::<u64>()
    }

    pub fn allocation_layout(&self) -> Result<Layout, LayoutError> {
        Layout::from_size_align(self.total_bytes(), core::mem::align_of::<Header>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_eight_bytes() {
        assert_eq!(core::mem::size_of::<Header>(), 8);
    }
}
