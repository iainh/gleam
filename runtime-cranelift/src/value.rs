use std::ptr::NonNull;

use crate::header::{Header, Tag};

#[repr(C)]
struct StaticValue {
    header: Header,
}

static FALSE_STATIC: StaticValue = StaticValue {
    header: Header::new(Tag::Boolean, 0, 0),
};

static TRUE_STATIC: StaticValue = StaticValue {
    header: Header::new(Tag::Boolean, 1, 0),
};

static NIL_STATIC: StaticValue = StaticValue {
    header: Header::new(Tag::Nil, 0, 0),
};

/// Pointer-sized Gleam value using the tagging scheme described in the design doc.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Value(pub u64);

impl Value {
    pub const TAG_MASK: u64 = 0b11;
    pub const SMALL_INT_TAG: u64 = 0b01;
    pub const OTHER_IMMEDIATE_TAG: u64 = 0b11;
    pub const ATOM_SHIFT: u32 = 2;

    pub const fn from_i63(value: i64) -> Self {
        let shifted = (value as i128) << 2;
        Self(((shifted as u128) as u64) | Self::SMALL_INT_TAG)
    }

    pub fn from_bool(value: bool) -> Self {
        if value {
            Self::from_static(&TRUE_STATIC)
        } else {
            Self::from_static(&FALSE_STATIC)
        }
    }

    pub fn nil() -> Self {
        Self::from_static(&NIL_STATIC)
    }

    pub fn to_i63(self) -> Option<i64> {
        if self.is_i63() {
            Some((self.0 as i64) >> 2)
        } else {
            None
        }
    }

    pub const fn atom(index: u32) -> Self {
        Self(((index as u64) << Self::ATOM_SHIFT) | Self::OTHER_IMMEDIATE_TAG)
    }

    pub fn atom_index(self) -> Option<u32> {
        if self.is_atom() {
            Some((self.0 >> Self::ATOM_SHIFT) as u32)
        } else {
            None
        }
    }

    pub const fn is_i63(self) -> bool {
        (self.0 & Self::TAG_MASK) == Self::SMALL_INT_TAG
    }

    pub const fn is_immediate(self) -> bool {
        (self.0 & Self::TAG_MASK) != 0
    }

    pub const fn is_atom(self) -> bool {
        (self.0 & Self::TAG_MASK) == Self::OTHER_IMMEDIATE_TAG
    }

    pub const fn is_boxed(self) -> bool {
        (self.0 & Self::TAG_MASK) == 0
    }

    pub fn as_boxed<T>(self) -> Option<NonNull<T>> {
        if self.is_boxed() {
            // SAFETY: Value guarantees the pointer is properly tagged; caller must ensure `T` matches.
            let ptr = core::ptr::NonNull::new(self.0 as *mut T)?;
            Some(ptr)
        } else {
            None
        }
    }

    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn to_raw(self) -> u64 {
        self.0
    }

    fn from_static(value: &'static StaticValue) -> Self {
        Self(value as *const StaticValue as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    #[test]
    fn roundtrip_small_int() {
        let values = [-5, -1, 0, 1, 42, i32::MAX as i64];
        for &value in &values {
            let tagged = Value::from_i63(value);
            assert_eq!(tagged.to_i63(), Some(value));
        }
    }

    #[test]
    fn atom_pattern_excludes_nil() {
        assert!(!Value::nil().is_atom());
        let atom = Value::atom(1);
        assert!(atom.is_atom());
        assert_eq!(atom.atom_index(), Some(1));
    }
}
