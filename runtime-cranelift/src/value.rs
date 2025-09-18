use std::ptr::NonNull;

/// Pointer-sized Gleam value using the tagging scheme described in the design doc.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Value(pub u64);

impl Value {
    pub const TAG_MASK: u64 = 0b11;
    pub const SMALL_INT_TAG: u64 = 0b01;
    pub const OTHER_IMMEDIATE_TAG: u64 = 0b11;
    pub const ATOM_TAG: u64 = 0b101;
    pub const ATOM_SHIFT: u32 = 3;
    pub const ATOM_MASK: u64 = (1 << Self::ATOM_SHIFT) - 1;

    pub const FALSE: Value = Value(0b001);
    pub const TRUE: Value = Value(0b011);
    pub const NIL: Value = Value(0b111);

    pub const fn from_i63(value: i64) -> Self {
        Self(((value as u64) << 1) | Self::SMALL_INT_TAG)
    }

    pub fn to_i63(self) -> Option<i64> {
        if self.is_i63() {
            let raw = self.0 as i64;
            Some(raw >> 1)
        } else {
            None
        }
    }

    pub const fn is_atom(self) -> bool {
        (self.0 & Self::TAG_MASK) == Self::OTHER_IMMEDIATE_TAG
            && (self.0 & Self::ATOM_MASK) == Self::ATOM_TAG
    }

    pub const fn atom(index: u32) -> Self {
        Self(((index as u64) << Self::ATOM_SHIFT) | Self::ATOM_TAG)
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
        assert!(!Value::NIL.is_atom());
        let atom = Value::atom(1);
        assert!(atom.is_atom());
        assert_eq!(atom.atom_index(), Some(1));
    }
}
