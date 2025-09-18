use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::layout::BinaryData;

/// Increase the reference count on the given binary buffer.
pub fn retain(data: NonNull<BinaryData>) {
    let atomic: &AtomicUsize = unsafe { &data.as_ref().ref_count };
    let _ = atomic.fetch_add(1, Ordering::Relaxed);
}

/// Decrease the reference count on the given binary buffer.
/// Returns `true` if this call observed the last reference.
pub fn release(data: NonNull<BinaryData>) -> bool {
    let atomic: &AtomicUsize = unsafe { &data.as_ref().ref_count };
    let previous = atomic.fetch_sub(1, Ordering::AcqRel);
    debug_assert!(previous > 0, "binary refcount underflow");
    previous == 1
}

/// Return the number of strong references currently held for the buffer.
pub fn ref_count(data: NonNull<BinaryData>) -> usize {
    let atomic: &AtomicUsize = unsafe { &data.as_ref().ref_count };
    atomic.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::{ref_count, release, retain};
    use crate::heap::Heap;

    #[test]
    fn retain_and_release_updates_counts() {
        let heap = Heap::new();
        let data = heap.alloc_binary_data(8).unwrap();
        assert_eq!(ref_count(data), 1);
        retain(data);
        assert_eq!(ref_count(data), 2);
        assert!(!release(data));
        assert_eq!(ref_count(data), 1);
        assert!(release(data));
        assert_eq!(ref_count(data), 0);
    }
}
