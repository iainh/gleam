//! Thin wrapper around the Boehm GC initialisation and allocation APIs.

use std::ffi::c_void;
use std::sync::Once;

static GC_INIT: Once = Once::new();

/// Ensure the Boehm GC is initialised and ready for allocations.
pub fn ensure_initialised() {
    GC_INIT.call_once(|| unsafe {
        bdwgc_sys::GC_init();
        bdwgc_sys::GC_enable();
    });
}

/// Allocate `size` bytes from the GC heap, returning a null pointer on failure.
///
/// # Safety
/// The caller must ensure the returned pointer is used according to the GC's
/// expectations and eventually released or traced as appropriate.
pub unsafe fn malloc(size: usize) -> *mut c_void {
    ensure_initialised();
    unsafe { bdwgc_sys::GC_malloc(size) }
}

/// Allocate `size` bytes for data that does not contain pointers to GC-managed memory.
///
/// # Safety
/// The caller must ensure the region truly holds no GC-managed pointers and
/// that the returned pointer is handled safely.
pub unsafe fn malloc_atomic(size: usize) -> *mut c_void {
    ensure_initialised();
    unsafe { bdwgc_sys::GC_malloc_atomic(size) }
}

/// Trigger a full garbage collection cycle.
pub fn collect() {
    ensure_initialised();
    unsafe {
        bdwgc_sys::GC_gcollect();
    }
}

/// Disable the GC. Mainly useful for tests where deterministic allocation behaviour is needed.
pub fn disable() {
    ensure_initialised();
    unsafe {
        bdwgc_sys::GC_disable();
    }
}

/// Re-enable the GC after a call to [`disable`].
pub fn enable() {
    ensure_initialised();
    unsafe {
        bdwgc_sys::GC_enable();
    }
}
