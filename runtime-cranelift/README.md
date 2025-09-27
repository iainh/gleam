# Gleam Native Runtime

The `runtime-cranelift` crate hosts the native runtime used by the Cranelift
backend. It provides the value representation, memory management helpers, and
`extern "C"` functions that Cranelift generated code and hand-written native
modules rely on.

If you are new to the native backend start with
[`cranelift_runtime_layout.md`](../cranelift_runtime_layout.md), which describes
the tagging scheme and heap layout that the code in this crate implements.

## Crate layout

- `lib.rs` re-exports the public surface area that other crates use. The runtime
  is intentionally small and mostly keeps items private so that changes can be
  made without forcing downstream consumers to rebuild against internal types.
- `value.rs` defines the tagged `Value` type that represents every Gleam value
  at runtime. It knows how to pack and unpack small integers, atoms, and boxed
  pointers.
- `header.rs` contains the common 64-bit header stored at the start of every
  boxed allocation together with the `Tag` enum used by the GC and pattern
  matching logic.
- `layout.rs` defines the concrete in-memory layouts for runtime data
  structures such as tuples, records, binaries, closures, and maps. These
  structs mirror the diagrams in the layout document and provide helpers (for
  example `BinaryData::layout_for`) used during allocation.
- `heap.rs` is the safe facade around the Boehm GC. It offers typed allocation
  routines (`alloc_tuple`, `alloc_record`, `alloc_bit_array`, …) that return a
  tagged `Value` so callers do not have to manually assemble headers.
- `gc.rs` wraps the low-level `bdwgc-sys` bindings and exposes a minimal API for
  initialisation, manual collection, and direct `malloc` style helpers.
- `binary.rs` implements reference counting for binary/bit array buffers. These
  buffers are GC-allocated but shareable between multiple `Binary` views.
- `atom.rs` maintains the global atom table used to intern module, function, and
  constructor names.
- `ffi.rs` is where the bulk of the runtime surface lives. It defines the
  `#[unsafe(no_mangle)] extern "C" fn` entry points invoked from generated code and the
  standard library when running on the native backend. Most of the logic in the
  runtime sits in this module.

## Runtime entry points (`ffi.rs`)

Functions in `ffi.rs` fall into a few groups:

- **Bootstrapping and GC**: `gleam_runtime_init`, `gleam_runtime_collect`, and
  helpers like `gleam_alloc_tuple` or `gleam_alloc_record` prepare the GC and
  allocate core data structures.
- **Value inspectors and constructors**: operations such as `classify_dynamic`,
  `list_to_array`, `is_null`, and the various `dynamic_*` functions let the
  standard library implement conversions and decoding without duplicating unsafe
  pointer logic.
- **String, binary, and bit-array utilities**: slicing, grapheme handling,
  encoding/decoding (base16/base64), UTF helpers, and the bit array builder all
  live here so they can share allocations and validation checks.
- **Math and numeric helpers**: floating-point conversions (`parse_float`,
  `float_to_string`, `ceiling`, `random_uniform`, etc.) and bitwise operations
  (`bitwise_and`, `bitwise_shift_left`, …) provide predictable behaviour across
  platforms.
- **Maps and dictionaries**: `dict_new`, `dict_get`, `dict_insert`, and related
  helpers wrap the runtime-managed hash map (`layout::Map` and `MapTable`).
- **IO & debugging**: printing helpers (`print`, `io_println`, `println_error`)
  are thin wrappers over `stdout`/`stderr` used by `gleam/io` and the test
  harness.
- **Glue for tests**: `gleeunit_main` and `gleeunit_do_main` are temporary hooks
  so the Gleeunit test runner can execute on the native backend.

Most functions follow the same pattern: convert raw `u64` values into `Value`,
perform some work via the safe helpers in `heap.rs`/`layout.rs`, and then return
results as raw words so they can be consumed from generated machine code.

## Adding new runtime features

When a new intrinsic or standard library feature needs runtime support:

1. Decide whether it belongs as a pure helper (add to an existing module) or as
   a new FFI export. Keep unsafe code isolated—prefer using `Heap`, `Value`, and
   layout helpers rather than pointer arithmetic in new code.
2. Add any required allocation or layout helpers first (for example a new
   header constructor in `layout.rs`).
3. Implement the public entry point in `ffi.rs`. Most of them should accept and
   return `u64` so they can be linked against directly from Cranelift-generated
   code. Reuse the `unwrap_allocation` helper to ensure allocation failures
  panic with a clear message.
4. Wire the new function into lowering or the standard library and add tests.
   Unit tests can live alongside the new logic (many modules already contain
   `#[cfg(test)]` sections). Integration coverage currently lives in the
   `native-test` Gleam project.

## Testing and debugging

- Run the crate’s unit tests with `cargo test -p runtime-cranelift`.
- The `native-test` project (`native-test/`) exercises the runtime via Gleam
  code. Rebuild it after making changes to ensure the generated binaries still
  link and run as expected.
- For ad-hoc debugging it can be useful to run binaries with `RUST_LOG=debug`
  and sprinkle temporary `eprintln!` statements in the runtime. Remember to
  remove debugging output before committing changes.

The runtime is still evolving along with the native backend. Expect some of the
APIs to change until the Cranelift target stabilises.

## Linking External Libraries

When a module uses `@external(cranelift, "lib", "symbol")`, the compiler now
expects to find a native library called `lib` (or a static archive/path) at link
time. You can configure additional linker arguments and search paths directly in
`gleam.toml`:

```toml
[cranelift]
linker = "clang"
linker-args = ["-Wl,-rpath,$ORIGIN/lib"]
search-paths = ["native/lib"]

[cranelift.targets."aarch64-apple-darwin"]
linker-args = ["-framework", "Security"]
search-paths = ["native/macos"]
```

The top-level `[cranelift]` table sets defaults for every Cranelift build. Use
`[cranelift.targets."<triple>"]` to extend or override those settings for a
particular platform triple. The collected settings are applied after Gleam’s
automatic `-l<library>` flags, so you can point the linker at custom locations
or pass through additional options as required.
