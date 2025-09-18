# Cranelift Runtime Value Layout

This document captures the proposed runtime representation for Gleam values when targeting the Cranelift backend. The goal is to balance execution speed, memory efficiency, and ease of interop with Rust-based runtime components while keeping room for future extensions (e.g. FFI, concurrency, multi-platform targeting).

## Word size and tagging

- Target platform: macOS arm64 initially, with 64-bit machine words.
- We reserve the lowest two bits of any pointer-sized word for tagging. Heap allocations respect a minimum alignment of 4 bytes, so these bits are otherwise unused.
- Tagged immediates share the same `Value` type as heap pointers. The general shape looks like:

```
| 63 ... 2 | 1 0 |
| payload  |tag|
```

- `tag = 0b01` signifies a small integer (fixnum). The payload is a signed 63-bit value obtained via arithmetic shift.
- `tag = 0b11` is used for other immediates (booleans, `Nil`, atoms) distinguished by additional high-bit patterns.
- `tag = 0b00` means the word is a pointer to a heap allocation whose header contains the runtime tag.
- `tag = 0b10` is currently unused, reserved for future encodings (e.g. NaN-boxed floats or special refs).

### Immediate encodings

- **Integers**: `(value << 2) | 0b01`, storing signed integers up to ±(2^61 − 1). Lowering emits overflow traps that fall back to boxed bigints later.
- **Booleans**: Boxed statics tagged as `Boolean` records so they behave like ordinary heap values while remaining globally shared.
- **Nil / empty list**: A boxed static tagged as `Nil`, shared across the runtime and treated as the list terminator.
- **Atoms**: `((index as u64) << 2) | 0b11`, where the upper bits store the atom table index. These remain immediate values unless the index exceeds the inline capacity.

## Heap object header

All boxed values begin with a common `Header` word:

```
struct Header {
    tag: u16,      // primary type tag (tuple, list, string, ...)
    arity: u16,    // payload-specific, e.g. tuple arity or constructor index
    size: u32,     // total payload words following the header (for GC)
}
```

- The header is stored as a single 64-bit word for compactness (`tag` in low 16 bits, `arity` next, `size` high 32 bits).
- Garbage collection can inspect the header to determine how many pointer fields follow versus raw bytes (see layout per type).

## Per-type layouts

### Floats

```
struct FloatBox {
    header(tag = FLOAT, arity = 0, size = 1)
    value: f64
}
```

- Every float is boxed to avoid NaN-tag corner cases on arm64.
- Future optimisation: inline small rationals or leverage vector units under a dedicated tag.

### Strings and binaries

```
struct Binary {
    header(tag = BINARY, arity = 0, size = 3)
    data: *mut u8       // pointer to owned buffer (aligned)
    len: usize          // number of bytes used
    capacity: usize     // buffer capacity for in-place growth
}
```

- Buffers are reference counted; the header is followed by a `RefCount` (stored just before `data`).
- A slice view uses the same structure with an additional `offset` payload word and shares the buffer.
- Match operations use dedicated runtime helpers to avoid copying slices unless mutated.

### Lists

```
struct ConsCell {
    header(tag = LIST, arity = 2, size = 2)
    head: Value
    tail: Value // either another cons cell or the Nil immediate
}
```

- The empty list uses the shared boxed `Nil` value provided by the runtime.
- Pattern matching checks the pointer tag; cons cells are linear in memory for efficient traversal.

### Tuples

```
struct Tuple {
    header(tag = TUPLE, arity = n, size = n)
    elements: [Value; n]
}
```

- Arity ≤ 2 may be specialised later (e.g. pair and closure-no-env) but initially share the generic form.
- The GC reads `arity` to know how many pointer slots follow.

### Records and custom types

```
struct Record {
    header(tag = RECORD, arity = field_count, size = field_count + 1)
    ctor_index: u32         // discriminant for the constructor
    padding: u32            // reserved for flags or cached hash
    fields: [Value; field_count]
}
```

- `ctor_index` allows pattern matching to branch without peeking at fields.
- Custom type variants are encoded as records with different constructor indices.

### Closures and functions

```
struct Closure {
    header(tag = CLOSURE, arity = env_size + 2, size = env_size + 2)
    code_ptr: *const fn(*const Closure, &[Value]) -> Value
    env_size: usize
    env: [Value; env_size]
}
```

- Zero-capture functions become tagged immediates referencing a static descriptor (future optimisation). Initially all functions box for simplicity.
- `env_size` allows GC to treat the environment slice as pointer fields.

### Maps / dictionaries

```
struct Map {
    header(tag = MAP, arity = 0, size = 1)
    table: *mut HashMap<Value, Value>
}
```

- Backed by a runtime-managed hash map implementing robin-hood probing, with copy-on-write semantics for structural sharing.
- GC treats the pointer as opaque; the hash map implementation performs tracing of entries.

### Bit arrays

```
struct BitArray {
    header(tag = BITARRAY, arity = 0, size = 4)
    data: *mut u8
    bit_offset: u32
    bit_len: u32
    capacity_bits: u32
}
```

- Shares the ref-counted buffer strategy with binaries but tracks bit-level start/end for pattern matching.

### Runtime-only nodes

- **Process mailbox (future)**: `struct Mailbox { header(tag = MAILBOX, ...); queue_head; queue_tail; }`
- **Resource handles** (ports, files) will use dedicated tags pointing to opaque structs managed by the runtime.

## Interaction with code generation

- The Cranelift lowering treats all values as `i64`. Pattern checks are comparisons against tag patterns or pointer loads of the header word.
- Constructors and destructors call runtime intrinsics to allocate the appropriate size, fill headers, and initialise payloads.
- Arithmetic and comparisons operate on immediates when tags allow; mismatched tags jump to runtime slow paths that perform dynamic dispatch.

## Memory management

- Reference-counted buffers (strings/binaries) use atomic counters to enable future multi-threading; helpers in `binary.rs` manage retain/release operations.
- Heap allocations route through the Boehm–Demers–Weiser collector via the vendored `bdwgc-sys` crate. The `Heap` facade calls into `GC_malloc`/`GC_gcollect` and keeps initialisation behind a `Once` guard.
- All other heap objects participate in this tracing GC; the header metadata (`payload_words`) makes scanning straightforward for future stack map integration.
- Roots will come from the stack maps generated by Cranelift; closures, list cells, and tuples are scanned uniformly once lowering wires in stack map emission.

## Open questions

- Finalise how constructor indices are assigned (per module vs. global registry) to ensure pattern matching across crates stays stable.
- Decide whether closure environments always store boxed values or can capture immediates directly without boxing.
- Evaluate the cost of boxing all floats; explore NaN-boxing or specialised SIMD values as the backend matures.

This layout is intended as the baseline for implementing runtime support and Cranelift lowering. Adjustments can be made as we collect performance data or introduce new language features.
