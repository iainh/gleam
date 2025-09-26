# facil.io + Gleam Cranelift Demo

This sample shows how to call a facil.io HTTP server from Gleam running on the
Cranelift backend. The project uses the new `[cranelift]` configuration in
`gleam.toml` so that Gleam can find the native static library produced from the
C sources.

## Prerequisites

1. A working C toolchain (`clang`/`cc`, `libtool`, `make`).
2. Access to the facil.io repository. The build script will automatically clone
   `git@github.com:boazsegev/facil.io.git` into `native/vendor/facil.io` if it is
   not already present. You can point `FACIL_DIR` at an existing checkout if you
   prefer.

## Building the native wrapper

Running `make` inside `native/` compiles `src/facil_io_ffi.c` and bundles it
together with facil.io’s static libraries into `native/build/libfacil_io.a`.
Gleam’s
linker picks up this archive automatically based on the `[cranelift]` settings.

```bash
cd examples/facil_io/native
make
# If you already have a facil.io checkout elsewhere:
# FACIL_DIR=/path/to/facil.io make
# The default uses `native/facil.io`; set `FACIL_DIR` to reuse an existing
# checkout or avoid cloning.
```

After the native library is in place, fetch Gleam dependencies. The project
expects a sibling checkout of the Cranelift stdlib branch of
[`gleam-stdlib`](https://github.com/iainh/gleam-stdlib/tree/feature/cranelift-backend)
at `../../../gleam-stdlib`.

```bash
cd ..
cargo run -p gleam -- fetch
```

## Running the Gleam application

With the static library in place you can compile and run the Gleam project via
the Gleam executable built from this checkout:

```bash
cd examples/facil_io
cargo run -p gleam -- run -t native
```

The server listens on `http://localhost:3000` and responds with the message
hard-coded in `src/facil_io_ffi.c`.

## Notes

- The build script always invokes the `lib` target in the facil.io checkout.
  Set `FACIL_DIR` if you already have a clone elsewhere or want to control how
  facil.io is built.
- The `[cranelift]` configuration automatically adds `native/build` to the
  linker search path, so no extra environment setup is required when you use
  the Gleam binary from this branch.
- On platforms where additional system frameworks are required (for example
  `Security` on macOS) you can extend the linker arguments under
  `[cranelift.targets."<triple>"]` in `gleam.toml`.
- The example is intentionally small; you can expose additional facil.io
  functionality by extending `src/facil_io_ffi.c` and adding matching
  `@external(cranelift, ...)` declarations in Gleam. The HTTP request handler
  itself lives in Gleam (`facil_io.handle_request/1`) and calls back into
  `facil_io_send_body` to write responses. The helper
  utilities exposed in `runtime-cranelift/include/gleam_ffi.h` handle tagged
  integer, resource, list, and UTF-8 conversions so the C shim can stay small
  while passing facil.io request pointers and response data straight through to
  Gleam.
