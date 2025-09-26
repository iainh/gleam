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

The `native/build.sh` script compiles `facil_wrapper.c` and bundles it together
with facil.io’s static libraries into `native/build/libfacil_io_demo.a`. Gleam’s
linker picks up this archive automatically based on the `[cranelift]` settings.

```bash
cd examples/facil_io_demo/native
./build.sh
```

## Running the Gleam application

With the static library in place you can compile and run the Gleam project via
Cranelift:

```bash
cd examples/facil_io_demo
GLEAM_TARGET=cranelift gleam run
```

The server listens on `http://localhost:3000` and responds with the message
hard-coded in `native/facil_wrapper.c`.

## Notes

- The build script first tries `make -C facil.io static` and falls back to the
  `build` target if `static` is unavailable. Set `FACIL_DIR` if you already have
  a clone elsewhere or want to control how facil.io is built.
- On platforms where additional system frameworks are required (for example
  `Security` on macOS) you can extend the linker arguments under
  `[cranelift.targets."<triple>"]` in `gleam.toml`.
- The example is intentionally small; you can expose additional facil.io
  functionality by adding new `extern "C"` functions to `facil_wrapper.c` and
  corresponding `@external(cranelift, ...)` declarations in Gleam.
