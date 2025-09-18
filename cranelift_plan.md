# Cranelift Backend Implementation Plan

## 1. Extend Target Plumbing
- Add a `Cranelift` variant to `compiler-core/src/build.rs::Target` and update helper methods (`variant_strings`, `is_*`).
- Extend `TargetCodegenConfiguration` to include a native configuration struct for Cranelift output (artifact layout, optimisation level, linking options).
- Update path helpers (`compiler-core/src/paths.rs`) to create `build/<mode>/cranelift` directories and any target-specific file locations.
- Wire defaults in `compiler-core/src/config.rs` and migrate `gleam.toml` parsing/serialisation so user projects can opt into the new target.

## 2. Cranelift Code Generation Module
- Create a `compiler-core/src/cranelift/` module (or feature-gated crate) mirroring the structure of `javascript` renderer.
- Define a lowering pipeline from `TypedModule` to Cranelift IR: handle expressions, pattern matching, tail recursion, data constructors, and effectful primitives.
- Leverage `LineNumbers` for diagnostics and emulate `ModuleConfig` to encapsulate per-module metadata.
- Implement support utilities (name mangling, environment tracking, constant pooling) akin to Erlang/JS generators.

## 3. Runtime & Value Representation
- Design Gleam value layouts for native code (records, custom types, lists, tuples, strings, numbers).
- Leverage existing Rust implementations for memory management and scheduling to minimise bespoke runtime work; adapt as needed for Gleam semantics.
- Defer FFI surface design until a later milestone; keep runtime boundaries clean so it can be added incrementally.
- Provide intrinsics/utilities callable from both generated code and user-written native modules.

- Introduce `perform_cranelift_codegen` in `compiler-core/src/build/package_compiler.rs`, invoked when the target config is Cranelift.
- Lower each module to native object files (`.o`/`.obj`) written under `build/<mode>/cranelift/<package>/artefacts`.
- Add a deterministic final link step that combines fresh objects with cached ones to produce both executables (for `gleam run`) and reusable libraries.
- Integrate with caching: store metadata per object so unchanged modules skip codegen and linking reuses prior outputs where possible.
- Generate entrypoint binaries, honour `copy_native_files`, and ensure native assets coexist cleanly with Cranelift output.

- Accept `--target cranelift` across CLI commands (`gleam build/run/test`, `CompilePackage`, etc.).
- Extend runtime selection/validation: for `gleam run`, build and execute the produced native binary, handling arguments and env vars.
- Update export/publish commands to include both executables and libraries in native artifacts.

- Standardise on per-module object files plus a final link stage; document default outputs (executables for `gleam run`, libraries for reuse).
- Target macOS arm64 initially; implement linker invocation via `cc`/`clang` for that platform first, leaving hooks for future expansion.
- Add configuration knobs in `gleam.toml` for optimisation level, debug info, target triple, linker flags, and runtime linkage.
- Implement diagnostics for missing or misconfigured toolchains, surfacing actionable `Error` variants with setup guidance.

## 7. Target Support & Analysis
- Update `TargetSupport` enforcement to know which standard-library features are available under Cranelift.
- Add target-specific intrinsics or library implementations, ensuring inline functions and metadata caches remain accurate.
- Expand `type_` and analysis tests covering target constraints (e.g., unsupported features returning compile-time errors).

## 8. Testing & Validation
- Create golden tests for generated Cranelift IR/object files and integration tests that compile & run sample projects.
- Add CI jobs (initially optional/feature-flagged) to exercise the backend on supported platforms.
- Verify incremental builds, `gleam run`, `gleam test`, and publish workflows across all targets before default rollout.

## 9. Documentation & Rollout
- Update documentation, CLI help, and changelog to describe the Cranelift backend and configuration.
- Provide migration guidance, noting current limitations (FFI support, concurrency model, platform coverage).
- Launch behind a preview flag until the runtime and tooling are production ready; gather feedback for stabilization.
