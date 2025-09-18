# Cranelift Backend TODOs

## Target Plumbing & Configuration
- [ ] Add `Cranelift` variant to `compiler-core/src/build.rs::Target`; update `variant_strings`, helpers, and `serde`/`strum` attributes.
- [ ] Extend `TargetCodegenConfiguration` with a `Cranelift` variant carrying native codegen settings (output paths, optimisation toggles).
- [ ] Update `compiler-core/src/paths.rs` helpers to create `build/<mode>/cranelift` directories and artefact paths.
- [ ] Wire `Cranelift` default behaviour into `compiler-core/src/config.rs` and ensure `gleam.toml` parsing emits/accepts the new enum value.
- [ ] Ensure all pattern matches over `Target` (core + CLI) handle the new variant.

## Cranelift Codegen Module
- [ ] Scaffold `compiler-core/src/cranelift/mod.rs` and submodules (expression lowering, pattern matching, name mangling, intrinsics).
- [ ] Define `ModuleConfig` equivalent carrying `TypedModule`, `LineNumbers`, and project metadata.
- [ ] Implement lowering for core Gleam constructs: literals, functions, pattern matches, custom types, lists/tuples, tail recursion.
- [ ] Emit Cranelift IR using per-function contexts; integrate with the chosen value representation.
- [ ] Provide diagnostics mapping back to source using `LineNumbers`.

## Runtime & Value Representation
- [ ] Document value layout decisions (records, enums, lists, strings, numbers) for native code.
- [ ] Evaluate existing Rust runtime components for reuse (memory management, scheduler) and outline integration strategy.
- [ ] Implement required wrappers/adapters exposing runtime functionality to generated Cranelift code.
- [ ] Add intrinsics for panic handling, comparison, arithmetic overflow, etc.
- [ ] Set up crate/module structure for the runtime (e.g. `runtime-cranelift`) and build scripts as needed.

## PackageCompiler Integration
- [ ] Add `perform_cranelift_codegen` to `compiler-core/src/build/package_compiler.rs` and invoke it for the Cranelift target.
- [ ] Generate per-module object files (`.o`) in `build/<mode>/cranelift/<package>/artefacts` using Cranelift’s backend.
- [ ] Store metadata (fingerprints, deps) alongside objects for cache reuse.
- [ ] Implement final link step producing both an executable entrypoint and a reusable static/dynamic library.
- [ ] Hook entrypoint generation into existing native file copying and metadata writing flows.

## CLI Workflow
- [ ] Accept `--target cranelift` in CLI argument parsing and help text.
- [ ] Update `gleam build/run/test` pipelines to select Cranelift configuration and call the new codegen path.
- [ ] Extend `gleam run` to link (if required) and execute the native binary, forwarding arguments.
- [ ] Ensure export/publish commands package native executables/libraries appropriately or explain limitations.

## Toolchain & Linking (macOS arm64 initial focus)
- [ ] Detect `cc`/`clang` availability; provide clear errors when missing.
- [ ] Implement linker invocation for macOS arm64 (object files -> executable/library) with support for debug info and optimisation flags.
- [ ] Add `gleam.toml` options for optimisation level, debug info, target triple (default `aarch64-apple-darwin`), linker flags, runtime linkage.
- [ ] Cache/link incremental outputs efficiently to avoid full relinks when possible.

## Target Support & Analysis
- [ ] Update `TargetSupport` enforcement to recognise Cranelift availability.
- [ ] Mark stdlib functions/features unsupported on Cranelift (initially) and emit helpful compile-time errors.
- [ ] Extend type-checker tests to cover Cranelift-specific constraints and inline function behaviour.

## Testing & Validation
- [ ] Add unit tests for Cranelift lowering modules with representative Gleam snippets.
- [ ] Create integration tests building/running sample projects via the Cranelift backend on macOS arm64.
- [ ] Mirror existing Erlang backend test coverage: replicate comparable suites (module generation, runtime behaviours) for Cranelift.
- [ ] Ensure incremental rebuild tests pass ( modify module, rebuild, verify reuse ).
- [ ] Wire backend tests into CI (initially optional or macOS-only job).

## Documentation & Rollout
- [ ] Update CLI help, README, and user guides with Cranelift usage instructions and macOS arm64 requirements.
- [ ] Document runtime expectations and limitations (no FFI yet, macOS-only initial support).
- [ ] Note the change in changelog and release notes once ready for distribution.
