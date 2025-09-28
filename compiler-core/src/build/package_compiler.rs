use crate::analyse::{ModuleAnalyzerConstructor, TargetSupport};
use crate::build::package_loader::CacheFiles;
use crate::inline;
use crate::io::files_with_extension;
use crate::line_numbers::{self, LineNumbers};
use crate::type_::PRELUDE_MODULE_NAME;
use crate::{
    Error, Result, Warning,
    ast::{SrcSpan, TypedModule, UntypedModule},
    build::{
        Mode, Module, Origin, Outcome, Package, SourceFingerprint, Target,
        collect_cranelift_external_modules,
        elixir_libraries::ElixirLibraries,
        native_file_copier::NativeFileCopier,
        package_loader::{CodegenRequired, PackageLoader, StaleTracker},
    },
    codegen::{Erlang, ErlangApp, JavaScript, TypeScriptDeclarations},
    config::{CraneliftLinkerSettings, PackageConfig},
    dep_tree, error,
    io::{BeamCompiler, Command, CommandExecutor, FileSystemReader, FileSystemWriter, Stdio},
    metadata::ModuleEncoder,
    parse::extra::ModuleExtra,
    paths, type_,
    uid::UniqueIdGenerator,
    warning::{TypeWarningEmitter, WarningEmitter},
};
use askama::Template;
use cranelift_codegen::{ir::InstBuilder, settings::Configurable};
use cranelift_module::Module as _;
use cranelift_native;
use ecow::EcoString;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::{env, fmt::write, fs, path::Path, time::SystemTime};
use vec1::Vec1;

use camino::{Utf8Path, Utf8PathBuf};

use super::{
    CraneliftCodegenConfiguration, ErlangAppCodegenConfiguration, TargetCodegenConfiguration,
    Telemetry,
    runtime_lib::{RuntimeArtifacts, RuntimeLibraryKind, locate_runtime_artifacts},
};

pub struct Compiled {
    /// The modules which were just compiled
    pub modules: Vec<Module>,
    /// The names of all cached modules, which are not present in the `modules` field.
    pub cached_module_names: Vec<EcoString>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CraneliftManifest {
    #[serde(default)]
    app_objects: Vec<String>,
    #[serde(default)]
    test_objects: Vec<String>,
    #[serde(default)]
    objects: Vec<String>,
}

struct NativeObjectGroups {
    app_objects: Vec<Utf8PathBuf>,
    test_objects: Vec<Utf8PathBuf>,
}

#[derive(Debug)]
pub struct PackageCompiler<'a, IO> {
    pub io: IO,
    pub out: &'a Utf8Path,
    pub lib: &'a Utf8Path,
    pub root: &'a Utf8Path,
    pub mode: Mode,
    pub target: &'a TargetCodegenConfiguration,
    pub config: &'a PackageConfig,
    pub ids: UniqueIdGenerator,
    pub write_metadata: bool,
    pub perform_codegen: bool,
    /// If set to false the compiler won't load and analyse any of the package's
    /// modules and always succeed compilation returning no compile modules.
    ///
    /// Code generation is still carried out so that a root package will have an
    /// entry point nonetheless.
    ///
    pub compile_modules: bool,
    pub write_entrypoint: bool,
    pub copy_native_files: bool,
    pub compile_beam_bytecode: bool,
    pub subprocess_stdio: Stdio,
    pub target_support: TargetSupport,
    pub cached_warnings: CachedWarnings,
    pub check_module_conflicts: CheckModuleConflicts,
}

impl<'a, IO> PackageCompiler<'a, IO>
where
    IO: FileSystemReader + FileSystemWriter + CommandExecutor + BeamCompiler + Clone,
{
    pub fn new(
        config: &'a PackageConfig,
        mode: Mode,
        root: &'a Utf8Path,
        out: &'a Utf8Path,
        lib: &'a Utf8Path,
        target: &'a TargetCodegenConfiguration,
        ids: UniqueIdGenerator,
        io: IO,
    ) -> Self {
        Self {
            io,
            ids,
            out,
            lib,
            root,
            mode,
            config,
            target,
            write_metadata: true,
            perform_codegen: true,
            compile_modules: true,
            write_entrypoint: false,
            copy_native_files: true,
            compile_beam_bytecode: true,
            subprocess_stdio: Stdio::Inherit,
            target_support: TargetSupport::NotEnforced,
            cached_warnings: CachedWarnings::Ignore,
            check_module_conflicts: CheckModuleConflicts::DoNotCheck,
        }
    }

    /// Compile the package.
    /// Returns a list of modules that were compiled. Any modules that were read
    /// from the cache will not be returned.
    // TODO: return the cached modules.
    pub fn compile(
        mut self,
        warnings: &WarningEmitter,
        existing_modules: &mut im::HashMap<EcoString, type_::ModuleInterface>,
        already_defined_modules: &mut im::HashMap<EcoString, Utf8PathBuf>,
        stale_modules: &mut StaleTracker,
        incomplete_modules: &mut HashSet<EcoString>,
        telemetry: &dyn Telemetry,
    ) -> Outcome<Compiled, Error> {
        let span = tracing::info_span!("compile", package = %self.config.name.as_str());
        let _enter = span.enter();

        // Ensure that the package is compatible with this version of Gleam
        if let Err(e) = self.config.check_gleam_compatibility() {
            return e.into();
        }

        let artefact_directory = self.out.join(paths::ARTEFACT_DIRECTORY_NAME);
        let codegen_required = if self.perform_codegen {
            CodegenRequired::Yes
        } else {
            CodegenRequired::No
        };

        let loader = PackageLoader::new(
            self.io.clone(),
            self.ids.clone(),
            self.mode,
            self.root,
            self.cached_warnings,
            warnings,
            codegen_required,
            &artefact_directory,
            self.target.target(),
            &self.config.name,
            stale_modules,
            already_defined_modules,
            incomplete_modules,
        );

        let loaded = if self.compile_modules {
            match loader.run() {
                Ok(loaded) => loaded,
                Err(error) => return error.into(),
            }
        } else {
            Loaded::empty()
        };

        let mut cached_module_names = Vec::new();

        // Load the cached modules that have previously been compiled
        for module in loaded.cached.into_iter() {
            // Emit any cached warnings.
            // Note that `self.cached_warnings` is set to `Ignore` (such as for
            // dependency packages) then this field will not be populated.
            if let Err(e) = self.emit_warnings(warnings, &module) {
                return e.into();
            }

            cached_module_names.push(module.name.clone());

            // Register the cached module so its type information etc can be
            // used for compiling futher modules.
            _ = existing_modules.insert(module.name.clone(), module);
        }

        if !loaded.to_compile.is_empty() {
            // Print that work is being done
            if self.perform_codegen {
                telemetry.compiling_package(&self.config.name);
            } else {
                telemetry.checking_package(&self.config.name)
            }
        }

        // Type check the modules that are new or have changed
        tracing::info!(count=%loaded.to_compile.len(), "analysing_modules");
        let outcome = analyse(
            &self.config,
            self.target.target(),
            self.mode,
            &self.ids,
            loaded.to_compile,
            existing_modules,
            warnings,
            self.target_support,
            incomplete_modules,
        );

        let modules = match outcome {
            Outcome::Ok(modules) => modules,
            Outcome::PartialFailure(modules, error) => {
                return Outcome::PartialFailure(
                    Compiled {
                        modules,
                        cached_module_names,
                    },
                    error,
                );
            }
            Outcome::TotalFailure(error) => return Outcome::TotalFailure(error),
        };

        tracing::debug!("performing_code_generation");

        let modules = if self.perform_codegen {
            modules
                .into_iter()
                .map(|mut module| {
                    module.ast = inline::module(module.ast, &existing_modules);
                    module
                })
                .collect()
        } else {
            modules
        };

        if let Err(error) = self.perform_codegen(&modules) {
            return error.into();
        }

        if let Err(error) = self.encode_and_write_metadata(&modules) {
            return error.into();
        }

        Outcome::Ok(Compiled {
            modules,
            cached_module_names,
        })
    }

    fn compile_erlang_to_beam(
        &mut self,
        modules: &HashSet<Utf8PathBuf>,
    ) -> Result<Vec<EcoString>, Error> {
        if modules.is_empty() {
            tracing::debug!("no_erlang_to_compile");
            return Ok(Vec::new());
        }

        tracing::debug!("compiling_erlang");

        self.io
            .compile_beam(self.out, self.lib, modules, self.subprocess_stdio)
            .map(|modules| modules.iter().map(|str| EcoString::from(str)).collect())
    }

    fn copy_project_native_files(
        &mut self,
        destination_dir: &Utf8Path,
        to_compile_modules: &mut HashSet<Utf8PathBuf>,
    ) -> Result<(), Error> {
        tracing::debug!("copying_native_source_files");

        // TODO: unit test
        let priv_source = self.root.join("priv");
        let priv_build = self.out.join("priv");
        if self.io.is_directory(&priv_source) && !self.io.is_directory(&priv_build) {
            tracing::debug!("linking_priv_to_build");
            self.io.symlink_dir(&priv_source, &priv_build)?;
        }

        let copier = NativeFileCopier::new(
            self.io.clone(),
            self.root.clone(),
            destination_dir,
            self.check_module_conflicts,
        );
        let copied = copier.run()?;

        to_compile_modules.extend(copied.to_compile.into_iter());

        // If there are any Elixir files then we need to locate Elixir
        // installed on this system for use in compilation.
        if copied.any_elixir {
            ElixirLibraries::make_available(
                &self.io,
                &self.lib.to_path_buf(),
                self.subprocess_stdio,
            )?;
        }

        Ok(())
    }

    fn encode_and_write_metadata(&mut self, modules: &[Module]) -> Result<()> {
        if !self.write_metadata {
            tracing::debug!("package_metadata_writing_disabled");
            return Ok(());
        }
        if modules.is_empty() {
            return Ok(());
        }

        let artefact_dir = self.out.join(paths::ARTEFACT_DIRECTORY_NAME);

        tracing::debug!("writing_module_caches");
        for module in modules {
            let cache_files = CacheFiles::new(&artefact_dir, &module.name);

            // Write cache file
            let bytes = ModuleEncoder::new(&module.ast.type_info).encode()?;
            self.io.write_bytes(&cache_files.cache_path, &bytes)?;

            // Write cache metadata
            let info = CacheMetadata {
                mtime: module.mtime,
                codegen_performed: self.perform_codegen,
                dependencies: module.dependencies.clone(),
                fingerprint: SourceFingerprint::new(&module.code),
                line_numbers: module.ast.type_info.line_numbers.clone(),
            };
            self.io
                .write_bytes(&cache_files.meta_path, &info.to_binary())?;

            let cache_inline = bincode::serialize(&module.ast.type_info.inline_functions)
                .expect("Failed to serialise inline functions");
            self.io.write_bytes(&cache_files.inline_path, &cache_inline);

            // Write warnings.
            // Dependency packages don't get warnings persisted as the
            // programmer doesn't want to be told every time about warnings they
            // cannot fix directly.
            if self.cached_warnings.should_use() {
                let warnings = &module.ast.type_info.warnings;
                let data = bincode::serialize(warnings).expect("Serialise warnings");
                self.io.write_bytes(&cache_files.warnings_path, &data)?;
            }
        }
        Ok(())
    }

    fn perform_codegen(&mut self, modules: &[Module]) -> Result<()> {
        if !self.perform_codegen {
            tracing::debug!("skipping_codegen");
            return Ok(());
        }

        match self.target {
            TargetCodegenConfiguration::JavaScript {
                emit_typescript_definitions,
                prelude_location,
            } => self.perform_javascript_codegen(
                modules,
                *emit_typescript_definitions,
                prelude_location,
            ),
            TargetCodegenConfiguration::Erlang { app_file } => {
                self.perform_erlang_codegen(modules, app_file.as_ref())
            }
            TargetCodegenConfiguration::Native { native } => {
                self.perform_cranelift_codegen(modules, native)
            }
        }
    }

    fn perform_cranelift_codegen(
        &mut self,
        modules: &[Module],
        _config: &CraneliftCodegenConfiguration,
    ) -> Result<(), Error> {
        use crate::cranelift;

        let artefact_dir = self.out.join(paths::ARTEFACT_DIRECTORY_NAME);
        if !self.io.is_directory(&artefact_dir) {
            self.io.mkdir(&artefact_dir)?;
        }

        let triple = cranelift_native::builder()
            .map(|builder| builder.triple().to_string())
            .ok();
        let linker_settings = self.config.cranelift_linker_settings(triple.as_deref());

        let preferred_test_module = format!("{}_test", self.config.name.as_str());

        let app_entry_index = modules
            .iter()
            .enumerate()
            .filter(|(_, module)| module.origin != Origin::Test)
            .filter(|(_, module)| cranelift::module_contains_public_main(&module.ast))
            .min_by_key(|(_, module)| {
                (
                    match module.origin {
                        Origin::Src => 0_u8,
                        Origin::Dev => 1,
                        Origin::Test => 2,
                    },
                    if module.name == self.config.name {
                        0_u8
                    } else {
                        1
                    },
                    module.name.clone(),
                )
            })
            .map(|(index, _)| index);

        let test_entry_index = modules
            .iter()
            .enumerate()
            .filter(|(_, module)| module.origin == Origin::Test)
            .filter(|(_, module)| cranelift::module_contains_public_main(&module.ast))
            .min_by_key(|(_, module)| {
                (
                    if module.name.as_str() == preferred_test_module {
                        0_u8
                    } else {
                        1
                    },
                    module.name.clone(),
                )
            })
            .map(|(index, _)| index);

        tracing::debug!(
            module_count = %modules.len(),
            ?app_entry_index,
            ?test_entry_index,
            "native_primary_entry"
        );

        let mut app_objects = Vec::new();
        let mut test_objects = Vec::new();
        let mut all_objects = Vec::new();
        let mut main_symbols: Vec<Option<String>> = Vec::with_capacity(modules.len());
        let mut compiled_any = false;
        let mut external_modules = BTreeSet::new();

        for (index, module) in modules.iter().enumerate() {
            let object_name = format!("{}.o", module.name.replace("/", "__"));
            let output_path = artefact_dir.join(&object_name);
            let wants_app_entry = self.write_entrypoint && app_entry_index == Some(index);
            let wants_test_entry = self.write_entrypoint && test_entry_index == Some(index);
            let module_config = cranelift::ModuleConfig::with_entrypoint(
                module,
                self.root,
                wants_app_entry || wants_test_entry,
            );
            let main_symbol = cranelift::emit_object(&self.io, module_config, &output_path)?;
            main_symbols.push(main_symbol);
            compiled_any = true;

            for lib in &module.cranelift_externals {
                let _ = external_modules.insert(lib.clone());
            }

            if module.origin != Origin::Test {
                app_objects.push(output_path.clone());
            }
            if module.origin == Origin::Test {
                test_objects.push(output_path.clone());
            }
            all_objects.push(output_path);
        }

        if compiled_any {
            self.write_cranelift_manifest(&artefact_dir, &app_objects, &test_objects)?;
        } else {
            if let Some(existing) = self.read_cranelift_manifest(&artefact_dir)? {
                app_objects = existing.app_objects;
                test_objects = existing.test_objects;
            } else {
                app_objects = self.discover_cranelift_objects(&artefact_dir)?;
                test_objects = Vec::new();
            }

            if app_objects.is_empty() && test_objects.is_empty() {
                tracing::debug!("native_no_objects_to_link");
                return Ok(());
            }

            self.write_cranelift_manifest(&artefact_dir, &app_objects, &test_objects)?;

            all_objects = app_objects
                .iter()
                .cloned()
                .chain(test_objects.iter().cloned())
                .collect();
        }

        let external_modules: Vec<EcoString> = external_modules.into_iter().collect();

        let _ = self.create_cranelift_archive(&artefact_dir, &all_objects)?;

        let app_entry_symbol = app_entry_index
            .and_then(|index| main_symbols.get(index))
            .cloned()
            .flatten();
        let should_link_app =
            self.write_entrypoint && app_entry_symbol.is_some() && !app_objects.is_empty();
        if should_link_app {
            let mut app_link_objects = app_objects.clone();
            if let (Some(index), Some(symbol)) = (app_entry_index, app_entry_symbol.as_deref()) {
                let stub_name = format!("{}__entry_app.o", modules[index].name.replace("/", "__"));
                let stub_path =
                    self.emit_cranelift_entrypoint_object(&artefact_dir, &stub_name, symbol)?;
                app_link_objects.push(stub_path);
            }

            self.link_cranelift_objects(
                &artefact_dir,
                &app_link_objects,
                self.config.name.as_str(),
                &linker_settings,
                &external_modules,
            )?;
        } else {
            tracing::debug!(reason = "no app entrypoint", "native_link_skipped_app");
        }

        let test_entry_symbol = test_entry_index
            .and_then(|index| main_symbols.get(index))
            .cloned()
            .flatten();
        let should_link_tests = self.write_entrypoint && test_entry_symbol.is_some();
        if should_link_tests {
            let mut test_link_objects = app_objects.clone();
            test_link_objects.extend(test_objects.clone());

            if let (Some(index), Some(symbol)) = (test_entry_index, test_entry_symbol.as_deref()) {
                let stub_name = format!("{}__entry_test.o", modules[index].name.replace("/", "__"));
                let stub_path =
                    self.emit_cranelift_entrypoint_object(&artefact_dir, &stub_name, symbol)?;
                test_link_objects.push(stub_path);
            }

            if !test_link_objects.is_empty() {
                let test_output = format!("{}_test", self.config.name.as_str());
                self.link_cranelift_objects(
                    &artefact_dir,
                    &test_link_objects,
                    &test_output,
                    &linker_settings,
                    &external_modules,
                )?;
            }
        } else if !test_objects.is_empty() {
            tracing::debug!(reason = "no test entrypoint", "native_link_skipped_tests");
        }

        Ok(())
    }

    fn link_cranelift_objects(
        &self,
        artefact_dir: &Utf8Path,
        objects: &[Utf8PathBuf],
        output_basename: &str,
        linker_settings: &CraneliftLinkerSettings,
        external_libraries: &[EcoString],
    ) -> Result<(), Error> {
        tracing::debug!(
            search_paths = ?linker_settings.search_paths,
            linker_args = ?linker_settings.linker_args,
            "native_linker_settings",
        );
        if objects.is_empty() {
            tracing::debug!("no_objects_to_link");
            return Ok(());
        }

        let mut output_name = output_basename.replace('/', "__");
        if output_name.is_empty() {
            output_name = "module".into();
        }
        let exe_suffix = env::consts::EXE_SUFFIX;
        if !exe_suffix.is_empty() && !output_name.ends_with(exe_suffix) {
            output_name.push_str(exe_suffix);
        }
        let executable_path = artefact_dir.join(output_name);

        let runtime = locate_runtime_artifacts(&self.io)?;
        let runtime_library_for_link = prepare_runtime_library(&self.io, &runtime, &artefact_dir)?;

        let mut args = Vec::with_capacity(objects.len() + runtime.additional_libs.len() + 16);
        let mut seen = HashSet::new();

        for search_path in &linker_settings.search_paths {
            let path = Utf8Path::new(search_path);
            let resolved = if path.is_absolute() {
                path.to_path_buf()
            } else {
                self.root.join(path)
            };
            let arg = format!("-L{}", resolved);
            if seen.insert(arg.clone()) {
                args.push(arg);
            }
        }

        for path in objects {
            if seen.insert(path.as_str().to_string()) {
                args.push(path.as_str().to_string());
            }
        }

        for dependency in self.collect_dependency_native_objects()? {
            if seen.insert(dependency.clone()) {
                args.push(dependency);
            }
        }

        if runtime.kind == RuntimeLibraryKind::Shared {
            if let Some(rpath) = shared_runtime_rpath_flag() {
                if !linker_settings
                    .linker_args
                    .iter()
                    .any(|arg| arg.as_str() == rpath)
                {
                    args.push(rpath);
                }
            }
        }

        match runtime.kind {
            RuntimeLibraryKind::Static => {
                args.push(runtime_library_for_link.as_str().to_string());
            }
            RuntimeLibraryKind::Shared => {
                let runtime_path = runtime_library_for_link.as_str().to_string();
                if seen.insert(runtime_path.clone()) {
                    args.push(runtime_path);
                }
            }
        }
        for lib in &runtime.additional_libs {
            args.push(lib.as_str().to_string());
        }

        for library in external_libraries {
            if library.as_str() == "runtime_cranelift" {
                continue;
            }

            let argument = if library.ends_with(".a") || library.contains('/') {
                library.as_str().to_string()
            } else {
                format!("-l{}", library)
            };

            if seen.insert(argument.clone()) {
                args.push(argument);
            }
        }

        args.extend(
            linker_settings
                .linker_args
                .iter()
                .map(|arg| arg.to_string()),
        );
        args.push("-lpthread".into());
        #[cfg(target_os = "linux")]
        {
            args.push("-ldl".into());
        }
        args.push("-o".into());
        args.push(executable_path.as_str().to_string());

        let linker_program = linker_settings.linker.as_deref().unwrap_or("cc");
        let command_for_display = args.join(" ");
        tracing::debug!(
            program = %linker_program,
            command = %command_for_display,
            "invoking_native_linker",
        );
        let link_output = std::process::Command::new(linker_program)
            .args(&args)
            .output()
            .map_err(|err| Error::NativeCodegen {
                message: format!("failed to invoke linker: {err}"),
            })?;

        if !link_output.status.success() {
            let stderr = String::from_utf8_lossy(&link_output.stderr);
            return Err(Error::NativeCodegen {
                message: format!(
                    "linker failed: {}\ncommand: {} {}",
                    stderr.trim(),
                    linker_program,
                    command_for_display
                ),
            });
        }

        #[cfg(target_os = "macos")]
        if runtime.kind == RuntimeLibraryKind::Shared {
            if let Some(binary) = runtime.runtime_binary.as_ref() {
                if let Some(file_name) = binary.file_name() {
                    update_macos_executable_runtime_ref(
                        &self.io,
                        &executable_path,
                        &runtime,
                        runtime_library_for_link.as_path(),
                        file_name,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn write_cranelift_manifest(
        &self,
        artefact_dir: &Utf8Path,
        app_objects: &[Utf8PathBuf],
        test_objects: &[Utf8PathBuf],
    ) -> Result<(), Error> {
        let manifest = CraneliftManifest {
            app_objects: app_objects.iter().map(|p| p.as_str().to_string()).collect(),
            test_objects: test_objects
                .iter()
                .map(|p| p.as_str().to_string())
                .collect(),
            objects: Vec::new(),
        };

        let manifest_text =
            serde_json::to_string_pretty(&manifest).map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?;

        self.io
            .write(&artefact_dir.join("manifest.json"), &manifest_text)
    }

    fn read_cranelift_manifest(
        &self,
        artefact_dir: &Utf8Path,
    ) -> Result<Option<NativeObjectGroups>, Error> {
        let manifest_path = artefact_dir.join("manifest.json");
        if !self.io.is_file(&manifest_path) {
            return Ok(None);
        }

        let manifest_text = self.io.read(&manifest_path)?;
        let mut manifest: CraneliftManifest =
            serde_json::from_str(&manifest_text).map_err(|err| Error::NativeCodegen {
                message: format!("failed to parse native manifest `{}`: {err}", manifest_path),
            })?;

        let mut app_objects = manifest
            .app_objects
            .into_iter()
            .map(Utf8PathBuf::from)
            .collect::<Vec<_>>();

        let mut test_objects = manifest
            .test_objects
            .into_iter()
            .map(Utf8PathBuf::from)
            .collect::<Vec<_>>();

        if app_objects.is_empty() && test_objects.is_empty() && !manifest.objects.is_empty() {
            app_objects = manifest
                .objects
                .into_iter()
                .map(Utf8PathBuf::from)
                .collect();
        }

        Ok(Some(NativeObjectGroups {
            app_objects,
            test_objects,
        }))
    }

    fn discover_cranelift_objects(
        &self,
        artefact_dir: &Utf8Path,
    ) -> Result<Vec<Utf8PathBuf>, Error> {
        if !self.io.is_directory(artefact_dir) {
            return Ok(Vec::new());
        }

        let mut objects = Vec::new();
        if let Ok(entries) = self.io.read_dir(artefact_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.into_path();
                    if path.extension() == Some("o") {
                        objects.push(path);
                    }
                }
            }
        }

        Ok(objects)
    }

    fn create_cranelift_archive(
        &self,
        artefact_dir: &Utf8Path,
        objects: &[Utf8PathBuf],
    ) -> Result<Option<Utf8PathBuf>, Error> {
        if objects.is_empty() {
            return Ok(None);
        }

        let mut archive_name = format!("lib{}", self.config.name.as_str().replace('/', "__"));
        if !archive_name.ends_with(".a") {
            archive_name.push_str(".a");
        }
        let archive_path = artefact_dir.join(archive_name);

        #[cfg(target_os = "windows")]
        let status = {
            let mut command = std::process::Command::new("lib");
            let _ = command.arg(format!("/OUT:{}", archive_path.as_str()));
            for object in objects {
                let _ = command.arg(object.as_str());
            }
            command.status()
        };

        #[cfg(not(target_os = "windows"))]
        let status = {
            let mut command = std::process::Command::new("ar");
            let _ = command.arg("crs");
            let _ = command.arg(archive_path.as_str());
            for object in objects {
                let _ = command.arg(object.as_str());
            }
            command.status()
        };

        let status = status.map_err(|err| Error::NativeCodegen {
            message: format!("failed to create native archive: {err}"),
        })?;

        if !status.success() {
            return Err(Error::NativeCodegen {
                message: format!("failed to create native archive `{}`", archive_path),
            });
        }

        #[cfg(not(target_os = "windows"))]
        {
            let ranlib_status = std::process::Command::new("ranlib")
                .arg(archive_path.as_str())
                .status()
                .map_err(|err| Error::NativeCodegen {
                    message: format!("failed to index native archive: {err}"),
                })?;

            if !ranlib_status.success() {
                return Err(Error::NativeCodegen {
                    message: format!("failed to index native archive `{}`", archive_path),
                });
            }
        }

        Ok(Some(archive_path))
    }

    fn collect_dependency_native_objects(&self) -> Result<Vec<String>, Error> {
        let mut libraries = Vec::new();
        let mut objects = Vec::new();

        if let Ok(entries) = self.io.read_dir(self.lib) {
            for entry in entries {
                let Ok(entry) = entry else { continue };
                let path = entry.into_path();
                if !self.io.is_directory(&path) {
                    continue;
                }

                if let Some(dir_name) = path.file_name() {
                    if dir_name == self.config.name.as_str() {
                        continue;
                    }
                }

                let artefacts_dir = path.join(paths::ARTEFACT_DIRECTORY_NAME);
                if !self.io.is_directory(&artefacts_dir) {
                    continue;
                }

                if let Ok(entries) = self.io.read_dir(&artefacts_dir) {
                    for entry in entries {
                        if let Ok(entry) = entry {
                            let path = entry.into_path();
                            if path.extension() == Some("a") {
                                libraries.push(path.to_string());
                            } else if path.extension() == Some("o") {
                                objects.push(path.to_string());
                            }
                        }
                    }
                }
            }
        }

        libraries.extend(objects);
        Ok(libraries)
    }

    fn perform_erlang_codegen(
        &mut self,
        modules: &[Module],
        app_file_config: Option<&ErlangAppCodegenConfiguration>,
    ) -> Result<(), Error> {
        let mut written = HashSet::new();
        let build_dir = self.out.join(paths::ARTEFACT_DIRECTORY_NAME);
        let include_dir = self.out.join("include");
        let io = self.io.clone();

        io.mkdir(&build_dir)?;

        if self.copy_native_files {
            self.copy_project_native_files(&build_dir, &mut written)?;
        } else {
            tracing::debug!("skipping_native_file_copying");
        }

        if self.compile_beam_bytecode && self.write_entrypoint {
            self.render_erlang_entrypoint_module(&build_dir, &mut written)?;
        } else {
            tracing::debug!("skipping_entrypoint_generation");
        }

        // NOTE: This must come after `copy_project_native_files` to ensure that
        // we overwrite any precompiled Erlang that was included in the Hex
        // package. Otherwise we will build the potentially outdated precompiled
        // version and not the newly compiled version.
        Erlang::new(&build_dir, &include_dir).render(io.clone(), modules, self.root)?;

        let native_modules: Vec<EcoString> = if self.compile_beam_bytecode {
            written.extend(modules.iter().map(Module::compiled_erlang_path));
            self.compile_erlang_to_beam(&written)?
        } else {
            tracing::debug!("skipping_erlang_bytecode_compilation");
            Vec::new()
        };

        if let Some(config) = app_file_config {
            ErlangApp::new(&self.out.join("ebin"), config).render(
                io,
                &self.config,
                modules,
                native_modules,
            )?;
        }
        Ok(())
    }

    fn perform_javascript_codegen(
        &mut self,
        modules: &[Module],
        typescript: bool,
        prelude_location: &Utf8Path,
    ) -> Result<(), Error> {
        let mut written = HashSet::new();
        let typescript = if typescript {
            TypeScriptDeclarations::Emit
        } else {
            TypeScriptDeclarations::None
        };

        JavaScript::new(&self.out, typescript, prelude_location, &self.root).render(
            &self.io,
            modules,
            self.stdlib_package(),
        )?;

        if self.copy_native_files {
            self.copy_project_native_files(&self.out, &mut written)?;
        } else {
            tracing::debug!("skipping_native_file_copying");
        }

        Ok(())
    }

    fn emit_cranelift_entrypoint_object(
        &self,
        artefact_dir: &Utf8Path,
        filename: &str,
        main_symbol: &str,
    ) -> Result<Utf8PathBuf, Error> {
        let isa_builder = cranelift_native::builder().map_err(|err| Error::NativeCodegen {
            message: err.to_string(),
        })?;

        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder
            .set("is_pic", "true")
            .map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?;
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);

        let isa = isa_builder
            .finish(flags)
            .map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?;

        let mut module = cranelift_object::ObjectModule::new(
            cranelift_object::ObjectBuilder::new(
                isa,
                format!("gleam_entry_{}", filename.replace('.', "_")),
                cranelift_module::default_libcall_names(),
            )
            .map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?,
        );

        let pointer_type = module.target_config().pointer_type();
        let mut signature = module.make_signature();
        signature
            .returns
            .push(cranelift_codegen::ir::AbiParam::new(pointer_type));
        let main_func = module
            .declare_function(main_symbol, cranelift_module::Linkage::Import, &signature)
            .map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?;

        let mut ctx = module.make_context();
        ctx.func
            .signature
            .returns
            .push(cranelift_codegen::ir::AbiParam::new(
                cranelift_codegen::ir::types::I32,
            ));

        let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
        let mut builder = cranelift_frontend::FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);

        let init_signature = module.make_signature();
        let runtime_init = module
            .declare_function(
                "gleam_runtime_init",
                cranelift_module::Linkage::Import,
                &init_signature,
            )
            .map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?;
        let runtime_init_ref = module.declare_func_in_func(runtime_init, &mut builder.func);
        let _ = builder.ins().call(runtime_init_ref, &[]);

        let main_ref = module.declare_func_in_func(main_func, &mut builder.func);
        let _ = builder.ins().call(main_ref, &[]);

        let zero = builder.ins().iconst(cranelift_codegen::ir::types::I32, 0);
        let _ = builder.ins().return_(&[zero]);
        builder.finalize();

        let func_id = module
            .declare_function(
                "main",
                cranelift_module::Linkage::Export,
                &ctx.func.signature,
            )
            .map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?;

        module
            .define_function(func_id, &mut ctx)
            .map_err(|err| Error::NativeCodegen {
                message: err.to_string(),
            })?;
        module.clear_context(&mut ctx);

        let product = module.finish();
        let bytes = product.emit().map_err(|err| Error::NativeCodegen {
            message: err.to_string(),
        })?;

        let output_path = artefact_dir.join(filename);
        self.io.write_bytes(&output_path, &bytes)?;
        Ok(output_path)
    }

    fn render_erlang_entrypoint_module(
        &mut self,
        out: &Utf8Path,
        modules_to_compile: &mut HashSet<Utf8PathBuf>,
    ) -> Result<(), Error> {
        let name = format!("{name}@@main.erl", name = self.config.name);
        let path = out.join(&name);

        // If the entrypoint module has already been created then we don't need
        // to write and compile it again.
        if self.io.is_file(&path) {
            tracing::debug!("erlang_entrypoint_already_exists");
            return Ok(());
        }

        let template = ErlangEntrypointModule {
            application: &self.config.name,
        };
        let module = template.render().expect("Erlang entrypoint rendering");
        self.io.write(&path, &module)?;
        let _ = modules_to_compile.insert(name.into());
        tracing::debug!("erlang_entrypoint_written");
        Ok(())
    }

    fn emit_warnings(
        &self,
        warnings: &WarningEmitter,
        module: &type_::ModuleInterface,
    ) -> Result<()> {
        for warning in &module.warnings {
            let src = self.io.read(&module.src_path)?;
            warnings.emit(Warning::Type {
                path: module.src_path.clone(),
                src: src.into(),
                warning: warning.clone(),
            });
        }

        Ok(())
    }

    fn stdlib_package(&self) -> StdlibPackage {
        if self.config.dependencies.contains_key("gleam_stdlib")
            || self.config.dev_dependencies.contains_key("gleam_stdlib")
        {
            StdlibPackage::Present
        } else {
            StdlibPackage::Missing
        }
    }
}

fn prepare_runtime_library<IO>(
    io: &IO,
    runtime: &RuntimeArtifacts,
    artefact_dir: &Utf8Path,
) -> Result<Utf8PathBuf, Error>
where
    IO: FileSystemWriter + CommandExecutor,
{
    if runtime.kind != RuntimeLibraryKind::Shared {
        return Ok(runtime.runtime_lib.clone());
    }

    let Some(binary) = runtime.runtime_binary.as_ref() else {
        return Ok(runtime.runtime_lib.clone());
    };

    let Some(file_name) = binary.file_name() else {
        return Ok(runtime.runtime_lib.clone());
    };

    let destination = artefact_dir.join(file_name);
    if destination != *binary {
        io.copy(binary, &destination)?;
    }

    #[cfg(target_os = "macos")]
    adjust_macos_install_name(io, &destination, file_name)?;

    if cfg!(target_os = "windows") {
        Ok(runtime.runtime_lib.clone())
    } else {
        Ok(destination)
    }
}

#[cfg(target_os = "macos")]
fn adjust_macos_install_name<IO>(
    io: &IO,
    library_path: &Utf8Path,
    file_name: &str,
) -> Result<(), Error>
where
    IO: CommandExecutor,
{
    let id = format!("@rpath/{file_name}");
    let status = io.exec(Command {
        program: "install_name_tool".into(),
        args: vec!["-id".into(), id, library_path.as_str().to_string()],
        env: Vec::new(),
        cwd: None,
        stdio: Stdio::Null,
    })?;

    if status != 0 {
        return Err(Error::NativeCodegen {
            message: format!("install_name_tool failed while updating {}", library_path),
        });
    }

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn adjust_macos_install_name<IO>(
    _io: &IO,
    _library_path: &Utf8Path,
    _file_name: &str,
) -> Result<(), Error>
where
    IO: CommandExecutor,
{
    Ok(())
}

#[cfg(target_os = "macos")]
fn update_macos_executable_runtime_ref<IO>(
    io: &IO,
    executable_path: &Utf8Path,
    runtime: &RuntimeArtifacts,
    runtime_library_for_link: &Utf8Path,
    file_name: &str,
) -> Result<(), Error>
where
    IO: CommandExecutor,
{
    use std::collections::HashSet;

    let new_path = format!("@rpath/{file_name}");
    let executable = executable_path.as_str().to_string();
    let mut attempted = HashSet::new();

    let mut candidates = Vec::new();
    candidates.push(runtime.runtime_lib.as_str().to_string());
    candidates.push(runtime_library_for_link.as_str().to_string());

    if let Some(parent) = runtime.runtime_lib.parent() {
        let candidate = parent.join("deps").join(file_name);
        candidates.push(candidate.to_string());
    }

    if let Some(parent) = runtime_library_for_link.parent() {
        let candidate = parent.join("deps").join(file_name);
        candidates.push(candidate.to_string());
    }

    for original in candidates {
        if !attempted.insert(original.clone()) {
            continue;
        }

        let status = io.exec(Command {
            program: "install_name_tool".into(),
            args: vec![
                "-change".into(),
                original.clone(),
                new_path.clone(),
                executable.clone(),
            ],
            env: Vec::new(),
            cwd: None,
            stdio: Stdio::Null,
        })?;

        if status == 0 {
            return Ok(());
        }
    }

    Err(Error::NativeCodegen {
        message: format!(
            "install_name_tool failed to update runtime reference in {}",
            executable_path
        ),
    })
}

#[cfg(not(target_os = "macos"))]
fn update_macos_executable_runtime_ref<IO>(
    _io: &IO,
    _executable_path: &Utf8Path,
    _runtime: &RuntimeArtifacts,
    _runtime_library_for_link: &Utf8Path,
    _file_name: &str,
) -> Result<(), Error>
where
    IO: CommandExecutor,
{
    Ok(())
}

fn shared_runtime_rpath_flag() -> Option<String> {
    match std::env::consts::OS {
        "macos" => Some("-Wl,-rpath,@loader_path".into()),
        "linux" => Some("-Wl,-rpath,$ORIGIN".into()),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StdlibPackage {
    Present,
    Missing,
}

fn analyse(
    package_config: &PackageConfig,
    target: Target,
    mode: Mode,
    ids: &UniqueIdGenerator,
    mut parsed_modules: Vec<UncompiledModule>,
    module_types: &mut im::HashMap<EcoString, type_::ModuleInterface>,
    warnings: &WarningEmitter,
    target_support: TargetSupport,
    incomplete_modules: &mut HashSet<EcoString>,
) -> Outcome<Vec<Module>, Error> {
    let mut modules = Vec::with_capacity(parsed_modules.len() + 1);
    let direct_dependencies = package_config.dependencies_for(mode).expect("Package deps");
    let dev_dependencies = package_config.dev_dependencies.keys().cloned().collect();

    // Insert the prelude
    // DUPE: preludeinsertion
    // TODO: Currently we do this here and also in the tests. It would be better
    // to have one place where we create all this required state for use in each
    // place.
    let _ = module_types.insert(PRELUDE_MODULE_NAME.into(), type_::build_prelude(ids));

    for UncompiledModule {
        name,
        code,
        ast,
        path,
        mtime,
        origin,
        package,
        dependencies,
        extra,
    } in parsed_modules
    {
        tracing::debug!(module = ?name, "Type checking");

        let line_numbers = LineNumbers::new(&code);

        let analysis = crate::analyse::ModuleAnalyzerConstructor {
            target,
            ids,
            origin,
            importable_modules: module_types,
            warnings: &TypeWarningEmitter::new(path.clone(), code.clone(), warnings.clone()),
            direct_dependencies: &direct_dependencies,
            dev_dependencies: &dev_dependencies,
            target_support,
            package_config,
        }
        .infer_module(ast, line_numbers, path.clone());

        match analysis {
            Outcome::Ok(ast) => {
                // Module has compiled successfully. Make sure it isn't marked as incomplete.
                let _ = incomplete_modules.remove(&name.clone());

                let cranelift_externals = collect_cranelift_external_modules(&ast);
                let mut module = Module {
                    dependencies,
                    origin,
                    extra,
                    mtime,
                    name,
                    code,
                    ast,
                    input_path: path,
                    cranelift_externals,
                };
                module.attach_doc_and_module_comments();

                // Register the types from this module so they can be imported into
                // other modules.
                let _ = module_types.insert(module.name.clone(), module.ast.type_info.clone());
                // Register the successfully type checked module data so that it can be
                // used for code generation and in the language server.
                modules.push(module);
            }

            Outcome::PartialFailure(ast, errors) => {
                let error = Error::Type {
                    names: Box::new(ast.names.clone()),
                    path: path.clone(),
                    src: code.clone(),
                    errors,
                };
                // Mark as incomplete so that this module isn't reloaded from cache.
                let _ = incomplete_modules.insert(name.clone());
                let cranelift_externals = collect_cranelift_external_modules(&ast);
                // Register the partially type checked module data so that it can be
                // used in the language server.
                modules.push(Module {
                    dependencies,
                    origin,
                    extra,
                    mtime,
                    name,
                    code,
                    ast,
                    input_path: path,
                    cranelift_externals,
                });
                // WARNING: This cannot be used for code generation as the code has errors.
                return Outcome::PartialFailure(modules, error);
            }

            Outcome::TotalFailure(errors) => {
                return Outcome::TotalFailure(Error::Type {
                    names: Default::default(),
                    path: path.clone(),
                    src: code.clone(),
                    errors,
                });
            }
        };
    }

    Outcome::Ok(modules)
}

#[derive(Debug)]
pub(crate) enum Input {
    New(UncompiledModule),
    Cached(CachedModule),
}

impl Input {
    pub fn name(&self) -> &EcoString {
        match self {
            Input::New(m) => &m.name,
            Input::Cached(m) => &m.name,
        }
    }

    pub fn source_path(&self) -> &Utf8Path {
        match self {
            Input::New(m) => &m.path,
            Input::Cached(m) => &m.source_path,
        }
    }

    pub fn dependencies(&self) -> Vec<EcoString> {
        match self {
            Input::New(m) => m.dependencies.iter().map(|(n, _)| n.clone()).collect(),
            Input::Cached(m) => m.dependencies.iter().map(|(n, _)| n.clone()).collect(),
        }
    }

    /// Returns `true` if the input is [`New`].
    ///
    /// [`New`]: Input::New
    #[must_use]
    pub(crate) fn is_new(&self) -> bool {
        matches!(self, Self::New(..))
    }

    /// Returns `true` if the input is [`Cached`].
    ///
    /// [`Cached`]: Input::Cached
    #[must_use]
    pub(crate) fn is_cached(&self) -> bool {
        matches!(self, Self::Cached(..))
    }
}

#[derive(Debug)]
pub(crate) struct CachedModule {
    pub name: EcoString,
    pub origin: Origin,
    pub dependencies: Vec<(EcoString, SrcSpan)>,
    pub source_path: Utf8PathBuf,
    pub line_numbers: LineNumbers,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct CacheMetadata {
    pub mtime: SystemTime,
    pub codegen_performed: bool,
    pub dependencies: Vec<(EcoString, SrcSpan)>,
    pub fingerprint: SourceFingerprint,
    pub line_numbers: LineNumbers,
}

impl CacheMetadata {
    pub fn to_binary(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Serializing cache info")
    }

    pub fn from_binary(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Loaded {
    pub to_compile: Vec<UncompiledModule>,
    pub cached: Vec<type_::ModuleInterface>,
}

impl Loaded {
    fn empty() -> Self {
        Self {
            to_compile: vec![],
            cached: vec![],
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct UncompiledModule {
    pub path: Utf8PathBuf,
    pub name: EcoString,
    pub code: EcoString,
    pub mtime: SystemTime,
    pub origin: Origin,
    pub package: EcoString,
    pub dependencies: Vec<(EcoString, SrcSpan)>,
    pub ast: UntypedModule,
    pub extra: ModuleExtra,
}

#[derive(Template)]
#[template(path = "gleam@@main.erl", escape = "none")]
struct ErlangEntrypointModule<'a> {
    application: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub enum CachedWarnings {
    Use,
    Ignore,
}
impl CachedWarnings {
    pub(crate) fn should_use(&self) -> bool {
        match self {
            CachedWarnings::Use => true,
            CachedWarnings::Ignore => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CheckModuleConflicts {
    Check,
    DoNotCheck,
}
impl CheckModuleConflicts {
    pub(crate) fn should_check(&self) -> bool {
        match self {
            CheckModuleConflicts::Check => true,
            CheckModuleConflicts::DoNotCheck => false,
        }
    }
}
