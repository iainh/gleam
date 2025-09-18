//! Cranelift-based native code generation for Gleam.
//!
//! This module will lower typed Gleam modules into Cranelift IR and emit native
//! object files. The initial implementation is a scaffold to be filled in as the
//! backend evolves.

use crate::{
    Result,
    build::Module,
    line_numbers::LineNumbers,
};
use camino::Utf8Path;
use tracing::instrument;

#[derive(Debug)]
pub struct ModuleConfig<'a> {
    pub module: &'a Module,
    pub line_numbers: LineNumbers,
    pub project_root: &'a Utf8Path,
}

impl<'a> ModuleConfig<'a> {
    pub fn new(module: &'a Module, project_root: &'a Utf8Path) -> Self {
        Self {
            line_numbers: LineNumbers::new(&module.code),
            module,
            project_root,
        }
    }
}

#[instrument(skip_all, fields(module = %config.module.name))]
pub fn emit_object(config: ModuleConfig<'_>) -> Result<()> {
    tracing::warn!(module = %config.module.name, "cranelift_backend_not_yet_implemented");
    let _ = config;
    Ok(())
}
