use crate::{
    ast::{Function, Publicity, TypedDefinition},
    build::Module as GleamModule,
    line_numbers::LineNumbers,
};
use camino::Utf8Path;

#[derive(Debug)]
pub struct ModuleConfig<'a> {
    pub module: &'a GleamModule,
    pub line_numbers: LineNumbers,
    pub project_root: &'a Utf8Path,
    pub has_entrypoint: bool,
}

impl<'a> ModuleConfig<'a> {
    pub fn with_entrypoint(
        module: &'a GleamModule,
        project_root: &'a Utf8Path,
        has_entrypoint: bool,
    ) -> Self {
        Self {
            line_numbers: LineNumbers::new(&module.code),
            module,
            project_root,
            has_entrypoint,
        }
    }

    pub fn new(module: &'a GleamModule, project_root: &'a Utf8Path) -> Self {
        Self::with_entrypoint(
            module,
            project_root,
            module_contains_public_main(&module.ast),
        )
    }
}

pub(crate) fn module_contains_public_main(module: &crate::ast::TypedModule) -> bool {
    module
        .definitions
        .iter()
        .any(|definition| match definition {
            TypedDefinition::Function(Function {
                name: Some((_, name)),
                publicity: Publicity::Public,
                arguments,
                ..
            }) if name == "main" && arguments.is_empty() => true,
            _ => false,
        })
}
