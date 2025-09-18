//! Cranelift-based native code generation for Gleam.
//!
//! This module will lower typed Gleam modules into Cranelift IR and emit native
//! object files. The initial implementation is a scaffold to be filled in as the
//! backend evolves.

use crate::{
    Result,
    build::Module as GleamModule,
    io::FileSystemWriter,
    line_numbers::LineNumbers,
};
use cranelift_codegen::{
    ir::InstBuilder,
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use cranelift_codegen::ir::{types, AbiParam};
use camino::Utf8Path;
use tracing::instrument;

#[derive(Debug)]
pub struct ModuleConfig<'a> {
    pub module: &'a GleamModule,
    pub line_numbers: LineNumbers,
    pub project_root: &'a Utf8Path,
}

impl<'a> ModuleConfig<'a> {
    pub fn new(module: &'a GleamModule, project_root: &'a Utf8Path) -> Self {
        Self {
            line_numbers: LineNumbers::new(&module.code),
            module,
            project_root,
        }
    }
}

#[instrument(skip_all, fields(module = %config.module.name, output = %output_path))]
pub fn emit_object(
    writer: &impl FileSystemWriter,
    config: ModuleConfig<'_>,
    output_path: &Utf8Path,
) -> Result<()> {
    let isa_builder = cranelift_native::builder()
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    let mut flag_builder = settings::builder();
    flag_builder
        .set("is_pic", "true")
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;
    let flags = settings::Flags::new(flag_builder);

    let isa = isa_builder
        .finish(flags)
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    let object_builder = ObjectBuilder::new(
        isa,
        format!("gleam_{}", config.module.name.replace("/", "_")),
        cranelift_module::default_libcall_names(),
    )
    .map_err(|err| crate::Error::CraneliftCodegen {
        message: err.to_string(),
    })?;

    let mut module = ObjectModule::new(object_builder);
    let mut ctx = module.make_context();
    ctx.func.signature.returns.push(AbiParam::new(types::I64));

    let mut func_builder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_builder_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);
        let zero = builder.ins().iconst(types::I64, 0);
        let _ = builder.ins().return_(&[zero]);
        builder.finalize();
    }

    let func_id = module
        .declare_function("gleam$module_stub", Linkage::Export, &ctx.func.signature)
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    module
        .define_function(func_id, &mut ctx)
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    module.clear_context(&mut ctx);

    let product = module.finish();
    let bytes = product
        .emit()
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    writer.write_bytes(output_path, &bytes)?;
    Ok(())
}
