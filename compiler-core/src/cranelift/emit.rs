use crate::{Result, io::FileSystemWriter};
use camino::Utf8Path;
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::{
    ir,
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use tracing::instrument;

use super::{config::ModuleConfig, lowering::lower_module_functions};

#[instrument(skip_all, fields(module = %config.module.name, output = %output_path))]
pub fn emit_object(
    writer: &impl FileSystemWriter,
    config: ModuleConfig<'_>,
    output_path: &Utf8Path,
) -> Result<()> {
    let isa_builder = cranelift_native::builder().map_err(|err| crate::Error::NativeCodegen {
        message: err.to_string(),
    })?;

    let mut flag_builder = settings::builder();
    flag_builder
        .set("is_pic", "true")
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;
    let flags = settings::Flags::new(flag_builder);

    let isa = isa_builder
        .finish(flags)
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;

    let object_builder = ObjectBuilder::new(
        isa,
        format!("gleam_{}", config.module.name.replace("/", "_")),
        cranelift_module::default_libcall_names(),
    )
    .map_err(|err| crate::Error::NativeCodegen {
        message: err.to_string(),
    })?;

    let mut module = ObjectModule::new(object_builder);

    let main_func = lower_module_functions(&mut module, &config)?;

    if config.has_entrypoint {
        let Some(main_func) = main_func else {
            return Err(crate::Error::NativeCodegen {
                message: format!("module `{}` is missing public main/0", config.module.name),
            });
        };
        build_entrypoint(&mut module, main_func)?;
    }

    let product = module.finish();
    let bytes = product.emit().map_err(|err| crate::Error::NativeCodegen {
        message: err.to_string(),
    })?;

    writer.write_bytes(output_path, &bytes)?;
    Ok(())
}

fn build_entrypoint(module: &mut ObjectModule, main_func: FuncId) -> Result<()> {
    let mut ctx = module.make_context();
    ctx.func
        .signature
        .returns
        .push(ir::AbiParam::new(ir::types::I32));

    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
    let block = builder.create_block();
    builder.switch_to_block(block);
    builder.seal_block(block);

    let init_signature = module.make_signature();
    let runtime_init = module
        .declare_function("gleam_runtime_init", Linkage::Import, &init_signature)
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;
    let runtime_init_ref = module.declare_func_in_func(runtime_init, &mut builder.func);
    let _ = builder.ins().call(runtime_init_ref, &[]);

    let main_ref = module.declare_func_in_func(main_func, &mut builder.func);
    let _ = builder.ins().call(main_ref, &[]);

    let zero = builder.ins().iconst(ir::types::I32, 0);
    let _ = builder.ins().return_(&[zero]);
    builder.finalize();

    let func_id = module
        .declare_function("main", Linkage::Export, &ctx.func.signature)
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;

    module
        .define_function(func_id, &mut ctx)
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;
    module.clear_context(&mut ctx);
    Ok(())
}
