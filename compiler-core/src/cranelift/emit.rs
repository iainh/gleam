use crate::{Result, io::FileSystemWriter};
use camino::Utf8Path;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_object::{ObjectBuilder, ObjectModule};
use tracing::instrument;

use super::{config::ModuleConfig, lowering::lower_module_functions};

#[instrument(skip_all, fields(module = %config.module.name, output = %output_path))]
pub fn emit_object(
    writer: &impl FileSystemWriter,
    config: ModuleConfig<'_>,
    output_path: &Utf8Path,
) -> Result<Option<String>> {
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

    let main_symbol =
        main_func.map(|_| format!("gleam${}_main__0", config.module.name.replace("/", "$")));

    let product = module.finish();
    let bytes = product.emit().map_err(|err| crate::Error::NativeCodegen {
        message: err.to_string(),
    })?;

    writer.write_bytes(output_path, &bytes)?;
    Ok(main_symbol)
}
