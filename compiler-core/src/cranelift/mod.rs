//! Cranelift-based native code generation for Gleam.
//!
//! This module will lower typed Gleam modules into Cranelift IR and emit native
//! object files. The initial implementation is a scaffold to be filled in as the
//! backend evolves.

use crate::{
    Result,
    ast::{Function, Pattern, Publicity, Statement, TypedDefinition, TypedExpr, TypedStatement},
    build::Module as GleamModule,
    io::FileSystemWriter,
    line_numbers::LineNumbers,
    type_::Type,
};
use camino::Utf8Path;
use cranelift_codegen::{
    ir::{self, InstBuilder, Value},
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use ecow::EcoString;
use num_traits::ToPrimitive;
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

#[derive(Debug)]
pub struct ModuleConfig<'a> {
    pub module: &'a GleamModule,
    pub line_numbers: LineNumbers,
    pub project_root: &'a Utf8Path,
    pub has_entrypoint: bool,
}

impl<'a> ModuleConfig<'a> {
    pub fn new(module: &'a GleamModule, project_root: &'a Utf8Path) -> Self {
        Self {
            line_numbers: LineNumbers::new(&module.code),
            module,
            project_root,
            has_entrypoint: module_contains_public_main(&module.ast),
        }
    }
}

fn module_contains_public_main(module: &crate::ast::TypedModule) -> bool {
    module.definitions.iter().any(|definition| match definition {
        TypedDefinition::Function(Function {
            name: Some((_, name)),
            publicity: Publicity::Public,
            arguments,
            ..
        }) if name == "main" && arguments.is_empty() => true,
        _ => false,
    })
}

fn lower_main_function(
    module: &mut ObjectModule,
    config: &ModuleConfig<'_>,
) -> Result<FuncId> {
    let main_fn = find_main_function(&config.module.ast).ok_or_else(|| {
        crate::Error::CraneliftCodegen {
            message: format!("module `{}` is missing public main/0", config.module.name),
        }
    })?;

    let mut ctx = module.make_context();
    ctx.func
        .signature
        .returns
        .push(ir::AbiParam::new(ir::types::I64));

    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);

        let mut lowering = LoweringContext::new(&mut builder);
        let value = lower_block(main_fn.body.as_slice(), &mut lowering)?;
        let _ = builder.ins().return_(&[value]);
        builder.finalize();
    }

    let func_id = module
        .declare_function(
            "gleam$main_impl",
            Linkage::Local,
            &ctx.func.signature,
        )
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    module
        .define_function(func_id, &mut ctx)
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    module.clear_context(&mut ctx);
    Ok(func_id)
}

fn find_main_function(module: &crate::ast::TypedModule) -> Option<&Function<Arc<Type>, TypedExpr>> {
    module.definitions.iter().find_map(|definition| match definition {
        TypedDefinition::Function(function)
            if function.publicity == Publicity::Public
                && function.name.as_ref().is_some_and(|(_, name)| name == "main")
                && function.arguments.is_empty() =>
        {
            Some(function)
        }
        _ => None,
    })
}

fn lower_block(
    statements: &[TypedStatement],
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<Value> {
    ctx.push_scope();
    let mut last = ctx.builder.ins().iconst(ir::types::I64, 0);
    for statement in statements {
        match statement {
            Statement::Expression(expr) => {
                last = lower_expression(expr, ctx)?;
            }
            Statement::Assignment(assignment) => {
                lower_assignment(assignment.as_ref(), ctx)?;
            }
            Statement::Use(_) | Statement::Assert(_) => {
                return Err(crate::Error::CraneliftCodegen {
                    message: "`use` and `assert` are not supported in native main yet".into(),
                })
            }
        }
    }
    ctx.pop_scope();
    Ok(last)
}

fn lower_expression(
    expression: &TypedExpr,
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<Value> {
    match expression {
        TypedExpr::Int { int_value, .. } => {
            let value = int_value.to_i64().ok_or_else(|| crate::Error::CraneliftCodegen {
                message: "integer literal out of range for 64-bit backend".into(),
            })?;
            Ok(ctx.builder.ins().iconst(ir::types::I64, value))
        }

        TypedExpr::Var { name, .. } => ctx.lookup(name).copied().ok_or_else(|| {
            crate::Error::CraneliftCodegen {
                message: format!("unknown variable `{name}` in native main"),
            }
        }),

        TypedExpr::Block { statements, .. } => lower_block(statements.as_slice(), ctx),

        TypedExpr::Tuple { elements, .. } if elements.is_empty() => {
            Ok(ctx.builder.ins().iconst(ir::types::I64, 0))
        }

        _ => Err(crate::Error::CraneliftCodegen {
            message: format!(
                "unsupported expression in main function: {expression:?}"
            ),
        }),
    }
}

fn lower_assignment(
    assignment: &crate::ast::TypedAssignment,
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<()> {
    if assignment.kind.is_assert() {
        return Err(crate::Error::CraneliftCodegen {
            message: "`let assert` is not yet supported in native main".into(),
        });
    }

    let value = lower_expression(&assignment.value, ctx)?;

    match &assignment.pattern {
        Pattern::Variable { name, .. } => {
            ctx.define(name, value);
            Ok(())
        }
        Pattern::Discard { .. } => Ok(()),
        _ => Err(crate::Error::CraneliftCodegen {
            message: "only simple variable patterns are supported in native main".into(),
        }),
    }
}

struct LoweringContext<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    scopes: Vec<HashMap<EcoString, Value>>,
}

impl<'a, 'b> LoweringContext<'a, 'b> {
    fn new(builder: &'a mut FunctionBuilder<'b>) -> Self {
        Self {
            builder,
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        let _ = self.scopes.pop();
    }

    fn define(&mut self, name: &EcoString, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            let _ = scope.insert(name.clone(), value);
        }
    }

    fn lookup(&self, name: &EcoString) -> Option<&Value> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
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

    if config.has_entrypoint {
        let main_func = lower_main_function(&mut module, &config)?;
        build_entrypoint(&mut module, main_func)?;
    }

    let product = module.finish();
    let bytes = product
        .emit()
        .map_err(|err| crate::Error::CraneliftCodegen {
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

    let main_ref = module.declare_func_in_func(main_func, &mut builder.func);
    let _ = builder.ins().call(main_ref, &[]);

    let zero = builder.ins().iconst(ir::types::I32, 0);
    let _ = builder.ins().return_(&[zero]);
    builder.finalize();

    let func_id = module
        .declare_function("main", Linkage::Export, &ctx.func.signature)
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    module
        .define_function(func_id, &mut ctx)
        .map_err(|err| crate::Error::CraneliftCodegen {
            message: err.to_string(),
        })?;

    module.clear_context(&mut ctx);
    Ok(())
}
