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
    ir::{self, InstBuilder, StackSlotData, StackSlotKind, Value},
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
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

fn lower_main_function(module: &mut ObjectModule, config: &ModuleConfig<'_>) -> Result<FuncId> {
    let main_fn =
        find_main_function(&config.module.ast).ok_or_else(|| crate::Error::CraneliftCodegen {
            message: format!("module `{}` is missing public main/0", config.module.name),
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

        let mut lowering = LoweringContext::new(
            &mut builder,
            module.target_config().pointer_type(),
            module.target_config().pointer_bytes(),
        );
        let value = lower_block(module, main_fn.body.as_slice(), &mut lowering)?;
        let _ = builder.ins().return_(&[value]);
        builder.finalize();
    }

    let func_id = module
        .declare_function("gleam$main_impl", Linkage::Local, &ctx.func.signature)
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
    module
        .definitions
        .iter()
        .find_map(|definition| match definition {
            TypedDefinition::Function(function)
                if function.publicity == Publicity::Public
                    && function
                        .name
                        .as_ref()
                        .is_some_and(|(_, name)| name == "main")
                    && function.arguments.is_empty() =>
            {
                Some(function)
            }
            _ => None,
        })
}

fn lower_block(
    module: &mut ObjectModule,
    statements: &[TypedStatement],
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<Value> {
    ctx.push_scope();
    let mut last = ctx.builder.ins().iconst(ir::types::I64, 0);
    for statement in statements {
        match statement {
            Statement::Expression(expr) => {
                last = lower_expression(module, expr, ctx)?;
            }
            Statement::Assignment(assignment) => {
                lower_assignment(module, assignment.as_ref(), ctx)?;
            }
            Statement::Use(_) | Statement::Assert(_) => {
                return Err(crate::Error::CraneliftCodegen {
                    message: "`use` and `assert` are not supported in native main yet".into(),
                });
            }
        }
    }
    ctx.pop_scope();
    Ok(last)
}

fn lower_expression(
    module: &mut ObjectModule,
    expression: &TypedExpr,
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<Value> {
    match expression {
        TypedExpr::Int { int_value, .. } => {
            let value = int_value
                .to_i64()
                .ok_or_else(|| crate::Error::CraneliftCodegen {
                    message: "integer literal out of range for 64-bit backend".into(),
                })?;
            let encoded = encode_small_int(value)?;
            Ok(ctx.builder.ins().iconst(ir::types::I64, encoded))
        }

        TypedExpr::Var { name, .. } => {
            ctx.lookup(name)
                .copied()
                .ok_or_else(|| crate::Error::CraneliftCodegen {
                    message: format!("unknown variable `{name}` in native main"),
                })
        }

        TypedExpr::Block { statements, .. } => lower_block(module, statements.as_slice(), ctx),

        TypedExpr::Tuple { elements, .. } => {
            if elements.is_empty() {
                let func_id = ctx.declare_runtime_nil(module)?;
                let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
                let call = ctx.builder.ins().call(func_ref, &[]);
                let results = ctx.builder.inst_results(call);
                return Ok(results[0]);
            }

            let count = elements.len();
            let slot = ctx
                .builder
                .create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (count * ctx.pointer_bytes()) as u32,
                ));

            for (index, element) in elements.iter().enumerate() {
                let value = lower_expression(module, element, ctx)?;
                let offset = (index * ctx.pointer_bytes()) as i32;
                let _ = ctx.builder.ins().stack_store(value, slot, offset);
            }

            let base_ptr = ctx.builder.ins().stack_addr(ctx.pointer_type, slot, 0);
            let len_value = ctx.builder.ins().iconst(ctx.pointer_type, count as i64);

            let func_id = ctx.declare_runtime_alloc_tuple(module)?;
            let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[base_ptr, len_value]);
            let results = ctx.builder.inst_results(call);
            Ok(results[0])
        }

        TypedExpr::Call { fun, arguments, .. } => lower_call(module, fun, arguments, ctx),

        TypedExpr::Case {
            subjects, clauses, ..
        } => lower_case(module, subjects, clauses, ctx),

        _ => Err(crate::Error::CraneliftCodegen {
            message: format!("unsupported expression in main function: {expression:?}"),
        }),
    }
}

fn lower_assignment(
    module: &mut ObjectModule,
    assignment: &crate::ast::TypedAssignment,
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<()> {
    if assignment.kind.is_assert() {
        return Err(crate::Error::CraneliftCodegen {
            message: "`let assert` is not yet supported in native main".into(),
        });
    }

    let value = lower_expression(module, &assignment.value, ctx)?;

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

fn lower_call(
    module: &mut ObjectModule,
    fun: &TypedExpr,
    arguments: &[crate::ast::CallArg<TypedExpr>],
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<Value> {
    if let TypedExpr::ModuleSelect {
        module_name, label, ..
    } = fun
    {
        if module_name == "gleam/io" && label == "println" && arguments.len() == 1 {
            lower_print_call(module, &arguments[0].value, ctx, true)?;
            return Ok(ctx.builder.ins().iconst(ir::types::I64, 0));
        }
        if module_name == "gleam/io" && label == "print" && arguments.len() == 1 {
            lower_print_call(module, &arguments[0].value, ctx, false)?;
            return Ok(ctx.builder.ins().iconst(ir::types::I64, 0));
        }
    }

    Err(crate::Error::CraneliftCodegen {
        message: format!("unsupported call in native main: {fun:?}"),
    })
}

fn lower_print_call(
    module: &mut ObjectModule,
    argument: &TypedExpr,
    ctx: &mut LoweringContext<'_, '_>,
    newline: bool,
) -> Result<()> {
    let text = match argument {
        TypedExpr::String { value, .. } => {
            let mut s = value.as_str().to_string();
            if newline && !s.ends_with('\n') {
                s.push('\n');
            }
            s
        }
        TypedExpr::Int { int_value, .. } => {
            let mut s = int_value
                .to_i64()
                .ok_or_else(|| crate::Error::CraneliftCodegen {
                    message: "integer literal out of range for print".into(),
                })?
                .to_string();
            if newline {
                s.push('\n');
            }
            s
        }
        _ => {
            return Err(crate::Error::CraneliftCodegen {
                message: "println currently supports only string or integer literals".into(),
            });
        }
    };

    let pointer = ctx.string_constant(module, &text)?;
    let puts = ctx.declare_puts(module)?;
    let func_ref = module.declare_func_in_func(puts, &mut ctx.builder.func);
    let _ = ctx.builder.ins().call(func_ref, &[pointer]);
    Ok(())
}

fn lower_case(
    module: &mut ObjectModule,
    subjects: &[TypedExpr],
    clauses: &[crate::ast::Clause<TypedExpr, Arc<Type>, EcoString>],
    ctx: &mut LoweringContext<'_, '_>,
) -> Result<Value> {
    if subjects.len() != 1 || clauses.len() != 1 {
        return Err(crate::Error::CraneliftCodegen {
            message:
                "case expressions in native main currently support only a single subject and clause"
                    .into(),
        });
    }

    let subject = &subjects[0];
    let subject_value = lower_expression(module, subject, ctx)?;

    ctx.push_scope();

    let clause = &clauses[0];
    if clause.pattern.len() != 1 {
        return Err(crate::Error::CraneliftCodegen {
            message: "case clause must have a single pattern".into(),
        });
    }

    let result = match &clause.pattern[0] {
        Pattern::Discard { .. } => {
            if clause.guard.is_some() {
                return Err(crate::Error::CraneliftCodegen {
                    message: "case clause guards are not yet supported in native main".into(),
                });
            }
            lower_expression(module, &clause.then, ctx)
        }
        Pattern::Variable { name, .. } => {
            if clause.guard.is_some() {
                return Err(crate::Error::CraneliftCodegen {
                    message: "case clause guards are not yet supported in native main".into(),
                });
            }
            ctx.define(name, subject_value);
            lower_expression(module, &clause.then, ctx)
        }
        _ => Err(crate::Error::CraneliftCodegen {
            message: "case patterns other than `_` are not yet supported in native main".into(),
        }),
    }?;

    ctx.pop_scope();
    Ok(result)
}

struct LoweringContext<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    pointer_type: ir::Type,
    scopes: Vec<HashMap<EcoString, Value>>,
    string_data: HashMap<EcoString, DataId>,
    puts: Option<FuncId>,
    runtime_nil: Option<FuncId>,
    runtime_alloc_tuple: Option<FuncId>,
    pointer_bytes: u8,
}

impl<'a, 'b> LoweringContext<'a, 'b> {
    fn new(
        builder: &'a mut FunctionBuilder<'b>,
        pointer_type: ir::Type,
        pointer_bytes: u8,
    ) -> Self {
        Self {
            builder,
            pointer_type,
            scopes: vec![HashMap::new()],
            string_data: HashMap::new(),
            puts: None,
            runtime_nil: None,
            runtime_alloc_tuple: None,
            pointer_bytes,
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

    fn string_constant(&mut self, module: &mut ObjectModule, text: &str) -> Result<Value> {
        let key: EcoString = text.into();
        let data_id = if let Some(id) = self.string_data.get(&key) {
            *id
        } else {
            let mut bytes = text.as_bytes().to_vec();
            if !bytes.ends_with(&[0]) {
                bytes.push(0);
            }
            let mut description = DataDescription::new();
            description.define(bytes.into_boxed_slice());
            let name = format!("gleam$str_{}", self.string_data.len());
            let id = module
                .declare_data(&name, Linkage::Local, false, false)
                .map_err(|err| crate::Error::CraneliftCodegen {
                    message: err.to_string(),
                })?;
            module
                .define_data(id, &description)
                .map_err(|err| crate::Error::CraneliftCodegen {
                    message: err.to_string(),
                })?;
            let _ = self.string_data.insert(key.clone(), id);
            id
        };

        let gv = module.declare_data_in_func(data_id, &mut self.builder.func);
        Ok(self.builder.ins().global_value(self.pointer_type, gv))
    }

    fn declare_puts(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.puts {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(ir::types::I32));

        let id = module
            .declare_function("puts", Linkage::Import, &signature)
            .map_err(|err| crate::Error::CraneliftCodegen {
                message: err.to_string(),
            })?;
        self.puts = Some(id);
        Ok(id)
    }

    fn declare_runtime_nil(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_nil {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_list_nil", Linkage::Import, &signature)
            .map_err(|err| crate::Error::CraneliftCodegen {
                message: err.to_string(),
            })?;
        self.runtime_nil = Some(id);
        Ok(id)
    }

    fn declare_runtime_alloc_tuple(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_alloc_tuple {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_alloc_tuple", Linkage::Import, &signature)
            .map_err(|err| crate::Error::CraneliftCodegen {
                message: err.to_string(),
            })?;
        self.runtime_alloc_tuple = Some(id);
        Ok(id)
    }

    fn pointer_bytes(&self) -> usize {
        self.pointer_bytes as usize
    }
}

fn encode_small_int(value: i64) -> Result<i64, crate::Error> {
    const MIN_I63: i64 = -(1i64 << 61);
    const MAX_I63: i64 = (1i64 << 61) - 1;

    if value < MIN_I63 || value > MAX_I63 {
        return Err(crate::Error::CraneliftCodegen {
            message: format!("integer literal out of range for Gleam immediate: {value}"),
        });
    }

    let shifted = (value as i128) << 2;
    Ok(((shifted as u128) as i64) | 0b01)
}

#[instrument(skip_all, fields(module = %config.module.name, output = %output_path))]
pub fn emit_object(
    writer: &impl FileSystemWriter,
    config: ModuleConfig<'_>,
    output_path: &Utf8Path,
) -> Result<()> {
    let isa_builder =
        cranelift_native::builder().map_err(|err| crate::Error::CraneliftCodegen {
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

    let init_signature = module.make_signature();
    let runtime_init = module
        .declare_function("gleam_runtime_init", Linkage::Import, &init_signature)
        .map_err(|err| crate::Error::CraneliftCodegen {
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
