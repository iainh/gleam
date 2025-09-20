//! Cranelift-based native code generation for Gleam.
//!
//! This module will lower typed Gleam modules into Cranelift IR and emit native
//! object files. The initial implementation is a scaffold to be filled in as the
//! backend evolves.

use crate::{
    Result,
    ast::{
        BinOp, ClauseGuard, Constant, Function, Pattern, Publicity, Statement, TypedArg,
        TypedClauseGuard, TypedConstant, TypedDefinition, TypedExpr, TypedStatement,
    },
    build::Module as GleamModule,
    io::FileSystemWriter,
    line_numbers::LineNumbers,
    type_::{ModuleValueConstructor, PatternConstructor, Type, ValueConstructorVariant},
};
use camino::Utf8Path;
use cranelift_codegen::{
    ir::{
        self, InstBuilder, MemFlags, StackSlotData, StackSlotKind, TrapCode, Value,
        condcodes::{FloatCC, IntCC},
    },
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use ecow::EcoString;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::{
    collections::{BTreeMap, HashMap},
    convert::TryFrom,
    sync::Arc,
};
use tracing::instrument;

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

type FunctionIdMap = HashMap<(EcoString, usize), FuncId>;

const VALUE_TAG_MASK: i64 = 0b11;
const HEADER_FIELD_MASK: i64 = 0xFFFF;
const HEADER_ARITY_SHIFT: i64 = 16;
const HEADER_SIZE: i32 = 8;
const TAG_RECORD: i64 = 6;
const TAG_BOOLEAN: i64 = 12;
const BOOLEAN_FALSE_ARITY: i64 = 0;
const BOOLEAN_TRUE_ARITY: i64 = 1;

fn function_symbol_name(module: &str, name: &EcoString, arity: usize) -> String {
    format!("gleam${}_{}__{}", module.replace('/', "$"), name, arity)
}

fn collect_module_functions(
    module: &crate::ast::TypedModule,
) -> Vec<&Function<Arc<Type>, TypedExpr>> {
    module
        .definitions
        .iter()
        .filter_map(|definition| match definition {
            TypedDefinition::Function(function) if function.name.is_some() => Some(function),
            _ => None,
        })
        .collect()
}

fn declare_module_functions(
    module: &mut ObjectModule,
    module_name: &EcoString,
    functions: &[&Function<Arc<Type>, TypedExpr>],
) -> Result<FunctionIdMap> {
    let mut ids = FunctionIdMap::with_capacity(functions.len());
    let pointer_type = module.target_config().pointer_type();

    for function in functions {
        let Some((_, name)) = &function.name else {
            continue;
        };
        let arity = function.arguments.len();
        let symbol = function_symbol_name(module_name, name, arity);

        let mut signature = module.make_signature();
        for _ in 0..arity {
            signature.params.push(ir::AbiParam::new(pointer_type));
        }
        signature.returns.push(ir::AbiParam::new(pointer_type));

        let func_id = module
            .declare_function(&symbol, Linkage::Local, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;

        let key = (name.clone(), arity);
        let _ = ids.insert(key, func_id);
    }

    Ok(ids)
}

fn lower_module_functions(
    module: &mut ObjectModule,
    config: &ModuleConfig<'_>,
) -> Result<Option<FuncId>> {
    let functions = collect_module_functions(&config.module.ast);
    if functions.is_empty() {
        return Ok(None);
    }

    let function_ids = declare_module_functions(module, &config.module.name, &functions)?;
    let mut zero_arity_records = HashMap::new();
    let mut closure_counter = 0usize;

    for function in functions {
        let Some((_, name)) = &function.name else {
            continue;
        };
        let arity = function.arguments.len();
        let key = (name.clone(), arity);
        let Some(&func_id) = function_ids.get(&key) else {
            continue;
        };
        lower_function(
            module,
            &config.module.name,
            function,
            func_id,
            &function_ids,
            &mut zero_arity_records,
            &mut closure_counter,
        )?;
    }

    let main_key = (EcoString::from("main"), 0);
    Ok(function_ids.get(&main_key).copied())
}

fn lower_function(
    module: &mut ObjectModule,
    module_name: &EcoString,
    function: &Function<Arc<Type>, TypedExpr>,
    func_id: FuncId,
    functions: &FunctionIdMap,
    zero_arity_records: &mut HashMap<(EcoString, u16), DataId>,
    closure_counter: &mut usize,
) -> Result<()> {
    let pointer_type = module.target_config().pointer_type();
    let pointer_bytes = module.target_config().pointer_bytes();

    let mut ctx = module.make_context();
    for _ in &function.arguments {
        ctx.func
            .signature
            .params
            .push(ir::AbiParam::new(pointer_type));
    }
    ctx.func
        .signature
        .returns
        .push(ir::AbiParam::new(pointer_type));

    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
    let block = builder.create_block();
    builder.append_block_params_for_function_params(block);
    builder.switch_to_block(block);
    builder.seal_block(block);
    let block_params: Vec<Value> = builder.block_params(block).to_vec();

    {
        let mut lowering = LoweringContext::new(
            &mut builder,
            pointer_type,
            pointer_bytes,
            functions,
            module_name,
            zero_arity_records,
            closure_counter,
        );

        for (value, arg) in block_params.iter().zip(function.arguments.iter()) {
            if let Some(name) = arg.get_variable_name() {
                lowering.define(name, *value);
            }
        }

        let value = lower_block(module, function.body.as_slice(), &mut lowering)?;
        let _ = lowering.builder.ins().return_(&[value]);
    }

    builder.finalize();

    module
        .define_function(func_id, &mut ctx)
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;
    module.clear_context(&mut ctx);
    Ok(())
}

fn lower_block(
    module: &mut ObjectModule,
    statements: &[TypedStatement],
    ctx: &mut LoweringContext<'_, '_, '_>,
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
                return Err(crate::Error::NativeCodegen {
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
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    match expression {
        TypedExpr::Int { int_value, .. } => {
            let value = int_value
                .to_i64()
                .ok_or_else(|| crate::Error::NativeCodegen {
                    message: "integer literal out of range for 64-bit backend".into(),
                })?;
            let encoded = encode_small_int(value)?;
            Ok(ctx.builder.ins().iconst(ir::types::I64, encoded))
        }

        TypedExpr::String { value, .. } => {
            let data_ptr = ctx.string_constant(module, value.as_str())?;
            let len =
                i64::try_from(value.as_str().len()).map_err(|_| crate::Error::NativeCodegen {
                    message: "string literal too long".into(),
                })?;
            let len_value = ctx.builder.ins().iconst(ctx.pointer_type, len);
            let func_id = ctx.declare_runtime_binary_from_slice(module)?;
            let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[data_ptr, len_value]);
            let results = ctx.builder.inst_results(call);
            Ok(results[0])
        }

        TypedExpr::Var {
            name, constructor, ..
        } => match &constructor.variant {
            ValueConstructorVariant::Record {
                arity,
                module: ctor_module,
                variant_index,
                ..
            } => {
                if constructor.type_.is_bool() {
                    let value = match name.as_str() {
                        "True" => true,
                        "False" => false,
                        other => {
                            return Err(crate::Error::NativeCodegen {
                                message: format!(
                                    "unsupported boolean constructor `{other}` in native functions"
                                ),
                            });
                        }
                    };
                    ctx.bool_constant(module, value)
                } else if *arity == 0 {
                    ctx.zero_arity_record_constant(module, ctor_module, *variant_index)
                } else {
                    Err(crate::Error::NativeCodegen {
                        message: format!(
                            "constructor functions are not yet supported in native functions: `{name}`"
                        ),
                    })
                }
            }
            ValueConstructorVariant::LocalVariable { .. } => {
                ctx.lookup(name)
                    .copied()
                    .ok_or_else(|| crate::Error::NativeCodegen {
                        message: format!("unknown variable `{name}` in native main"),
                    })
            }
            _ => ctx
                .lookup(name)
                .copied()
                .ok_or_else(|| crate::Error::NativeCodegen {
                    message: format!("unknown variable `{name}` in native main"),
                }),
        },

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
            let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
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

        TypedExpr::Fn {
            arguments, body, ..
        } => lower_function_literal(module, arguments, body, ctx),

        TypedExpr::ModuleSelect {
            constructor: ModuleValueConstructor::Record { variant_index, .. },
            module_name,
            ..
        } => ctx.zero_arity_record_constant(module, module_name, *variant_index),

        TypedExpr::BinOp {
            name, left, right, ..
        } => lower_bin_op(module, name, left, right, ctx),

        TypedExpr::Call { fun, arguments, .. } => lower_call(module, fun, arguments, ctx),

        TypedExpr::Case {
            subjects, clauses, ..
        } => lower_case(module, subjects, clauses, ctx),

        _ => Err(crate::Error::NativeCodegen {
            message: format!("unsupported expression in main function: {expression:?}"),
        }),
    }
}

fn lower_assignment(
    module: &mut ObjectModule,
    assignment: &crate::ast::TypedAssignment,
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<()> {
    if assignment.kind.is_assert() {
        return Err(crate::Error::NativeCodegen {
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
        _ => Err(crate::Error::NativeCodegen {
            message: "only simple variable patterns are supported in native main".into(),
        }),
    }
}

fn lower_call(
    module: &mut ObjectModule,
    fun: &TypedExpr,
    arguments: &[crate::ast::CallArg<TypedExpr>],
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    if let TypedExpr::ModuleSelect {
        module_name, label, ..
    } = fun
    {
        if module_name == "gleeunit" && label == "main" && arguments.is_empty() {
            return lower_gleeunit_main(module, ctx);
        }
        if module_name == "gleeunit" && label == "do_main" && arguments.is_empty() {
            return skip_gleeunit_do_main(module, ctx);
        }
        if module_name == "gleam/io" && arguments.len() == 1 {
            let result = match label.as_str() {
                "print" => Some(lower_print_call(
                    module,
                    &arguments[0].value,
                    ctx,
                    false,
                    false,
                )?),
                "println" => Some(lower_print_call(
                    module,
                    &arguments[0].value,
                    ctx,
                    true,
                    false,
                )?),
                "print_error" => Some(lower_print_call(
                    module,
                    &arguments[0].value,
                    ctx,
                    false,
                    true,
                )?),
                "println_error" => Some(lower_print_call(
                    module,
                    &arguments[0].value,
                    ctx,
                    true,
                    true,
                )?),
                _ => None,
            };

            if let Some(value) = result {
                return Ok(value);
            }
        }
    }

    if let TypedExpr::Var { constructor, .. } = fun {
        if let ValueConstructorVariant::ModuleFn {
            module: module_name,
            name,
            ..
        } = &constructor.variant
        {
            if module_name == "gleeunit" && name == "main" && arguments.is_empty() {
                return lower_gleeunit_main(module, ctx);
            }
            if module_name == "gleeunit" && name == "do_main" && arguments.is_empty() {
                return skip_gleeunit_do_main(module, ctx);
            }
        }
    }

    if let Some(value) = try_lower_defined_function(module, fun, arguments, ctx)? {
        return Ok(value);
    }

    let fun_value = lower_expression(module, fun, ctx)?;
    let pointer_bytes = ctx.pointer_bytes();
    let mut argument_values = Vec::with_capacity(arguments.len());
    for argument in arguments {
        argument_values.push(lower_expression(module, &argument.value, ctx)?);
    }

    let args_ptr = if argument_values.is_empty() {
        ctx.builder.ins().iconst(ctx.pointer_type, 0)
    } else {
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (argument_values.len() * pointer_bytes) as u32,
        ));
        for (index, value) in argument_values.iter().enumerate() {
            let offset = (index * pointer_bytes) as i32;
            let _ = ctx.builder.ins().stack_store(*value, slot, offset);
        }
        ctx.builder.ins().stack_addr(ctx.pointer_type, slot, 0)
    };

    let arg_count = ctx
        .builder
        .ins()
        .iconst(ctx.pointer_type, arguments.len() as i64);

    let apply_func = ctx.declare_runtime_apply_closure(module)?;
    let apply_ref = module.declare_func_in_func(apply_func, &mut ctx.builder.func);
    let call = ctx
        .builder
        .ins()
        .call(apply_ref, &[fun_value, args_ptr, arg_count]);
    let results = ctx.builder.inst_results(call);
    Ok(results[0])
}

fn try_lower_defined_function(
    module: &mut ObjectModule,
    fun: &TypedExpr,
    arguments: &[crate::ast::CallArg<TypedExpr>],
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Option<Value>> {
    if let TypedExpr::Var { constructor, .. } = fun {
        if let ValueConstructorVariant::ModuleFn {
            module: function_module,
            name,
            ..
        } = &constructor.variant
        {
            if function_module == ctx.module_name {
                return ctx.try_call_function(module, name, arguments);
            }
        }
    }

    if let TypedExpr::ModuleSelect {
        module_name, label, ..
    } = fun
    {
        if module_name == ctx.module_name {
            return ctx.try_call_function(module, label, arguments);
        }
    }

    Ok(None)
}

fn lower_gleeunit_main(
    module: &mut ObjectModule,
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    let func_id = ctx.declare_runtime_gleeunit_main(module)?;
    let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
    let call = ctx.builder.ins().call(func_ref, &[]);
    let results = ctx.builder.inst_results(call);
    Ok(results[0])
}

fn skip_gleeunit_do_main(
    module: &mut ObjectModule,
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    tracing::warn!("Skipping gleeunit.do_main; test runner not yet supported on Cranelift");
    let func_id = ctx.declare_runtime_gleeunit_do_main(module)?;
    let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
    let call = ctx.builder.ins().call(func_ref, &[]);
    let results = ctx.builder.inst_results(call);
    Ok(results[0])
}

fn lower_print_call(
    module: &mut ObjectModule,
    argument: &TypedExpr,
    ctx: &mut LoweringContext<'_, '_, '_>,
    newline: bool,
    stderr: bool,
) -> Result<Value> {
    let value = lower_expression(module, argument, ctx)?;

    let func_id = match (stderr, newline) {
        (false, false) => ctx.declare_runtime_print(module)?,
        (false, true) => ctx.declare_runtime_println(module)?,
        (true, false) => ctx.declare_runtime_print_error(module)?,
        (true, true) => ctx.declare_runtime_println_error(module)?,
    };

    let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
    let call = ctx.builder.ins().call(func_ref, &[value]);
    let results = ctx.builder.inst_results(call);
    Ok(results[0])
}

fn lower_case(
    module: &mut ObjectModule,
    subjects: &[TypedExpr],
    clauses: &[crate::ast::Clause<TypedExpr, Arc<Type>, EcoString>],
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    if clauses.is_empty() {
        return Err(crate::Error::NativeCodegen {
            message: "case expressions must have at least one clause".into(),
        });
    }

    let subject_count = subjects.len();
    if subject_count == 0 {
        return Err(crate::Error::NativeCodegen {
            message: "case expressions must match on at least one subject".into(),
        });
    }

    for clause in clauses {
        if clause.pattern.len() != subject_count {
            return Err(crate::Error::NativeCodegen {
                message: "native case clauses must supply a pattern for each subject".into(),
            });
        }
    }

    let mut subject_values = Vec::with_capacity(subject_count);
    for subject in subjects {
        subject_values.push(lower_expression(module, subject, ctx)?);
    }

    let exit_block = ctx.builder.create_block();
    let _ = ctx.builder.append_block_param(exit_block, ctx.pointer_type);

    let fallthrough_block = ctx.builder.create_block();
    for _ in 0..subject_count {
        let _ = ctx
            .builder
            .append_block_param(fallthrough_block, ctx.pointer_type);
    }
    let _ = ctx.builder.ins().jump(fallthrough_block, &subject_values);

    let mut fallthrough = Some(fallthrough_block);

    for (index, clause) in clauses.iter().enumerate() {
        let Some(current_block) = fallthrough else {
            break;
        };

        ctx.builder.switch_to_block(current_block);
        let next_block = ctx.create_subject_block(subject_count);

        let mut pattern_block = current_block;
        let mut pattern_subjects = ctx.builder.block_params(current_block).to_vec();
        let mut bindings: Vec<(EcoString, usize)> = Vec::new();

        for (subject_index, pattern) in clause.pattern.iter().enumerate() {
            match pattern {
                Pattern::Discard { .. } => {}
                Pattern::Variable { name, .. } => {
                    bindings.push((name.clone(), subject_index));
                }
                Pattern::Int { int_value, .. } => {
                    let (block, params) = ctx.branch_on_int_pattern(
                        pattern_block,
                        pattern_subjects[subject_index],
                        int_value,
                        next_block,
                        &pattern_subjects,
                        subject_count,
                    )?;
                    pattern_block = block;
                    pattern_subjects = params;
                }
                Pattern::Constructor {
                    constructor,
                    arguments,
                    spread,
                    type_,
                    ..
                } => {
                    if !arguments.is_empty() {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "constructor patterns with arguments are not yet supported in native functions"
                                    .into(),
                        });
                    }

                    if spread.is_some() {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "constructor patterns with spread are not yet supported in native functions"
                                    .into(),
                        });
                    }

                    let constructor = constructor.expect_ref(
                        "pattern constructor must be known during native code generation",
                    );

                    let (block, params) = ctx.branch_on_constructor_pattern(
                        pattern_block,
                        pattern_subjects[subject_index],
                        constructor,
                        type_,
                        next_block,
                        &pattern_subjects,
                        subject_count,
                    )?;
                    pattern_block = block;
                    pattern_subjects = params;
                }
                other => {
                    return Err(crate::Error::NativeCodegen {
                        message: format!(
                            "case pattern `{other:?}` is not yet supported in native functions"
                        ),
                    });
                }
            }
        }

        ctx.builder.switch_to_block(pattern_block);
        let mut final_subjects = ctx.builder.block_params(pattern_block).to_vec();

        if let Some(guard) = &clause.guard {
            ctx.push_scope();
            for (name, index) in &bindings {
                ctx.define(name, final_subjects[*index]);
            }
            let guard_condition = ctx.lower_clause_guard_condition(module, guard)?;
            ctx.pop_scope();

            let guard_success_block = ctx.create_subject_block(subject_count);
            let success_args = final_subjects.clone();
            let failure_args = final_subjects.clone();
            let _ = ctx.builder.ins().brif(
                guard_condition,
                guard_success_block,
                &success_args,
                next_block,
                &failure_args,
            );
            ctx.builder.seal_block(pattern_block);

            pattern_block = guard_success_block;
            pattern_subjects = ctx.builder.block_params(pattern_block).to_vec();
            final_subjects = pattern_subjects.clone();

            ctx.builder.switch_to_block(pattern_block);
        } else {
            final_subjects = pattern_subjects.clone();
        }

        ctx.push_scope();
        for (name, index) in &bindings {
            ctx.define(name, final_subjects[*index]);
        }
        let value = lower_expression(module, &clause.then, ctx)?;
        ctx.pop_scope();
        let _ = ctx.builder.ins().jump(exit_block, &[value]);
        ctx.builder.seal_block(pattern_block);

        fallthrough = Some(next_block);

        if index + 1 == clauses.len() {
            break;
        }
    }

    if let Some(block) = fallthrough {
        ctx.builder.switch_to_block(block);
        ctx.builder.seal_block(block);
        let _ = ctx.builder.ins().trap(TrapCode::User(0));
    }

    ctx.builder.seal_block(exit_block);
    ctx.builder.switch_to_block(exit_block);
    let result = ctx.builder.block_params(exit_block)[0];
    Ok(result)
}

fn lower_bin_op(
    module: &mut ObjectModule,
    op: &BinOp,
    left: &TypedExpr,
    right: &TypedExpr,
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    match op {
        BinOp::Eq | BinOp::NotEq => {
            let left_value = lower_expression(module, left, ctx)?;
            let right_value = lower_expression(module, right, ctx)?;
            let left_type = left.type_();
            let condition = if left_type.is_float() {
                let left_float = ctx.load_float(left_value);
                let right_float = ctx.load_float(right_value);
                let cmp = if matches!(op, BinOp::Eq) {
                    FloatCC::Equal
                } else {
                    FloatCC::NotEqual
                };
                ctx.builder.ins().fcmp(cmp, left_float, right_float)
            } else {
                let cmp = if matches!(op, BinOp::Eq) {
                    IntCC::Equal
                } else {
                    IntCC::NotEqual
                };
                ctx.builder.ins().icmp(cmp, left_value, right_value)
            };
            ctx.bool_from_condition(module, condition)
        }
        _ => Err(crate::Error::NativeCodegen {
            message: format!("binary operator `{op:?}` is not yet supported in native main"),
        }),
    }
}

fn lower_closure_function(
    module: &mut ObjectModule,
    pointer_type: ir::Type,
    pointer_bytes: usize,
    functions: &FunctionIdMap,
    module_name: &EcoString,
    zero_arity_records: &mut HashMap<(EcoString, u16), DataId>,
    closure_counter: &mut usize,
    closure_id: usize,
    capture_names: &[EcoString],
    arguments: &[TypedArg],
    body: &[TypedStatement],
) -> Result<FuncId> {
    let mut signature = module.make_signature();
    signature.params.push(ir::AbiParam::new(pointer_type));
    signature.params.push(ir::AbiParam::new(pointer_type));
    signature.params.push(ir::AbiParam::new(pointer_type));
    signature.returns.push(ir::AbiParam::new(pointer_type));

    let symbol = format!("{}$closure_{}", module_name.replace("/", "$"), closure_id);

    let func_id = module
        .declare_function(&symbol, Linkage::Local, &signature)
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;

    let mut ctx = module.make_context();
    ctx.func.signature = signature;

    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
    let block = builder.create_block();
    builder.append_block_params_for_function_params(block);
    builder.switch_to_block(block);
    builder.seal_block(block);

    let env_ptr = builder.block_params(block)[0];
    let args_ptr = builder.block_params(block)[1];
    let _arg_count = builder.block_params(block)[2];

    {
        let mut lowering = LoweringContext::new(
            &mut builder,
            pointer_type,
            pointer_bytes as u8,
            functions,
            module_name,
            zero_arity_records,
            closure_counter,
        );

        let mem_flags = MemFlags::trusted();

        for (index, name) in capture_names.iter().enumerate() {
            let offset = (index * pointer_bytes) as i32;
            let value = lowering
                .builder
                .ins()
                .load(pointer_type, mem_flags, env_ptr, offset);
            lowering.define(name, value);
        }

        for (index, argument) in arguments.iter().enumerate() {
            let offset = (index * pointer_bytes) as i32;
            let value = lowering
                .builder
                .ins()
                .load(pointer_type, mem_flags, args_ptr, offset);
            if let Some(name) = argument.get_variable_name() {
                lowering.define(name, value);
            }
        }

        let value = lower_block(module, body, &mut lowering)?;
        let _ = lowering.builder.ins().return_(&[value]);
    }

    builder.finalize();

    module
        .define_function(func_id, &mut ctx)
        .map_err(|err| crate::Error::NativeCodegen {
            message: err.to_string(),
        })?;
    module.clear_context(&mut ctx);

    Ok(func_id)
}

fn lower_function_literal(
    module: &mut ObjectModule,
    arguments: &[TypedArg],
    body: &[TypedStatement],
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    let captures = ctx.capture_environment_values();
    let capture_names: Vec<EcoString> = captures.iter().map(|(name, _)| name.clone()).collect();
    let closure_id = ctx.next_closure_id();

    let pointer_type = ctx.pointer_type;
    let pointer_bytes = ctx.pointer_bytes();
    let module_name = ctx.module_name.clone();
    let functions = ctx.functions;
    let closure_func_id = {
        let zero_arity_records = &mut *ctx.zero_arity_records;
        let closure_counter_ref = &mut *ctx.closure_counter;
        lower_closure_function(
            module,
            pointer_type,
            pointer_bytes,
            functions,
            &module_name,
            zero_arity_records,
            closure_counter_ref,
            closure_id,
            &capture_names,
            arguments,
            body,
        )?
    };

    let func_ref = module.declare_func_in_func(closure_func_id, &mut ctx.builder.func);
    let code_ptr = ctx.builder.ins().func_addr(ctx.pointer_type, func_ref);

    let env_ptr = if captures.is_empty() {
        ctx.builder.ins().iconst(ctx.pointer_type, 0)
    } else {
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (captures.len() * pointer_bytes) as u32,
        ));
        for (index, (_, value)) in captures.iter().enumerate() {
            let offset = (index * pointer_bytes) as i32;
            let _ = ctx.builder.ins().stack_store(*value, slot, offset);
        }
        ctx.builder.ins().stack_addr(ctx.pointer_type, slot, 0)
    };

    let env_len = ctx
        .builder
        .ins()
        .iconst(ctx.pointer_type, captures.len() as i64);

    let alloc_func = ctx.declare_runtime_alloc_closure(module)?;
    let alloc_ref = module.declare_func_in_func(alloc_func, &mut ctx.builder.func);
    let call = ctx
        .builder
        .ins()
        .call(alloc_ref, &[code_ptr, env_ptr, env_len]);
    let results = ctx.builder.inst_results(call);
    Ok(results[0])
}

struct LoweringContext<'a, 'b, 'c> {
    builder: &'a mut FunctionBuilder<'b>,
    pointer_type: ir::Type,
    scopes: Vec<HashMap<EcoString, Value>>,
    string_data: HashMap<EcoString, DataId>,
    zero_arity_records: &'c mut HashMap<(EcoString, u16), DataId>,
    closure_counter: &'c mut usize,
    runtime_nil: Option<FuncId>,
    runtime_alloc_tuple: Option<FuncId>,
    runtime_binary_from_slice: Option<FuncId>,
    runtime_print: Option<FuncId>,
    runtime_println: Option<FuncId>,
    runtime_print_error: Option<FuncId>,
    runtime_println_error: Option<FuncId>,
    runtime_bool_true: Option<FuncId>,
    runtime_bool_false: Option<FuncId>,
    runtime_alloc_closure: Option<FuncId>,
    runtime_apply_closure: Option<FuncId>,
    runtime_gleeunit_main: Option<FuncId>,
    runtime_gleeunit_do_main: Option<FuncId>,
    pointer_bytes: u8,
    functions: &'a FunctionIdMap,
    module_name: &'a EcoString,
}

impl<'a, 'b, 'c> LoweringContext<'a, 'b, 'c> {
    fn new(
        builder: &'a mut FunctionBuilder<'b>,
        pointer_type: ir::Type,
        pointer_bytes: u8,
        functions: &'a FunctionIdMap,
        module_name: &'a EcoString,
        zero_arity_records: &'c mut HashMap<(EcoString, u16), DataId>,
        closure_counter: &'c mut usize,
    ) -> Self {
        Self {
            builder,
            pointer_type,
            scopes: vec![HashMap::new()],
            string_data: HashMap::new(),
            zero_arity_records,
            closure_counter,
            runtime_nil: None,
            runtime_alloc_tuple: None,
            runtime_binary_from_slice: None,
            runtime_print: None,
            runtime_println: None,
            runtime_print_error: None,
            runtime_println_error: None,
            runtime_bool_true: None,
            runtime_bool_false: None,
            runtime_alloc_closure: None,
            runtime_apply_closure: None,
            runtime_gleeunit_main: None,
            runtime_gleeunit_do_main: None,
            pointer_bytes,
            functions,
            module_name,
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

    fn capture_environment_values(&self) -> Vec<(EcoString, Value)> {
        let mut map = BTreeMap::new();
        for scope in self.scopes.iter().rev() {
            for (name, value) in scope {
                let _ = map.entry(name.clone()).or_insert(*value);
            }
        }
        map.into_iter().collect()
    }

    fn next_closure_id(&mut self) -> usize {
        let id = *self.closure_counter;
        *self.closure_counter += 1;
        id
    }

    fn load_float(&mut self, value: Value) -> Value {
        let mem_flags = MemFlags::trusted();
        self.builder
            .ins()
            .load(ir::types::F64, mem_flags, value, HEADER_SIZE)
    }

    fn bool_from_condition(
        &mut self,
        module: &mut ObjectModule,
        condition: Value,
    ) -> Result<Value> {
        let true_value = self.bool_constant(module, true)?;
        let false_value = self.bool_constant(module, false)?;
        let result = self
            .builder
            .ins()
            .select(condition, true_value, false_value);
        Ok(result)
    }

    fn try_call_function(
        &mut self,
        module: &mut ObjectModule,
        name: &EcoString,
        arguments: &[crate::ast::CallArg<TypedExpr>],
    ) -> Result<Option<Value>> {
        let key = (name.clone(), arguments.len());
        let Some(&func_id) = self.functions.get(&key) else {
            return Ok(None);
        };

        let mut args = Vec::with_capacity(arguments.len());
        for argument in arguments {
            args.push(lower_expression(module, &argument.value, self)?);
        }

        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let call = self.builder.ins().call(func_ref, &args);
        let results = self.builder.inst_results(call);
        Ok(Some(results[0]))
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
                .map_err(|err| crate::Error::NativeCodegen {
                    message: err.to_string(),
                })?;
            module
                .define_data(id, &description)
                .map_err(|err| crate::Error::NativeCodegen {
                    message: err.to_string(),
                })?;
            let _ = self.string_data.insert(key.clone(), id);
            id
        };

        let gv = module.declare_data_in_func(data_id, &mut self.builder.func);
        Ok(self.builder.ins().global_value(self.pointer_type, gv))
    }

    fn declare_runtime_nil(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_nil {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_list_nil", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
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
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_alloc_tuple = Some(id);
        Ok(id)
    }

    fn declare_runtime_binary_from_slice(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_binary_from_slice {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_binary_from_slice", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_binary_from_slice = Some(id);
        Ok(id)
    }

    fn declare_runtime_print(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_print {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("io_print", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_print = Some(id);
        Ok(id)
    }

    fn declare_runtime_println(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_println {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("io_println", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_println = Some(id);
        Ok(id)
    }

    fn declare_runtime_print_error(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_print_error {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("io_print_error", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_print_error = Some(id);
        Ok(id)
    }

    fn declare_runtime_println_error(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_println_error {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("io_println_error", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_println_error = Some(id);
        Ok(id)
    }

    fn declare_runtime_bool_true(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_bool_true {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_bool_true", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bool_true = Some(id);
        Ok(id)
    }

    fn declare_runtime_bool_false(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_bool_false {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_bool_false", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bool_false = Some(id);
        Ok(id)
    }

    fn declare_runtime_alloc_closure(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_alloc_closure {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("gleam_alloc_closure", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_alloc_closure = Some(id);
        Ok(id)
    }

    fn declare_runtime_apply_closure(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_apply_closure {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("gleam_apply_closure", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_apply_closure = Some(id);
        Ok(id)
    }

    fn bool_constant(&mut self, module: &mut ObjectModule, value: bool) -> Result<Value> {
        let func_id = if value {
            self.declare_runtime_bool_true(module)?
        } else {
            self.declare_runtime_bool_false(module)?
        };
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let call = self.builder.ins().call(func_ref, &[]);
        let results = self.builder.inst_results(call);
        Ok(results[0])
    }

    fn zero_arity_record_constant(
        &mut self,
        module: &mut ObjectModule,
        module_name: &EcoString,
        variant_index: u16,
    ) -> Result<Value> {
        let key = (module_name.clone(), variant_index);
        let data_id = if let Some(id) = self.zero_arity_records.get(&key) {
            *id
        } else {
            let header = ((1u64) << 32) | (TAG_RECORD as u64);
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&header.to_le_bytes());
            bytes.extend_from_slice(&u32::from(variant_index).to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());

            let mut description = DataDescription::new();
            description.define(bytes.into_boxed_slice());

            let name = format!(
                "gleam$record0_{}_{}",
                module_name.replace("/", "$"),
                variant_index
            );
            let id = module
                .declare_data(&name, Linkage::Local, false, false)
                .map_err(|err| crate::Error::NativeCodegen {
                    message: err.to_string(),
                })?;
            module
                .define_data(id, &description)
                .map_err(|err| crate::Error::NativeCodegen {
                    message: err.to_string(),
                })?;
            let _ = self.zero_arity_records.insert(key.clone(), id);
            id
        };

        let gv = module.declare_data_in_func(data_id, &mut self.builder.func);
        Ok(self.builder.ins().global_value(self.pointer_type, gv))
    }

    fn create_subject_block(&mut self, subject_count: usize) -> ir::Block {
        let block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self.builder.append_block_param(block, self.pointer_type);
        }
        block
    }

    fn branch_on_int_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        int_value: &BigInt,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>)> {
        let success_block = self.create_subject_block(subject_count);

        self.builder.switch_to_block(current_block);
        let int = int_value
            .to_i64()
            .ok_or_else(|| crate::Error::NativeCodegen {
                message: format!("integer literal out of range for Gleam immediate: {int_value}"),
            })?;
        let encoded = encode_small_int(int)?;
        let literal = self.builder.ins().iconst(self.pointer_type, encoded);
        let cmp = self.builder.ins().icmp(IntCC::Equal, subject, literal);

        let args = failure_args.to_vec();
        let _ = self
            .builder
            .ins()
            .brif(cmp, success_block, &args, failure_block, &args);
        self.builder.seal_block(current_block);

        let params = self.builder.block_params(success_block).to_vec();
        Ok((success_block, params))
    }

    fn lower_clause_guard_condition(
        &mut self,
        module: &mut ObjectModule,
        guard: &TypedClauseGuard,
    ) -> Result<Value> {
        match guard {
            ClauseGuard::Block { value, .. } => self.lower_clause_guard_condition(module, value),
            ClauseGuard::Equals { left, right, .. } => {
                let left = self.lower_clause_guard_operand(module, left)?;
                let right = self.lower_clause_guard_operand(module, right)?;
                Ok(self.builder.ins().icmp(IntCC::Equal, left, right))
            }
            ClauseGuard::NotEquals { left, right, .. } => {
                let left = self.lower_clause_guard_operand(module, left)?;
                let right = self.lower_clause_guard_operand(module, right)?;
                Ok(self.builder.ins().icmp(IntCC::NotEqual, left, right))
            }
            ClauseGuard::Var { .. } | ClauseGuard::Constant(_) => {
                let value = self.lower_clause_guard_operand(module, guard)?;
                let true_value = self.bool_constant(module, true)?;
                Ok(self.builder.ins().icmp(IntCC::Equal, value, true_value))
            }
            other => Err(crate::Error::NativeCodegen {
                message: format!("guard `{other:?}` is not yet supported in native functions"),
            }),
        }
    }

    fn lower_clause_guard_operand(
        &mut self,
        module: &mut ObjectModule,
        guard: &TypedClauseGuard,
    ) -> Result<Value> {
        match guard {
            ClauseGuard::Var { name, .. } => {
                self.lookup(name)
                    .copied()
                    .ok_or_else(|| crate::Error::NativeCodegen {
                        message: format!(
                            "unknown guard variable `{name}` in native case expression"
                        ),
                    })
            }
            ClauseGuard::Constant(constant) => self.lower_clause_guard_constant(module, constant),
            ClauseGuard::Block { value, .. } => self.lower_clause_guard_operand(module, value),
            other => Err(crate::Error::NativeCodegen {
                message: format!(
                    "guard expression `{other:?}` is not yet supported in native functions"
                ),
            }),
        }
    }

    fn lower_clause_guard_constant(
        &mut self,
        module: &mut ObjectModule,
        constant: &TypedConstant,
    ) -> Result<Value> {
        match constant {
            Constant::Record { name, type_, .. } if type_.is_bool() && name == "True" => {
                self.bool_constant(module, true)
            }
            Constant::Record { name, type_, .. } if type_.is_bool() && name == "False" => {
                self.bool_constant(module, false)
            }
            Constant::Var { name, type_, .. } if type_.is_bool() && name == "True" => {
                self.bool_constant(module, true)
            }
            Constant::Var { name, type_, .. } if type_.is_bool() && name == "False" => {
                self.bool_constant(module, false)
            }
            other => Err(crate::Error::NativeCodegen {
                message: format!(
                    "guard constant `{other:?}` is not yet supported in native functions"
                ),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn branch_on_constructor_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        constructor: &PatternConstructor,
        type_: &Arc<Type>,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>)> {
        let success_block = self.create_subject_block(subject_count);
        let args = failure_args.to_vec();
        let mem_flags = MemFlags::trusted();

        self.builder.switch_to_block(current_block);
        let pointer_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(pointer_block, self.pointer_type);
        let tag_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(tag_block, self.pointer_type);
        let _ = self
            .builder
            .append_block_param(tag_block, self.pointer_type);

        let value_tag_mask = self.builder.ins().iconst(self.pointer_type, VALUE_TAG_MASK);
        let boxed_check = self.builder.ins().band(subject, value_tag_mask);
        let zero = self.builder.ins().iconst(self.pointer_type, 0);
        let is_boxed = self.builder.ins().icmp(IntCC::Equal, boxed_check, zero);

        let _ = self
            .builder
            .ins()
            .brif(is_boxed, pointer_block, &[subject], failure_block, &args);
        self.builder.seal_block(current_block);

        self.builder.switch_to_block(pointer_block);
        let pointer_subject = self.builder.block_params(pointer_block)[0];
        let header = self
            .builder
            .ins()
            .load(self.pointer_type, mem_flags, pointer_subject, 0);
        let _ = self
            .builder
            .ins()
            .jump(tag_block, &[pointer_subject, header]);
        self.builder.seal_block(pointer_block);

        self.builder.switch_to_block(tag_block);
        let tag_subject = self.builder.block_params(tag_block)[0];
        let tag_header = self.builder.block_params(tag_block)[1];
        let header_mask = self
            .builder
            .ins()
            .iconst(self.pointer_type, HEADER_FIELD_MASK);

        if type_.is_bool() {
            let expected_arity = match constructor.name.as_str() {
                "True" => BOOLEAN_TRUE_ARITY,
                "False" => BOOLEAN_FALSE_ARITY,
                other => {
                    return Err(crate::Error::NativeCodegen {
                        message: format!(
                            "unsupported boolean constructor `{other}` in native functions"
                        ),
                    });
                }
            };

            let boolean_tag = self.builder.ins().iconst(self.pointer_type, TAG_BOOLEAN);
            let header_tag = self.builder.ins().band(tag_header, header_mask);
            let tag_matches = self
                .builder
                .ins()
                .icmp(IntCC::Equal, header_tag, boolean_tag);

            let arity_shifted = self.builder.ins().ushr_imm(tag_header, HEADER_ARITY_SHIFT);
            let arity = self.builder.ins().band(arity_shifted, header_mask);
            let expected = self.builder.ins().iconst(self.pointer_type, expected_arity);
            let arity_matches = self.builder.ins().icmp(IntCC::Equal, arity, expected);

            let both_match = self.builder.ins().band(tag_matches, arity_matches);
            let _ = self
                .builder
                .ins()
                .brif(both_match, success_block, &args, failure_block, &args);
            self.builder.seal_block(tag_block);
        } else {
            let record_tag = self.builder.ins().iconst(self.pointer_type, TAG_RECORD);
            let header_tag = self.builder.ins().band(tag_header, header_mask);
            let tag_matches = self
                .builder
                .ins()
                .icmp(IntCC::Equal, header_tag, record_tag);

            let record_block = self.builder.create_block();
            let _ = self
                .builder
                .append_block_param(record_block, self.pointer_type);
            let _ = self.builder.ins().brif(
                tag_matches,
                record_block,
                &[tag_subject],
                failure_block,
                &args,
            );
            self.builder.seal_block(tag_block);

            self.builder.switch_to_block(record_block);
            let record_subject = self.builder.block_params(record_block)[0];
            let constructor_offset = self.pointer_bytes() as i32;
            let ctor_index = self.builder.ins().load(
                ir::types::I32,
                mem_flags,
                record_subject,
                constructor_offset,
            );
            let ctor_index = self.builder.ins().uextend(self.pointer_type, ctor_index);
            let expected = self
                .builder
                .ins()
                .iconst(self.pointer_type, i64::from(constructor.constructor_index));
            let index_matches = self.builder.ins().icmp(IntCC::Equal, ctor_index, expected);

            let _ =
                self.builder
                    .ins()
                    .brif(index_matches, success_block, &args, failure_block, &args);
            self.builder.seal_block(record_block);
        }

        let params = self.builder.block_params(success_block).to_vec();
        Ok((success_block, params))
    }

    fn declare_runtime_gleeunit_main(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_gleeunit_main {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleeunit_main", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_gleeunit_main = Some(id);
        Ok(id)
    }

    fn declare_runtime_gleeunit_do_main(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_gleeunit_do_main {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleeunit_do_main", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_gleeunit_do_main = Some(id);
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
        return Err(crate::Error::NativeCodegen {
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
