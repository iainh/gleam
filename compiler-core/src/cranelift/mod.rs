//! Cranelift-based native code generation for Gleam.
//!
//! This module will lower typed Gleam modules into Cranelift IR and emit native
//! object files. The initial implementation is a scaffold to be filled in as the
//! backend evolves.

use crate::{
    Result,
    ast::{
        BinOp, ClauseGuard, Constant, Function, Pattern, PipelineAssignmentKind, Publicity,
        Statement, TypedArg, TypedClauseGuard, TypedConstant, TypedDefinition, TypedExpr,
        TypedPipelineAssignment, TypedStatement,
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
    mem::size_of,
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
const TAG_FLOAT: u64 = 1;
const TAG_LIST: i64 = 4;
const TAG_TUPLE: i64 = 5;
const TAG_RECORD: i64 = 6;
const TAG_BOOLEAN: i64 = 12;
const BOOLEAN_FALSE_ARITY: i64 = 0;
const BOOLEAN_TRUE_ARITY: i64 = 1;
const FLOAT_HEADER: u64 = (1u64 << 32) | TAG_FLOAT;

fn function_symbol_name(module: &str, name: &EcoString, arity: usize) -> String {
    format!("gleam${}_{}__{}", module.replace('/', "$"), name, arity)
}

fn record_constructor_symbol(
    current_module: &EcoString,
    constructor_module: &EcoString,
    variant_index: u16,
    arity: u16,
) -> String {
    format!(
        "gleam${}_record_ctor_{}__{}_{}",
        current_module.replace("/", "$"),
        constructor_module.replace("/", "$"),
        variant_index,
        arity
    )
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
    let mut float_constants = HashMap::new();
    let mut record_constructors = HashMap::new();
    let mut module_functions = HashMap::new();
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
            &mut float_constants,
            &mut record_constructors,
            &mut module_functions,
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
    float_constants: &mut HashMap<EcoString, DataId>,
    record_constructors: &mut HashMap<(EcoString, u16, u16), FuncId>,
    module_functions: &mut HashMap<(EcoString, EcoString, usize), FuncId>,
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
            float_constants,
            record_constructors,
            module_functions,
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

        TypedExpr::Float { value, .. } => ctx.float_constant(module, value),

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
                    ctx.record_constructor_value(module, ctor_module, *variant_index, *arity)
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

        TypedExpr::NegateInt { value, .. } => {
            let inner = lower_expression(module, value, ctx)?;
            let func_id = ctx.declare_runtime_int_negate(module)?;
            let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[inner]);
            let results = ctx.builder.inst_results(call);
            Ok(results[0])
        }

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
        TypedExpr::List { elements, tail, .. } => {
            let mut values = Vec::with_capacity(elements.len());
            for element in elements {
                values.push(lower_expression(module, element, ctx)?);
            }

            let mut current = if let Some(tail_expr) = tail {
                lower_expression(module, tail_expr, ctx)?
            } else {
                let func_id = ctx.declare_runtime_nil(module)?;
                let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
                let call = ctx.builder.ins().call(func_ref, &[]);
                let results = ctx.builder.inst_results(call);
                results[0]
            };

            if !values.is_empty() {
                let func_id = ctx.declare_runtime_list_cons(module)?;
                let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
                for value in values.into_iter().rev() {
                    let call = ctx.builder.ins().call(func_ref, &[value, current]);
                    let results = ctx.builder.inst_results(call);
                    current = results[0];
                }
            }
            Ok(current)
        }

        TypedExpr::TupleIndex { index, tuple, .. } => {
            let tuple_value = lower_expression(module, tuple, ctx)?;
            ctx.tuple_element(tuple_value, *index)
        }

        TypedExpr::Fn {
            arguments, body, ..
        } => lower_function_literal(module, arguments, body, ctx),

        TypedExpr::ModuleSelect {
            constructor,
            module_name,
            type_,
            ..
        } => match constructor {
            ModuleValueConstructor::Record { variant_index, .. } => {
                ctx.zero_arity_record_constant(module, module_name, *variant_index)
            }
            ModuleValueConstructor::Fn {
                module: function_module,
                name,
                ..
            } => {
                let arity = type_
                    .fn_arity()
                    .ok_or_else(|| crate::Error::NativeCodegen {
                        message: format!(
                            "unable to determine arity for module function `{function_module}.{name}`"
                        ),
                    })?;
                ctx.module_function_value(module, function_module, name, arity)
            }
            ModuleValueConstructor::Constant { .. } => Err(crate::Error::NativeCodegen {
                message: "module constants are not yet supported in native functions".into(),
            }),
        },

        TypedExpr::BinOp {
            name, left, right, ..
        } => lower_bin_op(module, name, left, right, ctx),

        TypedExpr::Pipeline {
            first_value,
            assignments,
            finally,
            ..
        } => lower_pipeline(module, first_value, assignments, finally, ctx),

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

    #[derive(Clone, Copy)]
    enum BindingSource {
        Subject(usize),
        Value(Value),
    }

    for (index, clause) in clauses.iter().enumerate() {
        let Some(current_block) = fallthrough else {
            break;
        };

        ctx.builder.switch_to_block(current_block);
        let next_block = ctx.create_subject_block(subject_count);

        let mut pattern_block = current_block;
        let mut pattern_subjects = ctx.builder.block_params(current_block).to_vec();
        let mut bindings: Vec<(EcoString, BindingSource)> = Vec::new();

        for (subject_index, pattern) in clause.pattern.iter().enumerate() {
            match pattern {
                Pattern::Discard { .. } => {}
                Pattern::Variable { name, .. } => {
                    bindings.push((name.clone(), BindingSource::Subject(subject_index)));
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
                Pattern::Float { value, .. } => {
                    let cleaned = value.replace("_", "");
                    let float_value =
                        cleaned
                            .parse::<f64>()
                            .map_err(|_| crate::Error::NativeCodegen {
                                message: format!(
                                    "invalid float literal `{value}` in native pattern"
                                ),
                            })?;
                    let (block, params) = ctx.branch_on_float_pattern(
                        pattern_block,
                        pattern_subjects[subject_index],
                        float_value,
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

                    let mut capture_flags = Vec::with_capacity(arguments.len());
                    let mut binding_names = Vec::with_capacity(arguments.len());
                    for argument in arguments {
                        if argument.label.is_some() {
                            return Err(crate::Error::NativeCodegen {
                                message: "labelled constructor pattern arguments are not yet supported in native functions"
                                    .into(),
                            });
                        }

                        match &argument.value {
                            Pattern::Variable { name, .. } => {
                                capture_flags.push(true);
                                binding_names.push(Some(name.clone()));
                            }
                            Pattern::Discard { .. } => {
                                capture_flags.push(false);
                                binding_names.push(None);
                            }
                            other => {
                                return Err(crate::Error::NativeCodegen {
                                    message: format!(
                                        "constructor pattern argument `{other:?}` is not yet supported in native functions"
                                    ),
                                });
                            }
                        }
                    }

                    let (block, params, extras) = ctx.branch_on_constructor_pattern(
                        pattern_block,
                        pattern_subjects[subject_index],
                        constructor,
                        type_,
                        &capture_flags,
                        next_block,
                        &pattern_subjects,
                        subject_count,
                    )?;
                    pattern_block = block;
                    pattern_subjects = params;
                    if ctx.builder.current_block() != Some(pattern_block) {
                        ctx.builder.switch_to_block(pattern_block);
                    }

                    let mut extra_iter = extras.into_iter();
                    for (capture, name) in capture_flags.iter().zip(binding_names.iter()) {
                        if *capture {
                            let Some(value) = extra_iter.next() else {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "missing captured constructor argument in native case lowering"
                                            .into(),
                                });
                            };
                            if let Some(name) = name {
                                bindings.push((name.clone(), BindingSource::Value(value)));
                            }
                        }
                    }
                }
                Pattern::List { elements, tail, .. } => {
                    let mut capture_heads = Vec::with_capacity(elements.len());
                    let mut head_names = Vec::with_capacity(elements.len());
                    let mut head_matches: Vec<Option<ListHeadMatch<'_>>> =
                        Vec::with_capacity(elements.len());
                    let mut head_field_bindings: Vec<Option<Vec<Option<EcoString>>>> =
                        Vec::with_capacity(elements.len());

                    for element in elements {
                        match element {
                            Pattern::Variable { name, .. } => {
                                capture_heads.push(true);
                                head_names.push(Some(name.clone()));
                                head_matches.push(None);
                                head_field_bindings.push(None);
                            }
                            Pattern::Discard { .. } => {
                                capture_heads.push(false);
                                head_names.push(None);
                                head_matches.push(None);
                                head_field_bindings.push(None);
                            }
                            Pattern::Constructor {
                                constructor,
                                arguments,
                                spread,
                                type_,
                                ..
                            } if spread.is_none() => {
                                let constructor = constructor.expect_ref(
                                    "pattern constructor must be known during native code generation",
                                );

                                let mut capture_flags = Vec::with_capacity(arguments.len());
                                let mut binding_names = Vec::with_capacity(arguments.len());

                                for argument in arguments {
                                    match &argument.value {
                                        Pattern::Variable { name, .. } => {
                                            capture_flags.push(true);
                                            binding_names.push(Some(name.clone()));
                                        }
                                        Pattern::Discard { .. } => {
                                            capture_flags.push(false);
                                            binding_names.push(None);
                                        }
                                        other => {
                                            return Err(crate::Error::NativeCodegen {
                                                message: format!(
                                                    "constructor list head pattern argument `{other:?}` is not yet supported in native functions"
                                                ),
                                            });
                                        }
                                    }
                                }

                                capture_heads.push(false);
                                head_names.push(None);
                                head_matches.push(Some(ListHeadMatch::Constructor(
                                    ListConstructorInfo {
                                        constructor,
                                        type_,
                                        capture_flags,
                                    },
                                )));
                                head_field_bindings.push(Some(binding_names));
                            }
                            Pattern::Tuple { elements, .. } => {
                                let mut capture_flags = Vec::with_capacity(elements.len());
                                let mut binding_names = Vec::with_capacity(elements.len());

                                for element in elements {
                                    match element {
                                        Pattern::Variable { name, .. } => {
                                            capture_flags.push(true);
                                            binding_names.push(Some(name.clone()));
                                        }
                                        Pattern::Discard { .. } => {
                                            capture_flags.push(false);
                                            binding_names.push(None);
                                        }
                                        other => {
                                            return Err(crate::Error::NativeCodegen {
                                                message: format!(
                                                    "tuple list head element `{other:?}` is not yet supported in native functions"
                                                ),
                                            });
                                        }
                                    }
                                }

                                capture_heads.push(false);
                                head_names.push(None);
                                head_matches.push(Some(ListHeadMatch::Tuple(ListTupleInfo {
                                    arity: elements.len(),
                                    capture_flags,
                                })));
                                head_field_bindings.push(Some(binding_names));
                            }
                            other => {
                                return Err(crate::Error::NativeCodegen {
                                    message: format!(
                                        "list pattern element `{other:?}` is not yet supported in native functions"
                                    ),
                                });
                            }
                        }
                    }

                    let (capture_tail, tail_name) = match tail.as_deref() {
                        None => (false, None),
                        Some(Pattern::Variable { name, .. }) => (true, Some(name.clone())),
                        Some(Pattern::Discard { .. }) => (false, None),
                        Some(other) => {
                            return Err(crate::Error::NativeCodegen {
                                message: format!(
                                    "list tail pattern `{other:?}` is not yet supported in native functions"
                                ),
                            });
                        }
                    };

                    let (block, params, extras) = ctx.branch_on_list_pattern(
                        module,
                        pattern_block,
                        subject_index,
                        &capture_heads,
                        &head_matches,
                        capture_tail,
                        tail.is_none(),
                        next_block,
                        &pattern_subjects,
                        subject_count,
                    )?;
                    pattern_block = block;
                    pattern_subjects = params;

                    let mut extra_iter = extras.into_iter();
                    for (index, ((capture, name), field_bindings)) in capture_heads
                        .iter()
                        .zip(head_names.iter())
                        .zip(head_field_bindings.iter())
                        .enumerate()
                    {
                        if *capture {
                            let Some(value) = extra_iter.next() else {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "missing captured list head value in native case lowering"
                                            .into(),
                                });
                            };
                            if let Some(name) = name {
                                bindings.push((name.clone(), BindingSource::Value(value)));
                            }
                        }

                        if let Some(binding_names) = field_bindings {
                            let Some(info) = head_matches.get(index).and_then(|opt| opt.as_ref())
                            else {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "missing head match info for list bindings in native lowering"
                                            .into(),
                                });
                            };

                            if info.capture_flags().len() != binding_names.len() {
                                return Err(crate::Error::NativeCodegen {
                                    message: "list head binding length mismatch in native list pattern lowering"
                                        .into(),
                                });
                            }

                            for (capture_flag, binding_name) in
                                info.capture_flags().iter().zip(binding_names.iter())
                            {
                                if *capture_flag {
                                    let Some(value) = extra_iter.next() else {
                                        return Err(crate::Error::NativeCodegen {
                                            message:
                                                "missing captured constructor field in native case lowering"
                                                    .into(),
                                        });
                                    };
                                    if let Some(name) = binding_name {
                                        bindings.push((name.clone(), BindingSource::Value(value)));
                                    }
                                }
                            }
                        }
                    }
                    if capture_tail {
                        if let Some(name) = tail_name {
                            let Some(value) = extra_iter.next() else {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "missing captured list tail value in native case lowering"
                                            .into(),
                                });
                            };
                            bindings.push((name, BindingSource::Value(value)));
                        }
                    }
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

        if ctx.builder.current_block() != Some(pattern_block) {
            ctx.builder.switch_to_block(pattern_block);
        }
        let mut final_subjects = ctx.builder.block_params(pattern_block).to_vec();

        if let Some(guard) = &clause.guard {
            ctx.push_scope();
            for (name, source) in &bindings {
                let value = match source {
                    BindingSource::Subject(index) => final_subjects[*index],
                    BindingSource::Value(value) => *value,
                };
                ctx.define(name, value);
            }
            let guard_condition = ctx.lower_clause_guard_condition(module, guard)?;
            ctx.pop_scope();

            let guard_inputs = final_subjects.clone();
            let guard_success_block = ctx.builder.create_block();
            for _ in 0..guard_inputs.len() {
                let _ = ctx
                    .builder
                    .append_block_param(guard_success_block, ctx.pointer_type);
            }
            let success_args = guard_inputs.clone();
            let failure_args = pattern_subjects.clone();
            let _ = ctx.builder.ins().brif(
                guard_condition,
                guard_success_block,
                &success_args,
                next_block,
                &failure_args,
            );
            ctx.builder.seal_block(pattern_block);

            pattern_block = guard_success_block;
            let guard_params = ctx.builder.block_params(pattern_block).to_vec();

            for (_, source) in &mut bindings {
                if let BindingSource::Value(value) = source {
                    let Some(position) = guard_inputs.iter().position(|input| input == value)
                    else {
                        return Err(crate::Error::NativeCodegen {
                            message: "missing captured value in native guard lowering".into(),
                        });
                    };
                    *value = guard_params[position];
                }
            }

            pattern_subjects = guard_params[..subject_count].to_vec();
            final_subjects = pattern_subjects.clone();

            if ctx.builder.current_block() != Some(pattern_block) {
                ctx.builder.switch_to_block(pattern_block);
            }
        } else {
            final_subjects = pattern_subjects.clone();
        }

        ctx.push_scope();
        for (name, source) in &bindings {
            let value = match source {
                BindingSource::Subject(index) => final_subjects[*index],
                BindingSource::Value(value) => *value,
            };
            ctx.define(name, value);
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
        BinOp::LtFloat | BinOp::LtEqFloat | BinOp::GtFloat | BinOp::GtEqFloat => {
            let left_value = lower_expression(module, left, ctx)?;
            let right_value = lower_expression(module, right, ctx)?;
            let left_float = ctx.load_float(left_value);
            let right_float = ctx.load_float(right_value);
            let cmp = match op {
                BinOp::LtFloat => FloatCC::LessThan,
                BinOp::LtEqFloat => FloatCC::LessThanOrEqual,
                BinOp::GtFloat => FloatCC::GreaterThan,
                BinOp::GtEqFloat => FloatCC::GreaterThanOrEqual,
                _ => unreachable!(),
            };
            let condition = ctx.builder.ins().fcmp(cmp, left_float, right_float);
            ctx.bool_from_condition(module, condition)
        }
        BinOp::LtInt | BinOp::LtEqInt | BinOp::GtInt | BinOp::GtEqInt => {
            let left_value = lower_expression(module, left, ctx)?;
            let right_value = lower_expression(module, right, ctx)?;
            let cmp = match op {
                BinOp::LtInt => IntCC::SignedLessThan,
                BinOp::LtEqInt => IntCC::SignedLessThanOrEqual,
                BinOp::GtInt => IntCC::SignedGreaterThan,
                BinOp::GtEqInt => IntCC::SignedGreaterThanOrEqual,
                _ => unreachable!(),
            };
            let condition = ctx.builder.ins().icmp(cmp, left_value, right_value);
            ctx.bool_from_condition(module, condition)
        }
        BinOp::And | BinOp::Or => {
            let left_value = lower_expression(module, left, ctx)?;
            let true_value = ctx.bool_constant(module, true)?;
            let false_value = ctx.bool_constant(module, false)?;
            let condition = ctx.builder.ins().icmp(IntCC::Equal, left_value, true_value);

            let exit_block = ctx.builder.create_block();
            let _ = ctx.builder.append_block_param(exit_block, ctx.pointer_type);
            let right_block = ctx.builder.create_block();

            match op {
                BinOp::And => {
                    let _ = ctx.builder.ins().brif(
                        condition,
                        right_block,
                        &[],
                        exit_block,
                        &[false_value],
                    );
                }
                BinOp::Or => {
                    let _ = ctx.builder.ins().brif(
                        condition,
                        exit_block,
                        &[true_value],
                        right_block,
                        &[],
                    );
                }
                _ => unreachable!(),
            }
            ctx.builder.switch_to_block(right_block);
            let right_value = lower_expression(module, right, ctx)?;
            let _ = ctx.builder.ins().jump(exit_block, &[right_value]);
            ctx.builder.seal_block(right_block);

            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
            let result = ctx.builder.block_params(exit_block)[0];
            Ok(result)
        }
        BinOp::AddFloat | BinOp::SubFloat | BinOp::MultFloat | BinOp::DivFloat => {
            let left_value = lower_expression(module, left, ctx)?;
            let right_value = lower_expression(module, right, ctx)?;
            let left_float = ctx.load_float(left_value);
            let right_float = ctx.load_float(right_value);
            let result = match op {
                BinOp::AddFloat => ctx.builder.ins().fadd(left_float, right_float),
                BinOp::SubFloat => ctx.builder.ins().fsub(left_float, right_float),
                BinOp::MultFloat => ctx.builder.ins().fmul(left_float, right_float),
                BinOp::DivFloat => ctx.builder.ins().fdiv(left_float, right_float),
                _ => unreachable!(),
            };
            let func_id = ctx.declare_runtime_float_from_f64(module)?;
            let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[result]);
            let results = ctx.builder.inst_results(call);
            Ok(results[0])
        }
        BinOp::AddInt | BinOp::SubInt | BinOp::MultInt | BinOp::DivInt | BinOp::RemainderInt => {
            let left_value = lower_expression(module, left, ctx)?;
            let right_value = lower_expression(module, right, ctx)?;

            let left_int = ctx.builder.ins().sshr_imm(left_value, 2);
            let right_int = ctx.builder.ins().sshr_imm(right_value, 2);
            let raw_result = match op {
                BinOp::AddInt => ctx.builder.ins().iadd(left_int, right_int),
                BinOp::SubInt => ctx.builder.ins().isub(left_int, right_int),
                BinOp::MultInt => ctx.builder.ins().imul(left_int, right_int),
                BinOp::DivInt => ctx.builder.ins().sdiv(left_int, right_int),
                BinOp::RemainderInt => ctx.builder.ins().srem(left_int, right_int),
                _ => unreachable!(),
            };
            let shifted = ctx.builder.ins().ishl_imm(raw_result, 2);
            let tag = ctx.builder.ins().iconst(ctx.pointer_type, 1);
            let result = ctx.builder.ins().bor(shifted, tag);
            Ok(result)
        }
        _ => Err(crate::Error::NativeCodegen {
            message: format!("binary operator `{op:?}` is not yet supported in native main"),
        }),
    }
}

fn lower_pipeline(
    module: &mut ObjectModule,
    first_value: &TypedPipelineAssignment,
    assignments: &[(TypedPipelineAssignment, PipelineAssignmentKind)],
    finally: &TypedExpr,
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    ctx.push_scope();
    let result = (|| {
        let mut current = lower_expression(module, &first_value.value, ctx)?;
        ctx.define(&first_value.name, current);

        for (assignment, kind) in assignments {
            current = lower_pipeline_step(module, current, assignment, *kind, ctx)?;
        }

        let binding_name = assignments
            .last()
            .map(|(assignment, _)| &assignment.name)
            .unwrap_or(&first_value.name);
        ctx.define(binding_name, current);

        lower_expression(module, finally, ctx)
    })();
    ctx.pop_scope();
    result
}

fn lower_pipeline_step(
    module: &mut ObjectModule,
    current: Value,
    assignment: &TypedPipelineAssignment,
    kind: PipelineAssignmentKind,
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<Value> {
    if matches!(kind, PipelineAssignmentKind::Echo) {
        return Err(crate::Error::NativeCodegen {
            message: "pipeline `echo` expressions are not yet supported in native functions".into(),
        });
    }

    ctx.define(&assignment.name, current);
    let new_value = lower_expression(module, &assignment.value, ctx)?;
    ctx.define(&assignment.name, new_value);
    Ok(new_value)
}

fn lower_closure_function(
    module: &mut ObjectModule,
    pointer_type: ir::Type,
    pointer_bytes: usize,
    functions: &FunctionIdMap,
    module_name: &EcoString,
    zero_arity_records: &mut HashMap<(EcoString, u16), DataId>,
    float_constants: &mut HashMap<EcoString, DataId>,
    record_constructors: &mut HashMap<(EcoString, u16, u16), FuncId>,
    module_functions: &mut HashMap<(EcoString, EcoString, usize), FuncId>,
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
            float_constants,
            record_constructors,
            module_functions,
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

fn lower_record_constructor_function(
    module: &mut ObjectModule,
    pointer_type: ir::Type,
    module_name: &EcoString,
    constructor_module: &EcoString,
    variant_index: u16,
    arity: u16,
    alloc_record: FuncId,
) -> Result<FuncId> {
    let mut signature = module.make_signature();
    signature.params.push(ir::AbiParam::new(pointer_type));
    signature.params.push(ir::AbiParam::new(pointer_type));
    signature.params.push(ir::AbiParam::new(pointer_type));
    signature.returns.push(ir::AbiParam::new(pointer_type));

    let symbol = record_constructor_symbol(module_name, constructor_module, variant_index, arity);
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

    let args_ptr = builder.block_params(block)[1];
    let argc = builder.block_params(block)[2];

    let ctor_index = builder.ins().iconst(pointer_type, i64::from(variant_index));
    let alloc_ref = module.declare_func_in_func(alloc_record, &mut builder.func);
    let call = builder.ins().call(alloc_ref, &[ctor_index, args_ptr, argc]);
    let result = builder.inst_results(call)[0];
    let _ = builder.ins().return_(&[result]);

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
        let float_constants = &mut *ctx.float_constants;
        let record_constructors = &mut *ctx.record_constructors;
        let module_functions_ref = &mut *ctx.module_functions;
        let closure_counter_ref = &mut *ctx.closure_counter;
        lower_closure_function(
            module,
            pointer_type,
            pointer_bytes,
            functions,
            &module_name,
            zero_arity_records,
            float_constants,
            record_constructors,
            module_functions_ref,
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

#[derive(Debug)]
struct ListConstructorInfo<'a> {
    constructor: &'a PatternConstructor,
    type_: &'a Arc<Type>,
    capture_flags: Vec<bool>,
}

#[derive(Debug)]
struct ListTupleInfo {
    arity: usize,
    capture_flags: Vec<bool>,
}

#[derive(Debug)]
enum ListHeadMatch<'a> {
    Constructor(ListConstructorInfo<'a>),
    Tuple(ListTupleInfo),
}

impl<'a> ListHeadMatch<'a> {
    fn capture_flags(&self) -> &[bool] {
        match self {
            ListHeadMatch::Constructor(info) => &info.capture_flags,
            ListHeadMatch::Tuple(info) => &info.capture_flags,
        }
    }
}

struct LoweringContext<'a, 'b, 'c> {
    builder: &'a mut FunctionBuilder<'b>,
    pointer_type: ir::Type,
    scopes: Vec<HashMap<EcoString, Value>>,
    string_data: HashMap<EcoString, DataId>,
    float_constants: &'c mut HashMap<EcoString, DataId>,
    zero_arity_records: &'c mut HashMap<(EcoString, u16), DataId>,
    record_constructors: &'c mut HashMap<(EcoString, u16, u16), FuncId>,
    module_functions: &'c mut HashMap<(EcoString, EcoString, usize), FuncId>,
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
    runtime_list_cons: Option<FuncId>,
    runtime_int_negate: Option<FuncId>,
    runtime_float_from_f64: Option<FuncId>,
    runtime_alloc_closure: Option<FuncId>,
    runtime_apply_closure: Option<FuncId>,
    runtime_alloc_record: Option<FuncId>,
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
        float_constants: &'c mut HashMap<EcoString, DataId>,
        record_constructors: &'c mut HashMap<(EcoString, u16, u16), FuncId>,
        module_functions: &'c mut HashMap<(EcoString, EcoString, usize), FuncId>,
        closure_counter: &'c mut usize,
    ) -> Self {
        Self {
            builder,
            pointer_type,
            scopes: vec![HashMap::new()],
            string_data: HashMap::new(),
            float_constants,
            zero_arity_records,
            record_constructors,
            module_functions,
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
            runtime_list_cons: None,
            runtime_int_negate: None,
            runtime_float_from_f64: None,
            runtime_alloc_closure: None,
            runtime_apply_closure: None,
            runtime_alloc_record: None,
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

    fn ensure_record_constructor_function(
        &mut self,
        module: &mut ObjectModule,
        constructor_module: &EcoString,
        variant_index: u16,
        arity: u16,
    ) -> Result<FuncId> {
        let key = (constructor_module.clone(), variant_index, arity);
        if let Some(id) = self.record_constructors.get(&key) {
            return Ok(*id);
        }

        let alloc_record = self.declare_runtime_alloc_record(module)?;
        let func_id = lower_record_constructor_function(
            module,
            self.pointer_type,
            self.module_name,
            constructor_module,
            variant_index,
            arity,
            alloc_record,
        )?;
        let _ = self.record_constructors.insert(key, func_id);
        Ok(func_id)
    }

    fn record_constructor_value(
        &mut self,
        module: &mut ObjectModule,
        constructor_module: &EcoString,
        variant_index: u16,
        arity: u16,
    ) -> Result<Value> {
        let func_id = self.ensure_record_constructor_function(
            module,
            constructor_module,
            variant_index,
            arity,
        )?;
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let code_ptr = self.builder.ins().func_addr(self.pointer_type, func_ref);
        let env_ptr = self.builder.ins().iconst(self.pointer_type, 0);
        let env_len = self.builder.ins().iconst(self.pointer_type, 0);
        let alloc_func = self.declare_runtime_alloc_closure(module)?;
        let alloc_ref = module.declare_func_in_func(alloc_func, &mut self.builder.func);
        let call = self
            .builder
            .ins()
            .call(alloc_ref, &[code_ptr, env_ptr, env_len]);
        let results = self.builder.inst_results(call);
        Ok(results[0])
    }

    fn ensure_module_function(
        &mut self,
        module: &mut ObjectModule,
        function_module: &EcoString,
        function_name: &EcoString,
        arity: usize,
    ) -> Result<FuncId> {
        let key = (function_module.clone(), function_name.clone(), arity);
        if let Some(id) = self.module_functions.get(&key) {
            return Ok(*id);
        }

        let symbol = function_symbol_name(function_module, function_name, arity);
        let mut signature = module.make_signature();
        for _ in 0..arity {
            signature.params.push(ir::AbiParam::new(self.pointer_type));
        }
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function(&symbol, Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        let _ = self.module_functions.insert(key, id);
        Ok(id)
    }

    fn module_function_value(
        &mut self,
        module: &mut ObjectModule,
        function_module: &EcoString,
        function_name: &EcoString,
        arity: usize,
    ) -> Result<Value> {
        let func_id = self.ensure_module_function(module, function_module, function_name, arity)?;
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let code_ptr = self.builder.ins().func_addr(self.pointer_type, func_ref);
        let env_ptr = self.builder.ins().iconst(self.pointer_type, 0);
        let env_len = self.builder.ins().iconst(self.pointer_type, 0);
        let alloc_func = self.declare_runtime_alloc_closure(module)?;
        let alloc_ref = module.declare_func_in_func(alloc_func, &mut self.builder.func);
        let call = self
            .builder
            .ins()
            .call(alloc_ref, &[code_ptr, env_ptr, env_len]);
        let results = self.builder.inst_results(call);
        Ok(results[0])
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

    fn float_constant(&mut self, module: &mut ObjectModule, literal: &str) -> Result<Value> {
        let key: EcoString = literal.into();
        let data_id = if let Some(id) = self.float_constants.get(&key) {
            *id
        } else {
            let number = literal
                .parse::<f64>()
                .map_err(|err| crate::Error::NativeCodegen {
                    message: format!("invalid float literal `{literal}`: {err}"),
                })?;

            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&FLOAT_HEADER.to_le_bytes());
            bytes.extend_from_slice(&number.to_bits().to_le_bytes());

            let mut description = DataDescription::new();
            description.define(bytes.into_boxed_slice());

            let name = format!("gleam$float_{}", self.float_constants.len());
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
            let _ = self.float_constants.insert(key.clone(), id);
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

    fn declare_runtime_list_cons(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_list_cons {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("gleam_list_cons", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_list_cons = Some(id);
        Ok(id)
    }

    fn declare_runtime_alloc_record(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_alloc_record {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_alloc_record", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_alloc_record = Some(id);
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

    fn declare_runtime_int_negate(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_int_negate {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("gleam_int_negate", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_int_negate = Some(id);
        Ok(id)
    }

    fn declare_runtime_float_from_f64(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_float_from_f64 {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(ir::types::F64));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("gleam_float_from_f64", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_float_from_f64 = Some(id);
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

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        Ok((success_block, params))
    }

    fn branch_on_float_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        float_value: f64,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>)> {
        let success_block = self.create_subject_block(subject_count);

        self.builder.switch_to_block(current_block);
        let args = failure_args.to_vec();
        let value_tag_mask = self.builder.ins().iconst(self.pointer_type, VALUE_TAG_MASK);
        let boxed_check = self.builder.ins().band(subject, value_tag_mask);
        let zero = self.builder.ins().iconst(self.pointer_type, 0);
        let is_boxed = self.builder.ins().icmp(IntCC::Equal, boxed_check, zero);

        let pointer_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(pointer_block, self.pointer_type);
        }
        let _ = self
            .builder
            .ins()
            .brif(is_boxed, pointer_block, &args, failure_block, &args);
        self.builder.seal_block(current_block);

        self.builder.switch_to_block(pointer_block);
        let pointer_subject = self.builder.block_params(pointer_block)[0];
        let header =
            self.builder
                .ins()
                .load(self.pointer_type, MemFlags::trusted(), pointer_subject, 0);
        let header_mask = self
            .builder
            .ins()
            .iconst(self.pointer_type, HEADER_FIELD_MASK);
        let header_tag = self.builder.ins().band(header, header_mask);
        let float_tag = self
            .builder
            .ins()
            .iconst(self.pointer_type, TAG_FLOAT as i64);
        let tag_matches = self.builder.ins().icmp(IntCC::Equal, header_tag, float_tag);

        let float_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(float_block, self.pointer_type);
        }
        let _ = self
            .builder
            .ins()
            .brif(tag_matches, float_block, &args, failure_block, &args);
        self.builder.seal_block(pointer_block);

        self.builder.switch_to_block(float_block);
        let params = self.builder.block_params(float_block).to_vec();
        let float_subject = params[0];
        let loaded = self.load_float(float_subject);
        let constant = self.builder.ins().f64const(float_value);
        let cmp = self.builder.ins().fcmp(FloatCC::Equal, loaded, constant);
        let _ = self
            .builder
            .ins()
            .brif(cmp, success_block, &args, failure_block, &args);
        self.builder.seal_block(float_block);

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
            ClauseGuard::GtInt { left, right, .. } => {
                let left = self.lower_clause_guard_operand(module, left)?;
                let right = self.lower_clause_guard_operand(module, right)?;
                let left_int = self.builder.ins().sshr_imm(left, 2);
                let right_int = self.builder.ins().sshr_imm(right, 2);
                Ok(self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThan, left_int, right_int))
            }
            ClauseGuard::GtEqInt { left, right, .. } => {
                let left = self.lower_clause_guard_operand(module, left)?;
                let right = self.lower_clause_guard_operand(module, right)?;
                let left_int = self.builder.ins().sshr_imm(left, 2);
                let right_int = self.builder.ins().sshr_imm(right, 2);
                Ok(self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, left_int, right_int))
            }
            ClauseGuard::LtInt { left, right, .. } => {
                let left = self.lower_clause_guard_operand(module, left)?;
                let right = self.lower_clause_guard_operand(module, right)?;
                let left_int = self.builder.ins().sshr_imm(left, 2);
                let right_int = self.builder.ins().sshr_imm(right, 2);
                Ok(self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, left_int, right_int))
            }
            ClauseGuard::LtEqInt { left, right, .. } => {
                let left = self.lower_clause_guard_operand(module, left)?;
                let right = self.lower_clause_guard_operand(module, right)?;
                let left_int = self.builder.ins().sshr_imm(left, 2);
                let right_int = self.builder.ins().sshr_imm(right, 2);
                Ok(self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedLessThanOrEqual, left_int, right_int))
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
        capture_flags: &[bool],
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        let extra_count = capture_flags.iter().filter(|flag| **flag).count();

        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        for _ in 0..extra_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }

        let mut success_args = Vec::with_capacity(subject_count + extra_count);
        success_args.extend_from_slice(failure_args);
        let mem_flags = MemFlags::trusted();
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

        let _ = self.builder.ins().brif(
            is_boxed,
            pointer_block,
            &[subject],
            failure_block,
            failure_args,
        );
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
            if extra_count > 0 {
                return Err(crate::Error::NativeCodegen {
                    message:
                        "boolean constructors with arguments are not yet supported in native functions"
                            .into(),
                });
            }

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
            let _ = self.builder.ins().brif(
                both_match,
                success_block,
                &success_args,
                failure_block,
                failure_args,
            );
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
                failure_args,
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

            let mut captured_fields = Vec::with_capacity(extra_count);
            if extra_count > 0 {
                let field_base = HEADER_SIZE + (2 * size_of::<u32>() as i32);
                let pointer_stride = self.pointer_bytes() as i32;
                for (field_index, capture) in capture_flags.iter().enumerate() {
                    if *capture {
                        let field_offset = field_base + (field_index as i32) * pointer_stride;
                        let field_value = self.builder.ins().load(
                            self.pointer_type,
                            mem_flags,
                            record_subject,
                            field_offset,
                        );
                        captured_fields.push(field_value);
                    }
                }
            }

            success_args.extend(captured_fields.iter().copied());

            let _ = self.builder.ins().brif(
                index_matches,
                success_block,
                &success_args,
                failure_block,
                failure_args,
            );
            self.builder.seal_block(record_block);
        }

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    #[allow(clippy::too_many_arguments)]
    fn branch_on_tuple_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        arity: usize,
        capture_flags: &[bool],
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        let extra_count = capture_flags.iter().filter(|flag| **flag).count();

        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        for _ in 0..extra_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }

        let mut success_args = Vec::with_capacity(subject_count + extra_count);
        success_args.extend_from_slice(failure_args);
        let mem_flags = MemFlags::trusted();

        let pointer_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(pointer_block, self.pointer_type);

        let value_tag_mask = self.builder.ins().iconst(self.pointer_type, VALUE_TAG_MASK);
        let boxed_check = self.builder.ins().band(subject, value_tag_mask);
        let zero = self.builder.ins().iconst(self.pointer_type, 0);
        let is_boxed = self.builder.ins().icmp(IntCC::Equal, boxed_check, zero);

        let _ = self.builder.ins().brif(
            is_boxed,
            pointer_block,
            &[subject],
            failure_block,
            failure_args,
        );
        self.builder.seal_block(current_block);

        self.builder.switch_to_block(pointer_block);
        let tuple_subject = self.builder.block_params(pointer_block)[0];
        let header = self
            .builder
            .ins()
            .load(self.pointer_type, mem_flags, tuple_subject, 0);
        let header_mask = self
            .builder
            .ins()
            .iconst(self.pointer_type, HEADER_FIELD_MASK);
        let header_tag = self.builder.ins().band(header, header_mask);
        let tuple_tag = self.builder.ins().iconst(self.pointer_type, TAG_TUPLE);
        let tag_matches = self.builder.ins().icmp(IntCC::Equal, header_tag, tuple_tag);

        let arity_shifted = self.builder.ins().ushr_imm(header, HEADER_ARITY_SHIFT);
        let arity_value = self.builder.ins().band(arity_shifted, header_mask);
        let arity_u16 = u16::try_from(arity).map_err(|_| crate::Error::NativeCodegen {
            message: "tuple arity exceeds native runtime limits".into(),
        })?;
        let expected = self
            .builder
            .ins()
            .iconst(self.pointer_type, i64::from(arity_u16));
        let arity_matches = self.builder.ins().icmp(IntCC::Equal, arity_value, expected);

        let both_match = self.builder.ins().band(tag_matches, arity_matches);

        let tuple_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(tuple_block, self.pointer_type);
        let _ = self.builder.ins().brif(
            both_match,
            tuple_block,
            &[tuple_subject],
            failure_block,
            failure_args,
        );
        self.builder.seal_block(pointer_block);

        self.builder.switch_to_block(tuple_block);
        let tuple_subject = self.builder.block_params(tuple_block)[0];

        if capture_flags.len() != arity {
            return Err(crate::Error::NativeCodegen {
                message: "tuple capture flag length mismatch in native list pattern lowering"
                    .into(),
            });
        }

        let mut captured_fields = Vec::with_capacity(extra_count);
        if extra_count > 0 {
            let pointer_stride = self.pointer_bytes() as i32;
            for (field_index, capture) in capture_flags.iter().enumerate() {
                if *capture {
                    let offset = HEADER_SIZE + (field_index as i32) * pointer_stride;
                    let field_value = self.builder.ins().load(
                        self.pointer_type,
                        mem_flags,
                        tuple_subject,
                        offset,
                    );
                    captured_fields.push(field_value);
                }
            }
        }

        success_args.extend(captured_fields.iter().copied());

        let _ = self.builder.ins().jump(success_block, &success_args);
        self.builder.seal_block(tuple_block);

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    #[allow(clippy::too_many_arguments)]
    fn branch_on_list_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject_index: usize,
        capture_heads: &[bool],
        head_patterns: &[Option<ListHeadMatch<'_>>],
        capture_tail: bool,
        ensure_exact: bool,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        let head_capture_count = capture_heads.iter().filter(|capture| **capture).count();
        let pattern_capture_count: usize = head_patterns
            .iter()
            .filter_map(|info| info.as_ref())
            .map(|info| info.capture_flags().iter().filter(|flag| **flag).count())
            .sum();
        let extra_count =
            head_capture_count + pattern_capture_count + if capture_tail { 1 } else { 0 };

        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        for _ in 0..extra_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }

        let pointer_bytes = self.pointer_bytes();
        let storage_slot = if extra_count > 0 {
            Some(self.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                (extra_count * pointer_bytes) as u32,
            )))
        } else {
            None
        };

        let mem_flags = MemFlags::trusted();
        let mut subjects: Vec<Value> = failure_args.to_vec();
        let mut current_block = current_block;
        let mut current_subject =
            subjects
                .get(subject_index)
                .copied()
                .ok_or_else(|| crate::Error::NativeCodegen {
                    message: "invalid subject index for list pattern".into(),
                })?;
        let mut stored = 0usize;

        assert_eq!(capture_heads.len(), head_patterns.len());

        for (index, capture) in capture_heads.iter().enumerate() {
            self.builder.switch_to_block(current_block);

            let nil_func = self.declare_runtime_nil(module)?;
            let nil_ref = module.declare_func_in_func(nil_func, &mut self.builder.func);
            let nil_call = self.builder.ins().call(nil_ref, &[]);
            let nil_value = self.builder.inst_results(nil_call)[0];

            let is_nil = self
                .builder
                .ins()
                .icmp(IntCC::Equal, current_subject, nil_value);

            let non_nil_block = self.builder.create_block();
            for _ in 0..subject_count {
                let _ = self
                    .builder
                    .append_block_param(non_nil_block, self.pointer_type);
            }
            let args = subjects.clone();
            let _ = self
                .builder
                .ins()
                .brif(is_nil, failure_block, &args, non_nil_block, &args);
            self.builder.seal_block(current_block);

            self.builder.switch_to_block(non_nil_block);
            current_block = non_nil_block;
            subjects = self.builder.block_params(current_block).to_vec();
            current_subject = subjects[subject_index];

            let value_tag_mask = self.builder.ins().iconst(self.pointer_type, VALUE_TAG_MASK);
            let boxed_check = self.builder.ins().band(current_subject, value_tag_mask);
            let zero = self.builder.ins().iconst(self.pointer_type, 0);
            let is_boxed = self.builder.ins().icmp(IntCC::Equal, boxed_check, zero);

            let pointer_block = self.builder.create_block();
            for _ in 0..subject_count {
                let _ = self
                    .builder
                    .append_block_param(pointer_block, self.pointer_type);
            }
            let args = subjects.clone();
            let _ = self
                .builder
                .ins()
                .brif(is_boxed, pointer_block, &args, failure_block, &args);
            self.builder.seal_block(current_block);

            self.builder.switch_to_block(pointer_block);
            current_block = pointer_block;
            subjects = self.builder.block_params(current_block).to_vec();
            current_subject = subjects[subject_index];

            let header = self
                .builder
                .ins()
                .load(self.pointer_type, mem_flags, current_subject, 0);
            let header_mask = self
                .builder
                .ins()
                .iconst(self.pointer_type, HEADER_FIELD_MASK);
            let header_tag = self.builder.ins().band(header, header_mask);
            let list_tag = self.builder.ins().iconst(self.pointer_type, TAG_LIST);
            let is_list = self.builder.ins().icmp(IntCC::Equal, header_tag, list_tag);

            let list_block = self.builder.create_block();
            for _ in 0..subject_count {
                let _ = self
                    .builder
                    .append_block_param(list_block, self.pointer_type);
            }
            let args = subjects.clone();
            let _ = self
                .builder
                .ins()
                .brif(is_list, list_block, &args, failure_block, &args);
            self.builder.seal_block(current_block);

            self.builder.switch_to_block(list_block);
            current_block = list_block;
            subjects = self.builder.block_params(current_block).to_vec();
            current_subject = subjects[subject_index];

            let head_pattern = head_patterns[index].as_ref();
            let mut head_extras: Vec<Value> = Vec::new();
            if let Some(pattern) = head_pattern {
                let head_value = self.builder.ins().load(
                    self.pointer_type,
                    mem_flags,
                    current_subject,
                    HEADER_SIZE,
                );
                let (block, params, extras) = match pattern {
                    ListHeadMatch::Constructor(info) => self.branch_on_constructor_pattern(
                        current_block,
                        head_value,
                        info.constructor,
                        info.type_,
                        info.capture_flags.as_slice(),
                        failure_block,
                        &subjects,
                        subject_count,
                    )?,
                    ListHeadMatch::Tuple(info) => self.branch_on_tuple_pattern(
                        current_block,
                        head_value,
                        info.arity,
                        info.capture_flags.as_slice(),
                        failure_block,
                        &subjects,
                        subject_count,
                    )?,
                };
                current_block = block;
                subjects = params;
                current_subject = subjects[subject_index];
                head_extras = extras;
                if self.builder.current_block() != Some(current_block) {
                    self.builder.switch_to_block(current_block);
                }
            }

            let tail_offset = HEADER_SIZE + pointer_bytes as i32;
            let head_value =
                self.builder
                    .ins()
                    .load(self.pointer_type, mem_flags, current_subject, HEADER_SIZE);
            let tail =
                self.builder
                    .ins()
                    .load(self.pointer_type, mem_flags, current_subject, tail_offset);

            if *capture {
                if let Some(slot) = storage_slot {
                    let offset = (stored * pointer_bytes) as i32;
                    let _ = self.builder.ins().stack_store(head_value, slot, offset);
                }
                stored += 1;
            }

            if let Some(pattern) = head_pattern {
                let mut extras_iter = head_extras.into_iter();
                for capture_flag in pattern.capture_flags() {
                    if *capture_flag {
                        let Some(value) = extras_iter.next() else {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "missing head capture value in native list pattern lowering"
                                        .into(),
                            });
                        };
                        if let Some(slot) = storage_slot {
                            let offset = (stored * pointer_bytes) as i32;
                            let _ = self.builder.ins().stack_store(value, slot, offset);
                        }
                        stored += 1;
                    }
                }
            }

            subjects[subject_index] = tail;
            current_subject = tail;
        }

        if ensure_exact {
            self.builder.switch_to_block(current_block);

            let nil_func = self.declare_runtime_nil(module)?;
            let nil_ref = module.declare_func_in_func(nil_func, &mut self.builder.func);
            let nil_call = self.builder.ins().call(nil_ref, &[]);
            let nil_value = self.builder.inst_results(nil_call)[0];

            let is_nil = self
                .builder
                .ins()
                .icmp(IntCC::Equal, current_subject, nil_value);
            let exact_block = self.builder.create_block();
            for _ in 0..subject_count {
                let _ = self
                    .builder
                    .append_block_param(exact_block, self.pointer_type);
            }
            let args = subjects.clone();
            let _ = self
                .builder
                .ins()
                .brif(is_nil, exact_block, &args, failure_block, &args);
            self.builder.seal_block(current_block);

            self.builder.switch_to_block(exact_block);
            current_block = exact_block;
            subjects = self.builder.block_params(current_block).to_vec();
            current_subject = subjects[subject_index];
        }

        if capture_tail {
            if let Some(slot) = storage_slot {
                let offset = (stored * pointer_bytes) as i32;
                let _ = self
                    .builder
                    .ins()
                    .stack_store(current_subject, slot, offset);
            }
            stored += 1;
        }

        let mut extra_values = Vec::with_capacity(stored);
        if let Some(slot) = storage_slot {
            for index in 0..stored {
                let offset = (index * pointer_bytes) as i32;
                let value = self
                    .builder
                    .ins()
                    .stack_load(self.pointer_type, slot, offset);
                extra_values.push(value);
            }
        }

        let mut success_args = subjects.clone();
        success_args.extend(extra_values.iter().copied());

        let _ = self.builder.ins().jump(success_block, &success_args);
        self.builder.seal_block(current_block);
        self.builder.switch_to_block(success_block);
        let params = self.builder.block_params(success_block).to_vec();
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    fn tuple_element(&mut self, tuple: Value, index: u64) -> Result<Value> {
        let pointer_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(pointer_block, self.pointer_type);
        let failure_block = self.builder.create_block();

        let mem_flags = MemFlags::trusted();
        let value_tag_mask = self.builder.ins().iconst(self.pointer_type, VALUE_TAG_MASK);
        let boxed_check = self.builder.ins().band(tuple, value_tag_mask);
        let zero = self.builder.ins().iconst(self.pointer_type, 0);
        let is_boxed = self.builder.ins().icmp(IntCC::Equal, boxed_check, zero);

        let _ = self
            .builder
            .ins()
            .brif(is_boxed, pointer_block, &[tuple], failure_block, &[]);

        self.builder.switch_to_block(pointer_block);
        let pointer_subject = self.builder.block_params(pointer_block)[0];
        let header = self
            .builder
            .ins()
            .load(self.pointer_type, mem_flags, pointer_subject, 0);
        let header_mask = self
            .builder
            .ins()
            .iconst(self.pointer_type, HEADER_FIELD_MASK);
        let header_tag = self.builder.ins().band(header, header_mask);
        let tuple_tag = self.builder.ins().iconst(self.pointer_type, TAG_TUPLE);
        let is_tuple = self.builder.ins().icmp(IntCC::Equal, header_tag, tuple_tag);

        let tuple_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(tuple_block, self.pointer_type);
        let _ = self
            .builder
            .append_block_param(tuple_block, self.pointer_type);
        let _ = self.builder.ins().brif(
            is_tuple,
            tuple_block,
            &[pointer_subject, header],
            failure_block,
            &[],
        );
        self.builder.seal_block(pointer_block);

        self.builder.switch_to_block(tuple_block);
        let tuple_ptr = self.builder.block_params(tuple_block)[0];
        let tuple_header = self.builder.block_params(tuple_block)[1];
        let header_mask = self
            .builder
            .ins()
            .iconst(self.pointer_type, HEADER_FIELD_MASK);
        let arity_shifted = self
            .builder
            .ins()
            .ushr_imm(tuple_header, HEADER_ARITY_SHIFT);
        let arity_value = self.builder.ins().band(arity_shifted, header_mask);

        let index_i64 = i64::try_from(index).map_err(|_| crate::Error::NativeCodegen {
            message: "tuple index exceeds native backend limits".into(),
        })?;
        let index_value = self.builder.ins().iconst(self.pointer_type, index_i64);
        let in_bounds = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, index_value, arity_value);

        let element_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(element_block, self.pointer_type);
        let _ = self
            .builder
            .ins()
            .brif(in_bounds, element_block, &[tuple_ptr], failure_block, &[]);
        self.builder.seal_block(tuple_block);

        self.builder.switch_to_block(failure_block);
        let _ = self.builder.ins().trap(TrapCode::User(0));
        self.builder.seal_block(failure_block);

        self.builder.switch_to_block(element_block);
        let tuple_ptr = self.builder.block_params(element_block)[0];
        let index_usize = usize::try_from(index).map_err(|_| crate::Error::NativeCodegen {
            message: "tuple index exceeds native backend limits".into(),
        })?;
        let index_i32 = i32::try_from(index_usize).map_err(|_| crate::Error::NativeCodegen {
            message: "tuple index offset exceeds native backend limits".into(),
        })?;
        let pointer_stride = self.pointer_bytes() as i32;
        let offset = HEADER_SIZE + index_i32 * pointer_stride;
        let element = self
            .builder
            .ins()
            .load(self.pointer_type, mem_flags, tuple_ptr, offset);
        self.builder.seal_block(element_block);
        Ok(element)
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
