use crate::{
    Result,
    ast::{
        AssignmentKind, BinOp, BitArrayOption, BitArraySize, Endianness, Function, ModuleConstant,
        Pattern, PipelineAssignmentKind, Statement, TypedArg, TypedAssert, TypedDefinition,
        TypedExpr, TypedPipelineAssignment, TypedStatement,
    },
    bit_array::GetLiteralValue,
    type_::{ModuleValueConstructor, Type, ValueConstructorVariant},
};
use cranelift_codegen::ir::{
    self, InstBuilder, MemFlags, StackSlotData, StackSlotKind, TrapCode, Value,
    condcodes::{FloatCC, IntCC},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataId, FuncId, Linkage, Module};
use cranelift_object::ObjectModule;
use ecow::EcoString;
use num_traits::ToPrimitive;
use std::{collections::HashMap, convert::TryFrom, sync::Arc};

use super::context::{LoweringContext, encode_small_int};
use super::patterns::{
    ConstructorTupleCondition, ConstructorTupleInfo, ListConstructorInfo, ListHeadMatch,
    ListTupleInfo, collect_list_pattern_info,
};
use super::{BindingSource, FunctionIdMap, ModuleConfig};

pub(super) fn function_symbol_name(module: &str, name: &EcoString, arity: usize) -> String {
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

fn resolve_assign_pattern<'pattern>(
    mut pattern: &'pattern Pattern<Arc<Type>>,
    bindings: &mut Vec<(EcoString, BindingSource)>,
    subject_index: usize,
) -> &'pattern Pattern<Arc<Type>> {
    loop {
        match pattern {
            Pattern::Assign {
                name,
                pattern: inner,
                ..
            } => {
                #[cfg(debug_assertions)]
                eprintln!(
                    "native lowering: resolving assign pattern `{name}` on subject index {subject_index}"
                );
                bindings.push((name.clone(), BindingSource::Subject(subject_index)));
                pattern = inner;
            }
            _ => return pattern,
        }
    }
}

fn strip_assign_aliases<'pattern>(
    mut pattern: &'pattern Pattern<Arc<Type>>,
    aliases: &mut Vec<EcoString>,
) -> &'pattern Pattern<Arc<Type>> {
    loop {
        match pattern {
            Pattern::Assign {
                name,
                pattern: inner,
                ..
            } => {
                aliases.push(name.clone());
                pattern = inner;
            }
            _ => return pattern,
        }
    }
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

fn collect_module_constants(
    module: &crate::ast::TypedModule,
) -> Vec<&ModuleConstant<Arc<Type>, EcoString>> {
    module
        .definitions
        .iter()
        .filter_map(|definition| match definition {
            TypedDefinition::ModuleConstant(constant) => Some(constant),
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

pub(crate) fn lower_module_functions(
    module: &mut ObjectModule,
    config: &ModuleConfig<'_>,
) -> Result<Option<FuncId>> {
    let functions = collect_module_functions(&config.module.ast);
    if functions.is_empty() {
        return Ok(None);
    }

    let module_constants = collect_module_constants(&config.module.ast);

    let function_ids = declare_module_functions(module, &config.module.name, &functions)?;
    let mut zero_arity_records = HashMap::new();
    let mut string_data = HashMap::new();
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
            &module_constants,
            &mut zero_arity_records,
            &mut string_data,
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
    module_constants: &[&ModuleConstant<Arc<Type>, EcoString>],
    zero_arity_records: &mut HashMap<(EcoString, u16), DataId>,
    string_data: &mut HashMap<EcoString, DataId>,
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
            string_data,
            float_constants,
            record_constructors,
            module_functions,
            closure_counter,
        );
        lowering.mark_sealed(block);

        for constant in module_constants {
            let value = lowering.lower_constant(module, &constant.value)?;
            lowering.define(&constant.name, value);
        }

        for (value, arg) in block_params.iter().zip(function.arguments.iter()) {
            if let Some(name) = arg.get_variable_name() {
                lowering.define(name, *value);
            }
        }

        let value = lower_block(module, function.body.as_slice(), &mut lowering)?;
        let _ = lowering.builder.ins().return_(&[value]);
    }

    builder.finalize();

    if let Err(err) = module.define_function(func_id, &mut ctx) {
        let clif = format!("{}", ctx.func.display());
        let func_name = function
            .name
            .as_ref()
            .map(|(_, name)| name.as_str())
            .unwrap_or("<anonymous>");
        return Err(crate::Error::NativeCodegen {
            message: format!("error lowering {module_name}.{func_name}: {err} ({err:?})\n{clif}",),
        });
    }
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
            Statement::Use(use_) => {
                last = lower_expression(module, &use_.call, ctx)?;
            }
            Statement::Assert(assert) => {
                lower_assert_statement(module, assert, ctx)?;
            }
        }
    }
    ctx.pop_scope();
    Ok(last)
}

pub(super) fn lower_expression(
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
            ValueConstructorVariant::ModuleConstant { literal, .. } => {
                ctx.lower_constant(module, literal)
            }
            ValueConstructorVariant::LocalConstant { literal } => {
                ctx.lower_constant(module, literal)
            }
            ValueConstructorVariant::ModuleFn {
                module: function_module,
                name: function_name,
                arity,
                ..
            } => ctx.module_function_value(module, function_module, function_name, *arity),
            ValueConstructorVariant::LocalVariable { .. } => {
                ctx.lookup(name)
                    .copied()
                    .ok_or_else(|| crate::Error::NativeCodegen {
                        message: format!("unknown variable `{name}` in native main"),
                    })
            }
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
            ModuleValueConstructor::Constant { literal, .. } => ctx.lower_constant(module, literal),
        },

        TypedExpr::BitArray { segments, .. } => {
            let builder_id = ctx.declare_runtime_bit_array_builder_new(module)?;
            let builder_ref = module.declare_func_in_func(builder_id, &mut ctx.builder.func);
            let call = ctx.builder.ins().call(builder_ref, &[]);
            let mut builder_value = ctx.builder.inst_results(call)[0];

            if segments.is_empty() {
                let finish_id = ctx.declare_runtime_bit_array_builder_finish(module)?;
                let finish_ref = module.declare_func_in_func(finish_id, &mut ctx.builder.func);
                let finish_call = ctx.builder.ins().call(finish_ref, &[builder_value]);
                let result = ctx.builder.inst_results(finish_call)[0];
                return Ok(result);
            }

            for segment in segments {
                if segment.type_.is_bit_array() {
                    let value = lower_expression(module, &segment.value, ctx)?;
                    let (size_value, has_size_value) = if let Some(size_expr) = segment.size() {
                        let size = lower_expression(module, size_expr, ctx)?;
                        (size, ctx.builder.ins().iconst(ctx.pointer_type, 1))
                    } else {
                        (
                            ctx.builder.ins().iconst(ctx.pointer_type, 0),
                            ctx.builder.ins().iconst(ctx.pointer_type, 0),
                        )
                    };
                    let unit_value = ctx
                        .builder
                        .ins()
                        .iconst(ctx.pointer_type, i64::from(segment.unit()));
                    let func_id = ctx.declare_runtime_bit_array_builder_append_bit_array(module)?;
                    let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
                    let call = ctx.builder.ins().call(
                        func_ref,
                        &[builder_value, value, size_value, has_size_value, unit_value],
                    );
                    builder_value = ctx.builder.inst_results(call)[0];
                    continue;
                }

                if segment.type_.is_int() {
                    let value = lower_expression(module, &segment.value, ctx)?;
                    let (size_value, has_size_value) = if let Some(size_expr) = segment.size() {
                        let size = lower_expression(module, size_expr, ctx)?;
                        (size, ctx.builder.ins().iconst(ctx.pointer_type, 1))
                    } else {
                        (
                            ctx.builder.ins().iconst(ctx.pointer_type, 0),
                            ctx.builder.ins().iconst(ctx.pointer_type, 0),
                        )
                    };
                    let unit_value = ctx
                        .builder
                        .ins()
                        .iconst(ctx.pointer_type, i64::from(segment.unit()));
                    let default_size_value = ctx.builder.ins().iconst(ctx.pointer_type, 8);
                    let signed_value = ctx.builder.ins().iconst(ctx.pointer_type, 1);
                    let endianness = match segment.endianness() {
                        Endianness::Big => 0,
                        Endianness::Little => 1,
                    };
                    let endianness_value = ctx
                        .builder
                        .ins()
                        .iconst(ctx.pointer_type, i64::from(endianness));
                    let func_id = ctx.declare_runtime_bit_array_builder_append_int(module)?;
                    let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
                    let call = ctx.builder.ins().call(
                        func_ref,
                        &[
                            builder_value,
                            value,
                            size_value,
                            has_size_value,
                            unit_value,
                            default_size_value,
                            signed_value,
                            endianness_value,
                        ],
                    );
                    builder_value = ctx.builder.inst_results(call)[0];
                    continue;
                }

                if segment.type_.is_string() {
                    if segment.has_utf16_option() || segment.has_utf32_option() {
                        return Err(crate::Error::NativeCodegen {
                            message: "utf16 and utf32 string segments are not yet supported in native functions"
                                .into(),
                        });
                    }
                    let value = lower_expression(module, &segment.value, ctx)?;
                    let func_id =
                        ctx.declare_runtime_bit_array_builder_append_string_utf8(module)?;
                    let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
                    let call = ctx.builder.ins().call(func_ref, &[builder_value, value]);
                    builder_value = ctx.builder.inst_results(call)[0];
                    continue;
                }

                return Err(crate::Error::NativeCodegen {
                    message: format!(
                        "bit array segment type is not yet supported in native functions: {:?}",
                        segment.type_
                    ),
                });
            }

            let finish_id = ctx.declare_runtime_bit_array_builder_finish(module)?;
            let finish_ref = module.declare_func_in_func(finish_id, &mut ctx.builder.func);
            let call = ctx.builder.ins().call(finish_ref, &[builder_value]);
            let result = ctx.builder.inst_results(call)[0];
            Ok(result)
        }

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
    let value = lower_expression(module, &assignment.value, ctx)?;
    let failure = match &assignment.kind {
        AssignmentKind::Let | AssignmentKind::Generated => AssignmentFailure::Trap,
        AssignmentKind::Assert { message, .. } => AssignmentFailure::Assert {
            message: message.as_ref(),
        },
    };

    lower_pattern_assignment(module, ctx, &assignment.pattern, value, failure)
}

fn lower_assert_statement(
    module: &mut ObjectModule,
    assert: &TypedAssert,
    ctx: &mut LoweringContext<'_, '_, '_>,
) -> Result<()> {
    let condition = lower_expression(module, &assert.value, ctx)?;
    let true_value = ctx.bool_constant(module, true)?;
    let is_true = ctx.builder.ins().icmp(IntCC::Equal, condition, true_value);

    let current_block = ctx
        .builder
        .current_block()
        .ok_or_else(|| crate::Error::NativeCodegen {
            message: "assert lowering requires an active block".into(),
        })?;

    let success_block = ctx.builder.create_block();
    let failure_block = ctx.builder.create_block();

    let _ = ctx
        .builder
        .ins()
        .brif(is_true, success_block, &[], failure_block, &[]);
    ctx.seal_block(current_block);

    ctx.builder.switch_to_block(failure_block);
    if let Some(message) = &assert.message {
        let _ = lower_expression(module, message, ctx)?;
    }
    let _ = ctx.builder.ins().trap(TrapCode::User(0));
    ctx.seal_block(failure_block);

    ctx.builder.switch_to_block(success_block);
    Ok(())
}

enum AssignmentFailure<'a> {
    Trap,
    Assert { message: Option<&'a TypedExpr> },
}

fn lower_pattern_assignment(
    module: &mut ObjectModule,
    ctx: &mut LoweringContext<'_, '_, '_>,
    pattern: &Pattern<Arc<Type>>,
    value: Value,
    failure: AssignmentFailure<'_>,
) -> Result<()> {
    match pattern {
        Pattern::Variable { name, .. } => {
            ctx.define(name, value);
            return Ok(());
        }
        Pattern::Discard { .. } => {
            return Ok(());
        }
        _ => {}
    }

    let subject_count = 1usize;
    let subjects = vec![value];
    let mut pattern_block =
        ctx.builder
            .current_block()
            .ok_or_else(|| crate::Error::NativeCodegen {
                message: "pattern assignment requires an active block".into(),
            })?;

    let trap_block = ctx.create_subject_block(subject_count);

    let mut bindings = Vec::new();

    match pattern {
        Pattern::Constructor {
            constructor,
            arguments,
            spread,
            type_,
            ..
        } if spread.is_none() => {
            let constructor = constructor
                .expect_ref("pattern constructor must be known during native code generation");

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
                                "constructor assignment pattern argument `{other:?}` is not yet supported in native functions"
                            ),
                        });
                    }
                }
            }

            let failure_args = subjects.clone();
            let failure_block = ctx.create_subject_block(subject_count);
            let (block, _params, extras) = ctx.branch_on_constructor_pattern(
                pattern_block,
                subjects[0],
                constructor,
                type_,
                &capture_flags,
                failure_block,
                &failure_args,
                subject_count,
            )?;
            pattern_block = block;
            if ctx.builder.current_block() != Some(pattern_block) {
                ctx.builder.switch_to_block(pattern_block);
            }

            let mut extras_iter = extras.into_iter();
            for (capture_flag, binding_name) in capture_flags.iter().zip(binding_names.iter()) {
                if *capture_flag {
                    let Some(value) = extras_iter.next() else {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "missing constructor capture value in native assignment lowering"
                                    .into(),
                        });
                    };
                    if let Some(name) = binding_name {
                        bindings.push((name.clone(), value));
                    }
                }
            }

            ctx.builder.switch_to_block(failure_block);
            let params = ctx.builder.block_params(failure_block).to_vec();
            let _ = ctx.builder.ins().jump(trap_block, &params);
            ctx.seal_block(failure_block);
            ctx.builder.switch_to_block(pattern_block);
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
                        head_matches.push(Some(ListHeadMatch::Constructor(ListConstructorInfo {
                            constructor,
                            type_,
                            capture_flags,
                        })));
                        head_field_bindings.push(Some(binding_names));
                    }
                    Pattern::Tuple { elements, .. } => {
                        let mut capture_flags = Vec::with_capacity(elements.len());
                        let mut binding_names = Vec::with_capacity(elements.len());

                        for pattern in elements {
                            match pattern {
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

            let failure_args = subjects.clone();
            let failure_block = ctx.create_subject_block(subject_count);
            let (block, _params, extras) = ctx.branch_on_list_pattern(
                module,
                pattern_block,
                0,
                &capture_heads,
                &head_matches,
                capture_tail,
                tail.is_none(),
                failure_block,
                &failure_args,
                subject_count,
                subject_count,
            )?;
            pattern_block = block;
            if ctx.builder.current_block() != Some(pattern_block) {
                ctx.builder.switch_to_block(pattern_block);
            }

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
                                "missing captured list head value in native assignment lowering"
                                    .into(),
                        });
                    };
                    if let Some(name) = name {
                        bindings.push((name.clone(), value));
                    }
                }

                if let Some(binding_names) = field_bindings {
                    let Some(info) = head_matches.get(index).and_then(|opt| opt.as_ref()) else {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "missing head match info for list bindings in native assignment lowering"
                                    .into(),
                        });
                    };

                    if info.capture_flags().len() != binding_names.len() {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "list head binding length mismatch in native assignment lowering"
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
                                        "missing captured constructor field in native assignment lowering"
                                            .into(),
                                });
                            };
                            if let Some(name) = binding_name {
                                bindings.push((name.clone(), value));
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
                                "missing captured list tail value in native assignment lowering"
                                    .into(),
                        });
                    };
                    bindings.push((name, value));
                }
            }

            ctx.builder.switch_to_block(failure_block);
            let params = ctx.builder.block_params(failure_block).to_vec();
            let _ = ctx.builder.ins().jump(trap_block, &params);
            ctx.seal_block(failure_block);
            ctx.builder.switch_to_block(pattern_block);
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
                                "tuple assignment pattern element `{other:?}` is not yet supported in native functions"
                            ),
                        });
                    }
                }
            }

            let failure_args = subjects.clone();
            let failure_block = ctx.create_subject_block(subject_count);
            let (block, _params, extras) = ctx.branch_on_tuple_pattern(
                pattern_block,
                subjects[0],
                elements.len(),
                &capture_flags,
                failure_block,
                &failure_args,
                subject_count,
            )?;
            pattern_block = block;
            if ctx.builder.current_block() != Some(pattern_block) {
                ctx.builder.switch_to_block(pattern_block);
            }

            let mut extras_iter = extras.into_iter();
            for (capture_flag, binding_name) in capture_flags.iter().zip(binding_names.iter()) {
                if *capture_flag {
                    let Some(value) = extras_iter.next() else {
                        return Err(crate::Error::NativeCodegen {
                            message: "missing captured tuple element in native assignment lowering"
                                .into(),
                        });
                    };
                    if let Some(name) = binding_name {
                        bindings.push((name.clone(), value));
                    }
                }
            }

            ctx.builder.switch_to_block(failure_block);
            let params = ctx.builder.block_params(failure_block).to_vec();
            let _ = ctx.builder.ins().jump(trap_block, &params);
            ctx.seal_block(failure_block);
            ctx.builder.switch_to_block(pattern_block);
        }
        other => {
            return Err(crate::Error::NativeCodegen {
                message: format!("pattern `{other:?}` is not yet supported in native assignments"),
            });
        }
    }

    if ctx.builder.current_block() != Some(pattern_block) {
        ctx.builder.switch_to_block(pattern_block);
    }

    ctx.builder.switch_to_block(trap_block);
    let _ = ctx.builder.block_params(trap_block);
    match failure {
        AssignmentFailure::Trap => {
            let _ = ctx.builder.ins().trap(TrapCode::User(0));
        }
        AssignmentFailure::Assert { message } => {
            if let Some(message) = message {
                let _ = lower_expression(module, message, ctx)?;
            }
            let _ = ctx.builder.ins().trap(TrapCode::User(0));
        }
    }
    ctx.seal_block(trap_block);

    ctx.builder.switch_to_block(pattern_block);

    for (name, value) in bindings {
        ctx.define(&name, value);
    }
    Ok(())
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
        let mut bindings: Vec<(EcoString, BindingSource)> = Vec::new();

        for (subject_index, pattern) in clause.pattern.iter().enumerate() {
            let current_params = ctx.builder.block_params(pattern_block).to_vec();
            for (_, source) in &mut bindings {
                match source {
                    BindingSource::Value(value) => {
                        if let Some(position) =
                            current_params.iter().position(|param| *param == *value)
                        {
                            *value = current_params[position];
                        }
                    }
                    BindingSource::BlockParam { index, value } => {
                        if let Some(param) = current_params.get(*index) {
                            *value = *param;
                        }
                    }
                    BindingSource::Subject(_) => {}
                }
            }

            #[cfg(debug_assertions)]
            eprintln!("native lowering: pattern before resolution = {:?}", pattern);
            let pattern = resolve_assign_pattern(pattern, &mut bindings, subject_index);
            debug_assert!(
                !matches!(pattern, Pattern::Assign { .. }),
                "assign pattern should be resolved before lowering"
            );
            #[cfg(debug_assertions)]
            eprintln!(
                "native lowering: pattern after assign resolution = {:?}",
                pattern
            );

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
                Pattern::String { value, .. } => {
                    let literal = ctx.string_constant(module, value.as_str())?;
                    let (block, params) = ctx.branch_on_string_pattern(
                        module,
                        pattern_block,
                        pattern_subjects[subject_index],
                        literal,
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
                    let mut argument_aliases: Vec<Vec<EcoString>> =
                        Vec::with_capacity(arguments.len());
                    let mut tuple_patterns: Vec<Option<ConstructorTupleInfo>> =
                        Vec::with_capacity(arguments.len());
                    for argument in arguments {
                        if argument.label.is_some() {
                            return Err(crate::Error::NativeCodegen {
                                message: "labelled constructor pattern arguments are not yet supported in native functions"
                                    .into(),
                            });
                        }

                        let mut aliases = Vec::new();
                        let pattern = strip_assign_aliases(&argument.value, &mut aliases);

                        match pattern {
                            Pattern::Variable { name, .. } => {
                                capture_flags.push(true);
                                binding_names.push(Some(name.clone()));
                                argument_aliases.push(aliases);
                                tuple_patterns.push(None);
                            }
                            Pattern::Discard { .. } => {
                                let capture = !aliases.is_empty();
                                capture_flags.push(capture);
                                binding_names.push(None);
                                argument_aliases.push(aliases);
                                tuple_patterns.push(None);
                            }
                            Pattern::Tuple { elements, .. } => {
                                let mut element_capture_flags = Vec::with_capacity(elements.len());
                                let mut element_binding_names = Vec::with_capacity(elements.len());
                                let mut element_conditions = Vec::with_capacity(elements.len());

                                for element in elements {
                                    match element {
                                        Pattern::Variable { name, .. } => {
                                            element_capture_flags.push(true);
                                            element_binding_names.push(Some(name.clone()));
                                            element_conditions
                                                .push(ConstructorTupleCondition::None);
                                        }
                                        Pattern::Discard { .. } => {
                                            element_capture_flags.push(false);
                                            element_binding_names.push(None);
                                            element_conditions
                                                .push(ConstructorTupleCondition::None);
                                        }
                                        Pattern::String { value, .. } => {
                                            element_capture_flags.push(true);
                                            element_binding_names.push(None);
                                            element_conditions.push(
                                                ConstructorTupleCondition::String(value.clone()),
                                            );
                                        }
                                        other => {
                                            return Err(crate::Error::NativeCodegen {
                                                message: format!(
                                                    "tuple constructor pattern element `{other:?}` is not yet supported in native functions"
                                                ),
                                            });
                                        }
                                    }
                                }

                                capture_flags.push(true);
                                binding_names.push(None);
                                argument_aliases.push(aliases);
                                tuple_patterns.push(Some(ConstructorTupleInfo {
                                    capture_flags: element_capture_flags,
                                    binding_names: element_binding_names,
                                    conditions: element_conditions,
                                }));
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

                    while argument_aliases.len() < capture_flags.len() {
                        argument_aliases.push(Vec::new());
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
                    // Ensure we resume writing instructions in the pattern block returned by
                    // the branch lowering helper.
                    if ctx.builder.current_block() != Some(pattern_block) {
                        ctx.builder.switch_to_block(pattern_block);
                    }

                    let mut extra_iter = extras.into_iter();
                    for (index, capture) in capture_flags.iter().enumerate() {
                        if *capture {
                            let Some(value) = extra_iter.next() else {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "missing captured constructor argument in native case lowering"
                                            .into(),
                                });
                            };
                            if let Some(tuple_info) = &tuple_patterns[index] {
                                ctx.lower_constructor_tuple(
                                    module,
                                    &mut pattern_block,
                                    &mut pattern_subjects,
                                    value,
                                    tuple_info,
                                    next_block,
                                    &mut bindings,
                                )?;
                                if ctx.builder.current_block() != Some(pattern_block) {
                                    ctx.builder.switch_to_block(pattern_block);
                                }
                            }
                            if let Some(name) = &binding_names[index] {
                                bindings.push((name.clone(), BindingSource::Value(value)));
                            }
                            for alias in &argument_aliases[index] {
                                bindings.push((alias.clone(), BindingSource::Value(value)));
                            }
                        } else if tuple_patterns[index].is_some() {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "nested tuple constructor pattern requires capturing the argument"
                                        .into(),
                            });
                        } else if !argument_aliases[index].is_empty() {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "constructor alias pattern requires capturing the argument in native functions"
                                        .into(),
                            });
                        }
                    }
                    if extra_iter.next().is_some() {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "unexpected extra constructor capture values in native case lowering"
                                    .into(),
                        });
                    }
                }
                Pattern::List { elements, tail, .. } => {
                    let info = collect_list_pattern_info(elements, tail.as_deref())?;

                    let (block, params, extras) = ctx.branch_on_list_pattern(
                        module,
                        pattern_block,
                        subject_index,
                        info.capture_heads.as_slice(),
                        info.head_matches.as_slice(),
                        info.capture_tail,
                        info.ensure_exact,
                        next_block,
                        pattern_subjects.as_slice(),
                        subject_count,
                        subject_count,
                    )?;
                    pattern_block = block;
                    pattern_subjects = params;

                    if ctx.builder.current_block() != Some(pattern_block) {
                        ctx.builder.switch_to_block(pattern_block);
                    }

                    let mut extra_iter = extras.into_iter();
                    for index in 0..info.capture_heads.len() {
                        let capture = info.capture_heads[index];
                        let name = &info.head_names[index];
                        let field_bindings = &info.head_field_bindings[index];
                        let extra_aliases = &info.head_extra_bindings[index];
                        let nested = &info.head_nested_lists[index];
                        if capture {
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
                            for alias in extra_aliases {
                                bindings.push((alias.clone(), BindingSource::Value(value)));
                            }

                            if let Some(nested_info) = nested {
                                ctx.lower_nested_list(
                                    module,
                                    &mut pattern_block,
                                    &mut pattern_subjects,
                                    value,
                                    nested_info,
                                    next_block,
                                    &mut bindings,
                                )?;
                                if ctx.builder.current_block() != Some(pattern_block) {
                                    ctx.builder.switch_to_block(pattern_block);
                                }
                            }
                        } else if nested.is_some() {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "nested list pattern requires head capture in native case lowering"
                                        .into(),
                            });
                        }

                        if let Some(binding_names) = field_bindings {
                            let Some(info) = info.head_matches[index].as_ref() else {
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
                    if info.capture_tail {
                        if let Some(name) = info.tail_name {
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

                    if extra_iter.next().is_some() {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "unexpected extra values after lowering list pattern in native backend"
                                    .into(),
                        });
                    }
                }
                Pattern::BitArray { segments, .. } => {
                    if segments.is_empty() {
                        let (block, params) = ctx.branch_on_empty_bit_array_pattern(
                            module,
                            pattern_block,
                            pattern_subjects[subject_index],
                            next_block,
                            pattern_subjects.as_slice(),
                            subject_count,
                        )?;
                        pattern_block = block;
                        pattern_subjects = params;
                        continue;
                    }

                    if segments.len() == 1 {
                        let segment = &segments[0];
                        let is_bytes = segment
                            .options
                            .iter()
                            .all(|option| matches!(option, BitArrayOption::Bytes { .. }));

                        if !is_bytes {
                            if let Some(size_pattern) = segment.size()
                                && segment.type_.is_int()
                                && let Some(size_bits) = size_pattern.as_int_literal()
                                && let Some(size_bits) = size_bits.to_i64()
                            {
                                let (block, params, _value) = ctx
                                    .branch_on_sized_int_bit_array_pattern(
                                        module,
                                        pattern_block,
                                        pattern_subjects[subject_index],
                                        size_bits,
                                        next_block,
                                        pattern_subjects.as_slice(),
                                        subject_count,
                                    )?;
                                pattern_block = block;
                                pattern_subjects = params;

                                match segment.value.as_ref() {
                                    Pattern::Variable { name, .. } => {
                                        let block_params =
                                            ctx.builder.block_params(pattern_block).to_vec();
                                        let base_index = pattern_subjects.len();
                                        let captured = block_params
                                            .get(base_index)
                                            .copied()
                                            .ok_or_else(|| crate::Error::NativeCodegen {
                                                message:
                                                    "missing sized int capture parameter in native case lowering"
                                                        .into(),
                                            })?;
                                        bindings
                                            .push((name.clone(), BindingSource::Value(captured)));
                                    }
                                    Pattern::Discard { .. } => {}
                                    _ => {
                                        return Err(crate::Error::NativeCodegen {
                                            message:
                                                "only variable or discard patterns are supported for sized int segments in native functions"
                                                    .into(),
                                        });
                                    }
                                }
                                continue;
                            }

                            return Err(crate::Error::NativeCodegen {
                                message: format!(
                                    "bit array pattern options are not yet supported in native functions: {:?}",
                                    segment.options
                                ),
                            });
                        }

                        match segment.value.as_ref() {
                            Pattern::Variable { name, .. } => {
                                bindings
                                    .push((name.clone(), BindingSource::Subject(subject_index)));
                            }
                            Pattern::Discard { .. } => {}
                            _ => {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "only variable or discard patterns are supported for bytes segments in native functions"
                                            .into(),
                                });
                            }
                        }
                        continue;
                    }

                    if segments.len() != 2 {
                        return Err(crate::Error::NativeCodegen {
                            message: "bit array patterns with unsupported segment count in native functions"
                                .into(),
                        });
                    }

                    let first_segment = &segments[0];
                    let rest_segment = &segments[1];

                    let first_is_bits = first_segment.options.iter().all(|option| {
                        matches!(
                            option,
                            BitArrayOption::Bits { .. } | BitArrayOption::Size { .. }
                        )
                    }) && first_segment
                        .options
                        .iter()
                        .any(|option| matches!(option, BitArrayOption::Bits { .. }));
                    let rest_is_bits = rest_segment
                        .options
                        .iter()
                        .all(|option| matches!(option, BitArrayOption::Bits { .. }));

                    if first_is_bits && rest_is_bits && rest_segment.size().is_none() {
                        let Some(size_pattern) = first_segment.size() else {
                            return Err(crate::Error::NativeCodegen {
                                message: "sized bit array segment missing size in native functions"
                                    .into(),
                            });
                        };

                        let mut resolved_size = match size_pattern {
                            Pattern::BitArraySize(size) => size,
                            other => {
                                return Err(crate::Error::NativeCodegen {
                                    message: format!(
                                        "bit array size pattern `{other:?}` is not yet supported in native functions"
                                    ),
                                });
                            }
                        };

                        loop {
                            match resolved_size {
                                BitArraySize::Block { inner, .. } => {
                                    resolved_size = inner.as_ref();
                                }
                                _ => break,
                            }
                        }

                        let size_value = match resolved_size {
                            BitArraySize::Int { int_value, .. } => {
                                let Some(bits) = int_value.to_i64() else {
                                    return Err(crate::Error::NativeCodegen {
                                        message: "bit array segment size exceeds native limits"
                                            .into(),
                                    });
                                };
                                let encoded = encode_small_int(bits)?;
                                ctx.builder.ins().iconst(ctx.pointer_type, encoded)
                            }
                            BitArraySize::Variable { name, .. } => {
                                let Some(value) = ctx.lookup(name) else {
                                    return Err(crate::Error::NativeCodegen {
                                        message: format!(
                                            "bit array segment size variable `{name}` is not defined in native functions"
                                        ),
                                    });
                                };
                                *value
                            }
                            other => {
                                return Err(crate::Error::NativeCodegen {
                                    message: format!(
                                        "bit array size expression `{:?}` is not yet supported in native functions",
                                        other
                                    ),
                                });
                            }
                        };

                        let prefix_binding = match first_segment.value.as_ref() {
                            Pattern::Variable { name, .. } => Some(name.clone()),
                            Pattern::Discard { .. } => None,
                            _ => {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "only variable or discard patterns are supported for bits segments in native functions"
                                            .into(),
                                });
                            }
                        };

                        let rest_binding = match rest_segment.value.as_ref() {
                            Pattern::Variable { name, .. } => Some(name.clone()),
                            Pattern::Discard { .. } => None,
                            _ => {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "only variable or discard patterns are supported for bits segments in native functions"
                                            .into(),
                                });
                            }
                        };

                        let (block, params, extras) = ctx.branch_on_bit_array_prefix_pattern(
                            module,
                            pattern_block,
                            pattern_subjects[subject_index],
                            size_value,
                            next_block,
                            pattern_subjects.as_slice(),
                            subject_count,
                        )?;
                        pattern_block = block;
                        pattern_subjects = params;

                        if ctx.builder.current_block() != Some(pattern_block) {
                            ctx.builder.switch_to_block(pattern_block);
                        }

                        if extras.len() != 2 {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "unexpected bit array prefix extras in native case lowering"
                                        .into(),
                            });
                        }

                        let block_params = ctx.builder.block_params(pattern_block).to_vec();
                        let base_index = pattern_subjects.len();
                        let prefix_value = block_params.get(base_index).copied().ok_or_else(|| {
                            crate::Error::NativeCodegen {
                                message:
                                    "missing bit array prefix capture parameter in native case lowering"
                                        .into(),
                            }
                        })?;
                        let rest_value = block_params
                            .get(base_index + 1)
                            .copied()
                            .ok_or_else(|| crate::Error::NativeCodegen {
                                message:
                                    "missing bit array rest capture parameter in native case lowering"
                                        .into(),
                            })?;

                        if let Some(name) = prefix_binding {
                            bindings.push((
                                name,
                                BindingSource::BlockParam {
                                    index: base_index,
                                    value: prefix_value,
                                },
                            ));
                        }
                        if let Some(name) = rest_binding {
                            bindings.push((
                                name,
                                BindingSource::BlockParam {
                                    index: base_index + 1,
                                    value: rest_value,
                                },
                            ));
                        }

                        continue;
                    }

                    if rest_is_bits
                        && first_segment.type_.is_int()
                        && rest_segment.type_.is_bit_array()
                        && first_segment.size().is_none()
                        && first_segment.options.iter().all(|option| {
                            matches!(
                                option,
                                BitArrayOption::Int { .. }
                                    | BitArrayOption::Signed { .. }
                                    | BitArrayOption::Unsigned { .. }
                                    | BitArrayOption::Bytes { .. }
                            )
                        })
                    {
                        let first_binding = match first_segment.value.as_ref() {
                            Pattern::Variable { name, .. } => Some(name.clone()),
                            Pattern::Discard { .. } => None,
                            _ => {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "only variable or discard patterns are supported for byte segments in native functions"
                                            .into(),
                                });
                            }
                        };

                        let rest_binding = match rest_segment.value.as_ref() {
                            Pattern::Variable { name, .. } => Some(name.clone()),
                            Pattern::Discard { .. } => None,
                            _ => {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "only variable or discard patterns are supported for bits segments in native functions"
                                            .into(),
                                });
                            }
                        };

                        let (block, params, extras) = ctx.branch_on_bit_array_byte_pattern(
                            module,
                            pattern_block,
                            pattern_subjects[subject_index],
                            next_block,
                            pattern_subjects.as_slice(),
                            subject_count,
                        )?;
                        pattern_block = block;
                        pattern_subjects = params;

                        if ctx.builder.current_block() != Some(pattern_block) {
                            ctx.builder.switch_to_block(pattern_block);
                        }

                        if extras.len() != 2 {
                            return Err(crate::Error::NativeCodegen {
                                message: "unexpected byte pattern extras in native case lowering"
                                    .into(),
                            });
                        }

                        let block_params = ctx.builder.block_params(pattern_block).to_vec();
                        let base_index = pattern_subjects.len();
                        let byte_value =
                            block_params.get(base_index).copied().ok_or_else(|| {
                                crate::Error::NativeCodegen {
                                    message:
                                        "missing byte capture parameter in native case lowering"
                                            .into(),
                                }
                            })?;
                        let rest_value =
                            block_params.get(base_index + 1).copied().ok_or_else(|| {
                                crate::Error::NativeCodegen {
                                    message:
                                        "missing bits capture parameter in native case lowering"
                                            .into(),
                                }
                            })?;

                        if let Some(name) = first_binding {
                            bindings.push((
                                name,
                                BindingSource::BlockParam {
                                    index: base_index,
                                    value: byte_value,
                                },
                            ));
                        }
                        if let Some(name) = rest_binding {
                            bindings.push((
                                name,
                                BindingSource::BlockParam {
                                    index: base_index + 1,
                                    value: rest_value,
                                },
                            ));
                        }
                        continue;
                    }

                    let is_utf8_codepoint = first_segment
                        .options
                        .iter()
                        .all(|option| matches!(option, BitArrayOption::Utf8Codepoint { .. }));
                    let is_utf8_segment = first_segment
                        .options
                        .iter()
                        .all(|option| matches!(option, BitArrayOption::Utf8 { .. }));
                    let is_bytes = rest_segment
                        .options
                        .iter()
                        .all(|option| matches!(option, BitArrayOption::Bytes { .. }));

                    if (!is_utf8_codepoint && !is_utf8_segment) || !is_bytes {
                        return Err(crate::Error::NativeCodegen {
                            message: format!(
                                "bit array pattern options are not yet supported in native functions: {segments:?}"
                            ),
                        });
                    }

                    if is_utf8_segment
                        && !matches!(first_segment.value.as_ref(), Pattern::Discard { .. })
                    {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "utf8 bit array pattern segments must be discarded in native functions"
                                    .into(),
                        });
                    }

                    let first_binding = match first_segment.value.as_ref() {
                        Pattern::Variable { name, .. } => Some(name.clone()),
                        Pattern::Discard { .. } => None,
                        _ => {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "only variable or discard patterns are supported for utf8_codepoint segments in native functions"
                                        .into(),
                            });
                        }
                    };

                    let rest_binding = match rest_segment.value.as_ref() {
                        Pattern::Variable { name, .. } => Some(name.clone()),
                        Pattern::Discard { .. } => None,
                        _ => {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "only variable or discard patterns are supported for bytes segments in native functions"
                                        .into(),
                            });
                        }
                    };

                    let (block, params, extras) = ctx.branch_on_utf8_codepoint_pattern(
                        module,
                        pattern_block,
                        pattern_subjects[subject_index],
                        next_block,
                        pattern_subjects.as_slice(),
                        subject_count,
                    )?;
                    pattern_block = block;
                    pattern_subjects = params;

                    if ctx.builder.current_block() != Some(pattern_block) {
                        ctx.builder.switch_to_block(pattern_block);
                    }

                    if extras.len() != 2 {
                        return Err(crate::Error::NativeCodegen {
                            message: "unexpected utf8 pattern extras in native case lowering"
                                .into(),
                        });
                    }

                    let first_value = extras[0];
                    let rest_value = extras[1];

                    if let Some(name) = first_binding {
                        bindings.push((name, BindingSource::Value(first_value)));
                    }
                    if let Some(name) = rest_binding {
                        bindings.push((name, BindingSource::Value(rest_value)));
                    }
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
                                        "tuple pattern element `{other:?}` is not yet supported in native functions"
                                    ),
                                });
                            }
                        }
                    }

                    let (block, params, extras) = ctx.branch_on_tuple_pattern(
                        pattern_block,
                        pattern_subjects[subject_index],
                        elements.len(),
                        &capture_flags,
                        next_block,
                        pattern_subjects.as_slice(),
                        subject_count,
                    )?;
                    pattern_block = block;
                    pattern_subjects = params;

                    if ctx.builder.current_block() != Some(pattern_block) {
                        ctx.builder.switch_to_block(pattern_block);
                    }

                    let block_params = ctx.builder.block_params(pattern_block).to_vec();
                    let base_index = pattern_subjects.len();
                    let mut extras_iter = extras.into_iter();
                    let mut extra_position = 0usize;
                    for (capture_flag, binding_name) in
                        capture_flags.iter().zip(binding_names.iter())
                    {
                        if *capture_flag {
                            let Some(_value) = extras_iter.next() else {
                                return Err(crate::Error::NativeCodegen {
                                    message:
                                        "missing captured tuple element in native case lowering"
                                            .into(),
                                });
                            };
                            let param_index = base_index + extra_position;
                            let value =
                                block_params.get(param_index).copied().ok_or_else(|| {
                                    crate::Error::NativeCodegen {
                                    message:
                                        "missing tuple capture parameter in native case lowering"
                                            .into(),
                                }
                                })?;
                            extra_position += 1;
                            if let Some(name) = binding_name {
                                bindings.push((
                                    name.clone(),
                                    BindingSource::BlockParam {
                                        index: param_index,
                                        value,
                                    },
                                ));
                            }
                        }
                    }

                    if extras_iter.next().is_some() {
                        return Err(crate::Error::NativeCodegen {
                            message:
                                "unexpected extra tuple capture values in native case lowering"
                                    .into(),
                        });
                    }

                    continue;
                }
                other => {
                    eprintln!("UNSUPPORTED_PATTERN {:?}", other);
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
        let current_params = ctx.builder.block_params(pattern_block).to_vec();
        for (_, source) in &mut bindings {
            match source {
                BindingSource::Value(value) => {
                    if let Some(position) = current_params.iter().position(|param| *param == *value)
                    {
                        *value = current_params[position];
                    }
                }
                BindingSource::BlockParam { index, value } => {
                    if let Some(param) = current_params.get(*index) {
                        *value = *param;
                    }
                }
                BindingSource::Subject(_) => {}
            }
        }
        let mut final_subjects = ctx.builder.block_params(pattern_block).to_vec();

        if let Some(guard) = &clause.guard {
            ctx.push_scope();
            for (name, source) in &bindings {
                let value = match source {
                    BindingSource::Subject(index) => final_subjects[*index],
                    BindingSource::BlockParam { value, .. } => *value,
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
            ctx.seal_block(pattern_block);

            pattern_block = guard_success_block;
            let guard_params = ctx.builder.block_params(pattern_block).to_vec();

            let binding_names: Vec<EcoString> =
                bindings.iter().map(|(name, _)| name.clone()).collect();
            for (binding_index, (_, source)) in bindings.iter_mut().enumerate() {
                match source {
                    BindingSource::Value(value) => {
                        let Some(position) = guard_inputs.iter().position(|input| input == value)
                        else {
                            let binding_name = binding_names[binding_index].clone();
                            return Err(crate::Error::NativeCodegen {
                                message: format!(
                                    "missing captured value in native guard lowering (binding: {:?}, value: {:?}, inputs: {:?})",
                                    binding_name, value, guard_inputs
                                ),
                            });
                        };
                        *value = guard_params[position];
                    }
                    BindingSource::BlockParam { index, value } => {
                        let Some(param) = guard_params.get(*index) else {
                            let binding_name = binding_names[binding_index].clone();
                            return Err(crate::Error::NativeCodegen {
                                message: format!(
                                    "missing captured block param in native guard lowering (binding: {:?}, index: {:?}, params: {:?})",
                                    binding_name, index, guard_params
                                ),
                            });
                        };
                        *value = *param;
                    }
                    BindingSource::Subject(_) => {}
                }
            }

            final_subjects = guard_params;

            if ctx.builder.current_block() != Some(pattern_block) {
                ctx.builder.switch_to_block(pattern_block);
            }
        } else {
            final_subjects = ctx.builder.block_params(pattern_block).to_vec();
        }

        ctx.push_scope();
        for (name, source) in &bindings {
            let value = match source {
                BindingSource::Subject(index) => final_subjects[*index],
                BindingSource::BlockParam { value, .. } => *value,
                BindingSource::Value(value) => *value,
            };
            ctx.define(name, value);
        }
        let value = lower_expression(module, &clause.then, ctx)?;
        ctx.pop_scope();
        let _ = ctx.builder.ins().jump(exit_block, &[value]);
        ctx.seal_block(pattern_block);

        fallthrough = Some(next_block);

        if index + 1 == clauses.len() {
            break;
        }
    }

    if let Some(block) = fallthrough {
        ctx.builder.switch_to_block(block);
        ctx.seal_block(block);
        let _ = ctx.builder.ins().trap(TrapCode::User(0));
    }

    ctx.seal_block(exit_block);
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
            ctx.seal_block(right_block);

            ctx.builder.switch_to_block(exit_block);
            ctx.seal_block(exit_block);
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
        BinOp::Concatenate => {
            let left_value = lower_expression(module, left, ctx)?;
            let right_value = lower_expression(module, right, ctx)?;
            let func_id = ctx.declare_runtime_string_add(module)?;
            let func_ref = module.declare_func_in_func(func_id, &mut ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[left_value, right_value]);
            let results = ctx.builder.inst_results(call);
            Ok(results[0])
        }
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
    string_data: &mut HashMap<EcoString, DataId>,
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
            string_data,
            float_constants,
            record_constructors,
            module_functions,
            closure_counter,
        );
        lowering.mark_sealed(block);

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

    if let Err(err) = module.define_function(func_id, &mut ctx) {
        let clif = format!("{}", ctx.func.display());
        return Err(crate::Error::NativeCodegen {
            message: format!(
                "error lowering closure {module_name}.{closure_id}: {err} ({err:?})\n{clif}",
            ),
        });
    }
    module.clear_context(&mut ctx);

    Ok(func_id)
}

pub(super) fn lower_record_constructor_function(
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

    if let Err(err) = module.define_function(func_id, &mut ctx) {
        let clif = format!("{}", ctx.func.display());
        return Err(crate::Error::NativeCodegen {
            message: format!(
                "error lowering record constructor {module_name}.{constructor_module}.{variant_index}: {err} ({err:?})\n{clif}",
            ),
        });
    }
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
            &mut *ctx.string_data,
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
