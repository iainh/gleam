//! Shared lowering state used while building Cranelift functions.

use crate::{
    Result,
    ast::{ClauseGuard, Constant, TypedClauseGuard, TypedConstant, TypedExpr},
    type_::{PatternConstructor, Type, ValueConstructorVariant},
};
use cranelift_codegen::ir::{
    self, InstBuilder, MemFlags, StackSlotData, StackSlotKind, TrapCode, Value,
    condcodes::{FloatCC, IntCC},
};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::ObjectModule;
use ecow::EcoString;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    convert::TryFrom,
    mem::size_of,
    sync::Arc,
};

use super::expressions::{
    function_symbol_name, lower_expression, lower_record_constructor_function,
};
use super::patterns::{
    ConstructorTupleCondition, ConstructorTupleInfo, ListConstructorCondition, ListHeadMatch,
    NestedConstructorInfo, NestedListInfo,
};
use super::{
    BOOLEAN_FALSE_ARITY, BOOLEAN_TRUE_ARITY, BindingSource, FLOAT_HEADER, FunctionIdMap,
    HEADER_ARITY_SHIFT, HEADER_FIELD_MASK, HEADER_SIZE, TAG_BOOLEAN, TAG_FLOAT, TAG_LIST,
    TAG_RECORD, TAG_TUPLE, VALUE_TAG_MASK,
};

/// Mutable state threaded through expression lowering.
pub(super) struct LoweringContext<'a, 'b, 'c> {
    pub(super) builder: &'a mut FunctionBuilder<'b>,
    pub(super) pointer_type: ir::Type,
    pub(super) scopes: Vec<HashMap<EcoString, Value>>,
    pub(super) string_data: &'c mut HashMap<EcoString, DataId>,
    pub(super) float_constants: &'c mut HashMap<EcoString, DataId>,
    pub(super) zero_arity_records: &'c mut HashMap<(EcoString, u16), DataId>,
    pub(super) record_constructors: &'c mut HashMap<(EcoString, u16, u16), FuncId>,
    pub(super) module_functions: &'c mut HashMap<(EcoString, EcoString, usize), FuncId>,
    pub(super) closure_counter: &'c mut usize,
    pub(super) runtime_nil: Option<FuncId>,
    pub(super) runtime_alloc_tuple: Option<FuncId>,
    pub(super) runtime_binary_from_slice: Option<FuncId>,
    pub(super) runtime_print: Option<FuncId>,
    pub(super) runtime_println: Option<FuncId>,
    pub(super) runtime_print_error: Option<FuncId>,
    pub(super) runtime_println_error: Option<FuncId>,
    pub(super) runtime_string_add: Option<FuncId>,
    pub(super) runtime_string_equal: Option<FuncId>,
    pub(super) runtime_string_prefix_split: Option<FuncId>,
    pub(super) runtime_bit_array_utf8_split: Option<FuncId>,
    pub(super) runtime_bit_array_bit_size: Option<FuncId>,
    pub(super) runtime_bit_array_to_int: Option<FuncId>,
    pub(super) runtime_bit_array_pop_byte: Option<FuncId>,
    pub(super) runtime_bit_array_split_bits: Option<FuncId>,
    pub(super) runtime_bit_array_builder_new: Option<FuncId>,
    pub(super) runtime_bit_array_builder_append_int: Option<FuncId>,
    pub(super) runtime_bit_array_builder_append_bit_array: Option<FuncId>,
    pub(super) runtime_bit_array_builder_append_string_utf8: Option<FuncId>,
    pub(super) runtime_bit_array_builder_append_utf8_codepoint: Option<FuncId>,
    pub(super) runtime_bit_array_builder_finish: Option<FuncId>,
    pub(super) runtime_bool_true: Option<FuncId>,
    pub(super) runtime_bool_false: Option<FuncId>,
    pub(super) runtime_list_cons: Option<FuncId>,
    pub(super) runtime_int_negate: Option<FuncId>,
    pub(super) runtime_float_from_f64: Option<FuncId>,
    pub(super) runtime_alloc_closure: Option<FuncId>,
    pub(super) runtime_apply_closure: Option<FuncId>,
    pub(super) runtime_alloc_record: Option<FuncId>,
    pub(super) runtime_panic: Option<FuncId>,
    pub(super) runtime_gleeunit_main: Option<FuncId>,
    pub(super) runtime_gleeunit_do_main: Option<FuncId>,
    pub(super) pointer_bytes: u8,
    pub(super) functions: &'a FunctionIdMap,
    pub(super) module_name: &'a EcoString,
    pub(super) sealed_blocks: HashSet<ir::Block>,
}

impl<'a, 'b, 'c> LoweringContext<'a, 'b, 'c> {
    pub(super) fn expect_result(&mut self, inst: ir::Inst, context: &'static str) -> Value {
        self.builder
            .inst_results(inst)
            .first()
            .copied()
            .unwrap_or_else(|| panic!("native lowering: {context} produced no value"))
    }

    pub(super) fn expect_block_param(
        &mut self,
        block: ir::Block,
        index: usize,
        context: &'static str,
    ) -> Value {
        self.builder
            .block_params(block)
            .get(index)
            .copied()
            .unwrap_or_else(|| panic!("native lowering: {context} missing block param {index}"))
    }

    #[cfg(debug_assertions)]
    fn assert_block_open(&self, block: ir::Block, context: &str) {
        if let Some(inst) = self.builder.func.layout.last_inst(block) {
            if self.builder.func.dfg.insts[inst].opcode().is_terminator() {
                panic!(
                    "native lowering: attempted to insert into filled block {:?} while {}",
                    block, context
                );
            }
        }
    }

    pub(super) fn new(
        builder: &'a mut FunctionBuilder<'b>,
        pointer_type: ir::Type,
        pointer_bytes: u8,
        functions: &'a FunctionIdMap,
        module_name: &'a EcoString,
        zero_arity_records: &'c mut HashMap<(EcoString, u16), DataId>,
        string_data: &'c mut HashMap<EcoString, DataId>,
        float_constants: &'c mut HashMap<EcoString, DataId>,
        record_constructors: &'c mut HashMap<(EcoString, u16, u16), FuncId>,
        module_functions: &'c mut HashMap<(EcoString, EcoString, usize), FuncId>,
        closure_counter: &'c mut usize,
    ) -> Self {
        Self {
            builder,
            pointer_type,
            scopes: vec![HashMap::new()],
            string_data,
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
            runtime_string_add: None,
            runtime_string_equal: None,
            runtime_string_prefix_split: None,
            runtime_bit_array_utf8_split: None,
            runtime_bit_array_bit_size: None,
            runtime_bit_array_to_int: None,
            runtime_bit_array_pop_byte: None,
            runtime_bit_array_split_bits: None,
            runtime_bit_array_builder_new: None,
            runtime_bit_array_builder_append_int: None,
            runtime_bit_array_builder_append_bit_array: None,
            runtime_bit_array_builder_append_string_utf8: None,
            runtime_bit_array_builder_append_utf8_codepoint: None,
            runtime_bit_array_builder_finish: None,
            runtime_bool_true: None,
            runtime_bool_false: None,
            runtime_list_cons: None,
            runtime_int_negate: None,
            runtime_float_from_f64: None,
            runtime_alloc_closure: None,
            runtime_apply_closure: None,
            runtime_alloc_record: None,
            runtime_panic: None,
            runtime_gleeunit_main: None,
            runtime_gleeunit_do_main: None,
            pointer_bytes,
            functions,
            module_name,
            sealed_blocks: HashSet::new(),
        }
    }

    pub(super) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub(super) fn pop_scope(&mut self) {
        let _ = self.scopes.pop();
    }

    pub(super) fn mark_sealed(&mut self, block: ir::Block) {
        let _ = self.sealed_blocks.insert(block);
    }

    pub(super) fn seal_block(&mut self, block: ir::Block) {
        if self.sealed_blocks.insert(block) {
            self.builder.seal_block(block);
        }
    }

    pub(super) fn define(&mut self, name: &EcoString, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            let _ = scope.insert(name.clone(), value);
        }
    }

    pub(super) fn lookup(&self, name: &EcoString) -> Option<&Value> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    pub(super) fn capture_environment_values(&self) -> Vec<(EcoString, Value)> {
        let mut map = BTreeMap::new();
        for scope in self.scopes.iter().rev() {
            for (name, value) in scope {
                let _ = map.entry(name.clone()).or_insert(*value);
            }
        }
        map.into_iter().collect()
    }

    pub(super) fn next_closure_id(&mut self) -> usize {
        let id = *self.closure_counter;
        *self.closure_counter += 1;
        id
    }

    pub(super) fn load_float(&mut self, value: Value) -> Value {
        let mem_flags = MemFlags::trusted();
        self.builder
            .ins()
            .load(ir::types::F64, mem_flags, value, HEADER_SIZE)
    }

    pub(super) fn bool_from_condition(
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

    pub(super) fn apply_closure(
        &mut self,
        module: &mut ObjectModule,
        fun_value: Value,
        arguments: &[Value],
    ) -> Result<Value> {
        let args_ptr = if arguments.is_empty() {
            self.builder.ins().iconst(self.pointer_type, 0)
        } else {
            let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                (arguments.len() * self.pointer_bytes()) as u32,
            ));
            for (index, value) in arguments.iter().enumerate() {
                let offset = (index * self.pointer_bytes()) as i32;
                let _ = self.builder.ins().stack_store(*value, slot, offset);
            }
            self.builder.ins().stack_addr(self.pointer_type, slot, 0)
        };

        let argc = self
            .builder
            .ins()
            .iconst(self.pointer_type, arguments.len() as i64);

        let apply_func = self.declare_runtime_apply_closure(module)?;
        let apply_ref = module.declare_func_in_func(apply_func, self.builder.func);
        let call = self
            .builder
            .ins()
            .call(apply_ref, &[fun_value, args_ptr, argc]);
        let result = self.expect_result(call, "apply closure");
        Ok(result)
    }

    pub(super) fn lower_constant(
        &mut self,
        module: &mut ObjectModule,
        constant: &TypedConstant,
    ) -> Result<Value> {
        match constant {
            Constant::Int { int_value, .. } => {
                let value = int_value
                    .to_i64()
                    .ok_or_else(|| crate::Error::NativeCodegen {
                        message: "integer literal out of range for 64-bit backend".into(),
                    })?;
                let encoded = encode_small_int(value)?;
                Ok(self.builder.ins().iconst(ir::types::I64, encoded))
            }
            Constant::Float { value, .. } => self.float_constant(module, value),
            Constant::String { value, .. } => self.string_constant(module, value.as_str()),
            Constant::Tuple { elements, .. } => {
                if elements.is_empty() {
                    let func_id = self.declare_runtime_nil(module)?;
                    let func_ref = module.declare_func_in_func(func_id, self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[]);
                    return Ok(self.expect_result(call, "nil constructor"));
                }

                let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (elements.len() * self.pointer_bytes()) as u32,
                ));

                for (index, element) in elements.iter().enumerate() {
                    let value = self.lower_constant(module, element)?;
                    let offset = (index * self.pointer_bytes()) as i32;
                    let _ = self.builder.ins().stack_store(value, slot, offset);
                }

                let base_ptr = self.builder.ins().stack_addr(self.pointer_type, slot, 0);
                let len_value = self
                    .builder
                    .ins()
                    .iconst(self.pointer_type, elements.len() as i64);

                let func_id = self.declare_runtime_alloc_tuple(module)?;
                let func_ref = module.declare_func_in_func(func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[base_ptr, len_value]);
                Ok(self.expect_result(call, "tuple allocation"))
            }
            Constant::List { elements, .. } => {
                let mut values = Vec::with_capacity(elements.len());
                for element in elements {
                    values.push(self.lower_constant(module, element)?);
                }

                let mut current = {
                    let func_id = self.declare_runtime_nil(module)?;
                    let func_ref = module.declare_func_in_func(func_id, self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[]);
                    self.expect_result(call, "list nil constructor")
                };

                if !values.is_empty() {
                    let func_id = self.declare_runtime_list_cons(module)?;
                    let func_ref = module.declare_func_in_func(func_id, self.builder.func);
                    for value in values.into_iter().rev() {
                        let call = self.builder.ins().call(func_ref, &[value, current]);
                        current = self.expect_result(call, "list cons");
                    }
                }

                Ok(current)
            }
            Constant::Record {
                arguments,
                record_constructor,
                ..
            } => {
                let constructor =
                    record_constructor
                        .as_ref()
                        .ok_or_else(|| crate::Error::NativeCodegen {
                            message: "missing record constructor in constant".into(),
                        })?;
                match &constructor.variant {
                    ValueConstructorVariant::Record {
                        module: constructor_module,
                        variant_index,
                        arity,
                        ..
                    } => {
                        let mut values = Vec::with_capacity(arguments.len());
                        for argument in arguments {
                            values.push(self.lower_constant(module, &argument.value)?);
                        }

                        if usize::from(*arity) != values.len() {
                            return Err(crate::Error::NativeCodegen {
                                message: format!(
                                    "record constant arity mismatch: expected {arity} arguments, got {}",
                                    values.len()
                                ),
                            });
                        }

                        if *arity == 0 {
                            self.zero_arity_record_constant(
                                module,
                                constructor_module,
                                *variant_index,
                            )
                        } else {
                            let fun_value = self.record_constructor_value(
                                module,
                                constructor_module,
                                *variant_index,
                                *arity,
                            )?;
                            self.apply_closure(module, fun_value, &values)
                        }
                    }
                    ValueConstructorVariant::ModuleConstant { literal, .. } => {
                        self.lower_constant(module, literal)
                    }
                    ValueConstructorVariant::LocalConstant { literal } => {
                        self.lower_constant(module, literal)
                    }
                    other => Err(crate::Error::NativeCodegen {
                        message: format!(
                            "record constant with unsupported constructor variant `{other:?}`"
                        ),
                    }),
                }
            }
            Constant::Var {
                name, constructor, ..
            } => {
                if let Some(constructor) = constructor {
                    match &constructor.variant {
                        ValueConstructorVariant::ModuleConstant { literal, .. } => {
                            self.lower_constant(module, literal)
                        }
                        ValueConstructorVariant::LocalConstant { literal } => {
                            self.lower_constant(module, literal)
                        }
                        ValueConstructorVariant::Record {
                            module: constructor_module,
                            variant_index,
                            arity,
                            ..
                        } => {
                            if *arity == 0 {
                                self.zero_arity_record_constant(
                                    module,
                                    constructor_module,
                                    *variant_index,
                                )
                            } else {
                                self.record_constructor_value(
                                    module,
                                    constructor_module,
                                    *variant_index,
                                    *arity,
                                )
                            }
                        }
                        ValueConstructorVariant::ModuleFn {
                            module: function_module,
                            name: function_name,
                            arity,
                            ..
                        } => self.module_function_value(
                            module,
                            function_module,
                            function_name,
                            *arity,
                        ),
                        ValueConstructorVariant::LocalVariable { .. } => self
                            .lookup(name)
                            .copied()
                            .ok_or_else(|| crate::Error::NativeCodegen {
                                message: format!("unknown variable `{name}` in native constant"),
                            }),
                    }
                } else {
                    self.lookup(name)
                        .copied()
                        .ok_or_else(|| crate::Error::NativeCodegen {
                            message: format!("unknown variable `{name}` in native constant"),
                        })
                }
            }
            Constant::StringConcatenation { left, right, .. } => {
                if let (Some(left), Some(right)) =
                    (constant_string_value(left), constant_string_value(right))
                {
                    let mut combined = left;
                    combined.push_str(&right);
                    self.string_constant(module, &combined)
                } else {
                    Err(crate::Error::NativeCodegen {
                        message: "unsupported string concatenation constant in native backend"
                            .into(),
                    })
                }
            }
            Constant::BitArray { .. } => Err(crate::Error::NativeCodegen {
                message: "bit array constants are not yet supported in native functions".into(),
            }),
            Constant::Invalid { .. } => Err(crate::Error::NativeCodegen {
                message: "invalid constant encountered during native lowering".into(),
            }),
        }
    }

    pub(super) fn ensure_record_constructor_function(
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

    pub(super) fn record_constructor_value(
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
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let code_ptr = self.builder.ins().func_addr(self.pointer_type, func_ref);
        let env_ptr = self.builder.ins().iconst(self.pointer_type, 0);
        let env_len = self.builder.ins().iconst(self.pointer_type, 0);
        let alloc_func = self.declare_runtime_alloc_closure(module)?;
        let alloc_ref = module.declare_func_in_func(alloc_func, self.builder.func);
        let call = self
            .builder
            .ins()
            .call(alloc_ref, &[code_ptr, env_ptr, env_len]);
        Ok(self.expect_result(call, "record constructor closure"))
    }

    pub(super) fn ensure_module_function(
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

    pub(super) fn module_function_value(
        &mut self,
        module: &mut ObjectModule,
        function_module: &EcoString,
        function_name: &EcoString,
        arity: usize,
    ) -> Result<Value> {
        let func_id = self.ensure_module_function(module, function_module, function_name, arity)?;
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let code_ptr = self.builder.ins().func_addr(self.pointer_type, func_ref);
        let env_ptr = self.builder.ins().iconst(self.pointer_type, 0);
        let env_len = self.builder.ins().iconst(self.pointer_type, 0);
        let alloc_func = self.declare_runtime_alloc_closure(module)?;
        let alloc_ref = module.declare_func_in_func(alloc_func, self.builder.func);
        let call = self
            .builder
            .ins()
            .call(alloc_ref, &[code_ptr, env_ptr, env_len]);
        let results = self.builder.inst_results(call);
        Ok(results[0])
    }

    pub(super) fn try_call_function(
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

        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &args);
        let results = self.builder.inst_results(call);
        Ok(Some(results[0]))
    }

    pub(super) fn string_constant(
        &mut self,
        module: &mut ObjectModule,
        text: &str,
    ) -> Result<Value> {
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

        let gv = module.declare_data_in_func(data_id, self.builder.func);
        Ok(self.builder.ins().global_value(self.pointer_type, gv))
    }

    pub(super) fn float_constant(
        &mut self,
        module: &mut ObjectModule,
        literal: &str,
    ) -> Result<Value> {
        let key: EcoString = literal.into();
        let data_id = if let Some(id) = self.float_constants.get(&key) {
            *id
        } else {
            let cleaned = literal.replace('_', "");
            let number = cleaned
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

        let gv = module.declare_data_in_func(data_id, self.builder.func);
        Ok(self.builder.ins().global_value(self.pointer_type, gv))
    }

    pub(super) fn declare_runtime_nil(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_alloc_tuple(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_list_cons(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_alloc_record(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_panic(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
        if let Some(id) = self.runtime_panic {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(ir::types::I64));

        let id = module
            .declare_function("gleam_panic", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_panic = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_binary_from_slice(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_print(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_println(&mut self, module: &mut ObjectModule) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_print_error(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_println_error(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_string_add(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_string_add {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("add", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_string_add = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_string_equal(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_string_equal {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("string_eq", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_string_equal = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_string_prefix_split(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_string_prefix_split {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("string_prefix_split", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_string_prefix_split = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_utf8_split(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_utf8_split {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_pop_utf8_codepoint", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_utf8_split = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_bit_size(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_bit_size {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_bit_size", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_bit_size = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_to_int(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_to_int {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_to_int_and_size", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_to_int = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_pop_byte(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_pop_byte {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_pop_byte", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_pop_byte = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_split_bits(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_split_bits {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_split_bits", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_split_bits = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_builder_new(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_builder_new {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_builder_new", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_builder_new = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_builder_append_int(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_builder_append_int {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        for _ in 0..8 {
            signature.params.push(ir::AbiParam::new(self.pointer_type));
        }
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_builder_append_int", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_builder_append_int = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_builder_append_bit_array(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_builder_append_bit_array {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        for _ in 0..5 {
            signature.params.push(ir::AbiParam::new(self.pointer_type));
        }
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function(
                "bit_array_builder_append_bit_array",
                Linkage::Import,
                &signature,
            )
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_builder_append_bit_array = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_builder_append_string_utf8(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_builder_append_string_utf8 {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function(
                "bit_array_builder_append_string_utf8",
                Linkage::Import,
                &signature,
            )
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_builder_append_string_utf8 = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_builder_append_utf8_codepoint(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_builder_append_utf8_codepoint {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function(
                "bit_array_builder_append_utf8_codepoint",
                Linkage::Import,
                &signature,
            )
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_builder_append_utf8_codepoint = Some(id);
        Ok(id)
    }

    pub(super) fn declare_runtime_bit_array_builder_finish(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_bit_array_builder_finish {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("bit_array_builder_finish", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_bit_array_builder_finish = Some(id);
        Ok(id)
    }

    pub(super) fn branch_on_empty_bit_array_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject: Value,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>)> {
        let success_block = self.create_subject_block(subject_count);

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }

        let func_id = self.declare_runtime_bit_array_bit_size(module)?;
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let size_value = self.expect_result(call, "call result");

        let zero_encoded = encode_small_int(0)?;
        let zero_value = self.builder.ins().iconst(self.pointer_type, zero_encoded);
        let is_empty = self
            .builder
            .ins()
            .icmp(IntCC::Equal, size_value, zero_value);

        let args = failure_args.to_vec();
        let _ = self
            .builder
            .ins()
            .brif(is_empty, success_block, &args, failure_block, &args);
        self.seal_block(current_block);

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        Ok((success_block, params))
    }

    pub(super) fn branch_on_sized_int_bit_array_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject: Value,
        size_bits: i64,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Value)> {
        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        let _ = self
            .builder
            .append_block_param(success_block, self.pointer_type);

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }

        let func_id = self.declare_runtime_bit_array_to_int(module)?;
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let tuple = self.expect_result(call, "call result");

        let value = self.tuple_element(tuple, 0)?;
        let size_value = self.tuple_element(tuple, 1)?;
        let expected_size = self
            .builder
            .ins()
            .iconst(self.pointer_type, encode_small_int(size_bits)?);
        let size_matches = self
            .builder
            .ins()
            .icmp(IntCC::Equal, size_value, expected_size);

        let args = failure_args.to_vec();
        let mut success_args = args.clone();
        success_args.push(value);
        let _ = self.builder.ins().brif(
            size_matches,
            success_block,
            &success_args,
            failure_block,
            &args,
        );
        self.seal_block(current_block);

        self.builder.switch_to_block(success_block);
        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        let subjects = params[..subject_count].to_vec();
        let captured = params[subject_count];
        Ok((success_block, subjects, captured))
    }

    pub(super) fn branch_on_bit_array_byte_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject: Value,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        let extract_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        for _ in 0..2 {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }

        let func_id = self.declare_runtime_bit_array_pop_byte(module)?;
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let result = self.expect_result(call, "call result");

        let flag = self.tuple_element(result, 0)?;
        let true_value = self.bool_constant(module, true)?;
        let condition = self.builder.ins().icmp(IntCC::Equal, flag, true_value);

        let _ = self
            .builder
            .ins()
            .brif(condition, extract_block, &[], failure_block, failure_args);
        self.seal_block(current_block);

        self.builder.switch_to_block(extract_block);
        let first_value = self.tuple_element(result, 1)?;
        let rest_value = self.tuple_element(result, 2)?;
        let mut success_args = failure_args.to_vec();
        success_args.push(first_value);
        success_args.push(rest_value);
        let _ = self.builder.ins().jump(success_block, &success_args);
        self.seal_block(extract_block);

        self.builder.switch_to_block(success_block);
        let params = self.builder.block_params(success_block).to_vec();
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    pub(super) fn branch_on_bit_array_prefix_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject: Value,
        size_value: Value,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        let extract_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        for _ in 0..2 {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }

        let func_id = self.declare_runtime_bit_array_split_bits(module)?;
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject, size_value]);
        let result = self.expect_result(call, "call result");

        let flag = self.tuple_element(result, 0)?;
        let true_value = self.bool_constant(module, true)?;
        let condition = self.builder.ins().icmp(IntCC::Equal, flag, true_value);

        let _ = self
            .builder
            .ins()
            .brif(condition, extract_block, &[], failure_block, failure_args);
        self.seal_block(current_block);

        self.builder.switch_to_block(extract_block);
        let prefix_value = self.tuple_element(result, 1)?;
        let rest_value = self.tuple_element(result, 2)?;
        let mut success_args = failure_args.to_vec();
        success_args.push(prefix_value);
        success_args.push(rest_value);
        let _ = self.builder.ins().jump(success_block, &success_args);
        self.seal_block(extract_block);

        self.builder.switch_to_block(success_block);
        let params = self.builder.block_params(success_block).to_vec();
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    pub(super) fn declare_runtime_bool_true(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_bool_false(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_int_negate(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_float_from_f64(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_alloc_closure(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_apply_closure(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn bool_constant(
        &mut self,
        module: &mut ObjectModule,
        value: bool,
    ) -> Result<Value> {
        let func_id = if value {
            self.declare_runtime_bool_true(module)?
        } else {
            self.declare_runtime_bool_false(module)?
        };
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[]);
        let results = self.builder.inst_results(call);
        Ok(results[0])
    }

    pub(super) fn zero_arity_record_constant(
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

        let gv = module.declare_data_in_func(data_id, self.builder.func);
        Ok(self.builder.ins().global_value(self.pointer_type, gv))
    }

    pub(super) fn create_subject_block(&mut self, subject_count: usize) -> ir::Block {
        let block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self.builder.append_block_param(block, self.pointer_type);
        }
        block
    }

    pub(super) fn branch_on_int_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        int_value: &BigInt,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>)> {
        let success_block = self.create_subject_block(subject_count);

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }

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
        self.seal_block(current_block);

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        Ok((success_block, params))
    }

    pub(super) fn branch_on_float_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        float_value: f64,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>)> {
        let success_block = self.create_subject_block(subject_count);

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }
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
        self.seal_block(current_block);

        self.builder.switch_to_block(pointer_block);
        let pointer_subject = self.expect_block_param(pointer_block, 0, "block param");
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
        self.seal_block(pointer_block);

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
        self.seal_block(float_block);

        let params = self.builder.block_params(success_block).to_vec();
        Ok((success_block, params))
    }

    pub(super) fn branch_on_string_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject: Value,
        string_value: Value,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>)> {
        let success_block = self.create_subject_block(subject_count);

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }
        let args = failure_args.to_vec();

        let eq_func = self.declare_runtime_string_equal(module)?;
        let eq_ref = module.declare_func_in_func(eq_func, self.builder.func);
        let compare_call = self.builder.ins().call(eq_ref, &[subject, string_value]);
        let compare_value = self.expect_result(compare_call, "call result");

        let true_func = self.declare_runtime_bool_true(module)?;
        let true_ref = module.declare_func_in_func(true_func, self.builder.func);
        let true_call = self.builder.ins().call(true_ref, &[]);
        let true_value = self.expect_result(true_call, "call result");

        let is_equal = self
            .builder
            .ins()
            .icmp(IntCC::Equal, compare_value, true_value);

        let _ = self
            .builder
            .ins()
            .brif(is_equal, success_block, &args, failure_block, &args);
        self.seal_block(current_block);

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        Ok((success_block, params))
    }

    pub(super) fn branch_on_string_prefix_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject: Value,
        prefix_value: Value,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        let extract_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        for _ in 0..2 {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }

        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }
        let failure_values = failure_args.to_vec();
        let func_id = self.declare_runtime_string_prefix_split(module)?;
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject, prefix_value]);
        let result = self.expect_result(call, "call result");

        let flag = self.tuple_element(result, 0)?;
        let true_value = self.bool_constant(module, true)?;
        let condition = self.builder.ins().icmp(IntCC::Equal, flag, true_value);

        let _ = self.builder.ins().brif(
            condition,
            extract_block,
            &[],
            failure_block,
            &failure_values,
        );
        self.seal_block(current_block);

        self.builder.switch_to_block(extract_block);
        let matched_prefix = self.tuple_element(result, 1)?;
        let rest = self.tuple_element(result, 2)?;
        let mut success_args = failure_args.to_vec();
        success_args.push(matched_prefix);
        success_args.push(rest);
        let _ = self.builder.ins().jump(success_block, &success_args);
        self.seal_block(extract_block);

        self.builder.switch_to_block(success_block);
        let params = self.builder.block_params(success_block).to_vec();
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    pub(super) fn branch_on_utf8_codepoint_pattern(
        &mut self,
        module: &mut ObjectModule,
        current_block: ir::Block,
        subject: Value,
        failure_block: ir::Block,
        failure_args: &[Value],
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        let extract_block = self.builder.create_block();
        let success_block = self.builder.create_block();
        for _ in 0..subject_count {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }
        for _ in 0..2 {
            let _ = self
                .builder
                .append_block_param(success_block, self.pointer_type);
        }

        self.builder.switch_to_block(current_block);
        let args = failure_args.to_vec();
        let func_id = self.declare_runtime_bit_array_utf8_split(module)?;
        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let result = self.expect_result(call, "call result");

        let flag = self.tuple_element(result, 0)?;
        let true_value = self.bool_constant(module, true)?;
        let condition = self.builder.ins().icmp(IntCC::Equal, flag, true_value);

        let _ = self
            .builder
            .ins()
            .brif(condition, extract_block, &[], failure_block, &args);
        self.seal_block(current_block);

        self.builder.switch_to_block(extract_block);
        let codepoint = self.tuple_element(result, 1)?;
        let rest = self.tuple_element(result, 2)?;
        let mut success_args = failure_args.to_vec();
        success_args.push(codepoint);
        success_args.push(rest);
        let _ = self.builder.ins().jump(success_block, &success_args);
        self.seal_block(extract_block);

        self.builder.switch_to_block(success_block);
        let params = self.builder.block_params(success_block).to_vec();
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    pub(super) fn lower_clause_guard_condition(
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
            ClauseGuard::And { left, right, .. } => {
                let left = self.lower_clause_guard_condition(module, left)?;
                let right = self.lower_clause_guard_condition(module, right)?;
                Ok(self.builder.ins().band(left, right))
            }
            ClauseGuard::Or { left, right, .. } => {
                let left = self.lower_clause_guard_condition(module, left)?;
                let right = self.lower_clause_guard_condition(module, right)?;
                Ok(self.builder.ins().bor(left, right))
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

    pub(super) fn lower_clause_guard_operand(
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

    pub(super) fn lower_clause_guard_constant(
        &mut self,
        module: &mut ObjectModule,
        constant: &TypedConstant,
    ) -> Result<Value> {
        self.lower_constant(module, constant)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn branch_on_constructor_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        constructor: &PatternConstructor,
        type_: &Arc<Type>,
        capture_flags: &[bool],
        alias_counts: &[usize],
        failure_block: ir::Block,
        failure_args: &[Value],
        failure_block_arg_count: usize,
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }

        let actual_failure_arg_count = self.builder.block_params(failure_block).len();
        debug_assert_eq!(
            actual_failure_arg_count, failure_block_arg_count,
            "constructor failure arg count mismatch: expected {}, actual {}",
            failure_block_arg_count, actual_failure_arg_count
        );
        let failure_block_arg_count = actual_failure_arg_count;
        #[cfg(debug_assertions)]
        eprintln!(
            "native lowering: constructor branch received failure_args_len={}, subject_count={}, failure_block_args={}",
            failure_args.len(),
            subject_count,
            failure_block_arg_count
        );

        if capture_flags.len() != alias_counts.len() {
            return Err(crate::Error::NativeCodegen {
                message:
                    "constructor capture flag and alias count length mismatch in native lowering"
                        .into(),
            });
        }

        if failure_block_arg_count > failure_args.len() {
            return Err(crate::Error::NativeCodegen {
                message: format!(
                    "constructor pattern requested {failure_block_arg_count} failure arguments, got {}",
                    failure_args.len()
                )
                .into(),
            });
        }

        if failure_args.len() < subject_count {
            #[cfg(debug_assertions)]
            eprintln!(
                "native lowering: constructor failure_args_len={}, subject_count={}, failure_block_args={}",
                failure_args.len(),
                subject_count,
                failure_block_arg_count
            );
            return Err(crate::Error::NativeCodegen {
                message: format!(
                    "constructor pattern mismatch: subjects={}, available={}",
                    subject_count,
                    failure_args.len()
                )
                .into(),
            });
        }

        let captures: usize = capture_flags.iter().filter(|flag| **flag).count();
        let alias_total: usize = alias_counts.iter().sum();
        let extra_count = captures + alias_total;

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
        success_args.extend(failure_args.iter().take(subject_count).copied());
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

        let failure_values = failure_args[..failure_block_arg_count].to_vec();
        let _ = self.builder.ins().brif(
            is_boxed,
            pointer_block,
            &[subject],
            failure_block,
            &failure_values,
        );
        self.seal_block(current_block);

        self.builder.switch_to_block(pointer_block);
        let pointer_subject = self.expect_block_param(pointer_block, 0, "block param");
        let header = self
            .builder
            .ins()
            .load(self.pointer_type, mem_flags, pointer_subject, 0);
        let _ = self
            .builder
            .ins()
            .jump(tag_block, &[pointer_subject, header]);
        self.seal_block(pointer_block);

        self.builder.switch_to_block(tag_block);
        let tag_subject = self.expect_block_param(tag_block, 0, "block param");
        let tag_header = self.expect_block_param(tag_block, 1, "block param");
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
                &failure_values,
            );
            self.seal_block(tag_block);
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
                &failure_values,
            );
            self.seal_block(tag_block);

            self.builder.switch_to_block(record_block);
            let record_subject = self.expect_block_param(record_block, 0, "block param");
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
                    let alias_count = alias_counts[field_index];
                    if !*capture {
                        if alias_count > 0 {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "constructor alias requires capturing the argument in native functions"
                                        .into(),
                            });
                        }
                        continue;
                    }

                    let field_offset = field_base + (field_index as i32) * pointer_stride;
                    let field_value = self.builder.ins().load(
                        self.pointer_type,
                        mem_flags,
                        record_subject,
                        field_offset,
                    );
                    captured_fields.push(field_value);
                    for _ in 0..alias_count {
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
                &failure_values,
            );
            self.seal_block(record_block);
        }

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        self.seal_block(success_block);
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn branch_on_tuple_pattern(
        &mut self,
        current_block: ir::Block,
        subject: Value,
        arity: usize,
        capture_flags: &[bool],
        failure_block: ir::Block,
        failure_args: &[Value],
        failure_block_arg_count: usize,
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        if self.builder.current_block() != Some(current_block) {
            self.builder.switch_to_block(current_block);
        }

        let actual_failure_arg_count = self.builder.block_params(failure_block).len();
        debug_assert_eq!(
            actual_failure_arg_count, failure_block_arg_count,
            "tuple failure arg count mismatch: expected {}, actual {}",
            failure_block_arg_count, actual_failure_arg_count
        );
        let failure_block_arg_count = actual_failure_arg_count;

        if failure_block_arg_count > failure_args.len() {
            return Err(crate::Error::NativeCodegen {
                message: format!(
                    "tuple pattern requested {failure_block_arg_count} failure arguments, got {}",
                    failure_args.len()
                )
                .into(),
            });
        }

        if failure_args.len() < subject_count {
            #[cfg(debug_assertions)]
            eprintln!(
                "native lowering: tuple success needs {subject_count} subjects, failure args available {}",
                failure_args.len()
            );
            return Err(crate::Error::NativeCodegen {
                message: format!(
                    "tuple pattern requires {subject_count} subject arguments, got {}",
                    failure_args.len()
                )
                .into(),
            });
        }

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
        success_args.extend(failure_args.iter().take(subject_count).copied());
        let mem_flags = MemFlags::trusted();

        let pointer_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(pointer_block, self.pointer_type);

        let value_tag_mask = self.builder.ins().iconst(self.pointer_type, VALUE_TAG_MASK);
        let boxed_check = self.builder.ins().band(subject, value_tag_mask);
        let zero = self.builder.ins().iconst(self.pointer_type, 0);
        let is_boxed = self.builder.ins().icmp(IntCC::Equal, boxed_check, zero);

        let failure_values = failure_args[..failure_block_arg_count].to_vec();
        let _ = self.builder.ins().brif(
            is_boxed,
            pointer_block,
            &[subject],
            failure_block,
            &failure_values,
        );
        self.seal_block(current_block);

        self.builder.switch_to_block(pointer_block);
        let tuple_subject = self.expect_block_param(pointer_block, 0, "block param");
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
            &failure_values,
        );
        self.seal_block(pointer_block);

        self.builder.switch_to_block(tuple_block);
        let tuple_subject = self.expect_block_param(tuple_block, 0, "block param");

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
        self.seal_block(tuple_block);

        let params = self.builder.func.dfg.block_params(success_block).to_vec();
        self.seal_block(success_block);
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn branch_on_list_pattern(
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
        failure_block_arg_count: usize,
        subject_count: usize,
    ) -> Result<(ir::Block, Vec<Value>, Vec<Value>)> {
        if failure_block_arg_count > failure_args.len() {
            return Err(crate::Error::NativeCodegen {
                message: "requested failure argument count exceeds available subjects".into(),
            });
        }

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
            if self.builder.current_block() != Some(current_block) {
                #[cfg(debug_assertions)]
                self.assert_block_open(current_block, "entering list head loop");
                self.builder.switch_to_block(current_block);
            }

            let nil_func = self.declare_runtime_nil(module)?;
            let nil_ref = module.declare_func_in_func(nil_func, self.builder.func);
            let nil_call = self.builder.ins().call(nil_ref, &[]);
            let nil_value = self.expect_result(nil_call, "call result");

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
            let failure_values: Vec<_> = subjects[..failure_block_arg_count].to_vec();
            let _ = self.builder.ins().brif(
                is_nil,
                failure_block,
                &failure_values,
                non_nil_block,
                &subjects,
            );
            self.seal_block(current_block);

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
            let failure_values: Vec<_> = subjects[..failure_block_arg_count].to_vec();
            let _ = self.builder.ins().brif(
                is_boxed,
                pointer_block,
                &subjects,
                failure_block,
                &failure_values,
            );
            self.seal_block(current_block);

            #[cfg(debug_assertions)]
            self.assert_block_open(pointer_block, "list pattern pointer block entry");
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
            let failure_values: Vec<_> = subjects[..failure_block_arg_count].to_vec();
            let _ = self.builder.ins().brif(
                is_list,
                list_block,
                &subjects,
                failure_block,
                &failure_values,
            );
            self.seal_block(current_block);

            #[cfg(debug_assertions)]
            self.assert_block_open(list_block, "list pattern list block entry");
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
                    ListHeadMatch::Constructor(info) => {
                        let alias_counts = vec![0; info.capture_flags.len()];
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "native lowering: list head constructor subjects={}, failure_params={}",
                            subjects.len(),
                            failure_block_arg_count
                        );
                        self.branch_on_constructor_pattern(
                            current_block,
                            head_value,
                            info.constructor,
                            info.type_,
                            info.capture_flags.as_slice(),
                            alias_counts.as_slice(),
                            failure_block,
                            subjects.as_slice(),
                            failure_block_arg_count,
                            subjects.len(),
                        )?
                    }
                    ListHeadMatch::Tuple(info) => self.branch_on_tuple_pattern(
                        current_block,
                        head_value,
                        info.arity,
                        info.capture_flags.as_slice(),
                        failure_block,
                        subjects.as_slice(),
                        failure_block_arg_count,
                        subjects.len(),
                    )?,
                };
                current_block = block;
                subjects = params;
                current_subject = subjects[subject_index];
                head_extras = extras;
                if self.builder.current_block() != Some(current_block) {
                    #[cfg(debug_assertions)]
                    self.assert_block_open(current_block, "after lowering list head pattern");
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
                match pattern {
                    ListHeadMatch::Constructor(info) => {
                        let mut extras_iter = head_extras.into_iter();
                        for (index, capture_flag) in info.capture_flags.iter().enumerate() {
                            if *capture_flag {
                                let Some(value) = extras_iter.next() else {
                                    return Err(crate::Error::NativeCodegen {
                                        message:
                                            "missing head capture value in native list pattern lowering"
                                                .into(),
                                    });
                                };

                                match info
                                    .conditions
                                    .get(index)
                                    .unwrap_or(&ListConstructorCondition::None)
                                {
                                    ListConstructorCondition::None => {}
                                    ListConstructorCondition::String(expected) => {
                                        let expected_value =
                                            self.string_constant(module, expected.as_str())?;
                                        let func_id = self.declare_runtime_string_equal(module)?;
                                        let func_ref =
                                            module.declare_func_in_func(func_id, self.builder.func);
                                        let call = self
                                            .builder
                                            .ins()
                                            .call(func_ref, &[value, expected_value]);
                                        let result = self.expect_result(call, "call result");
                                        let true_value = self.bool_constant(module, true)?;
                                        let is_equal = self.builder.ins().icmp(
                                            IntCC::Equal,
                                            result,
                                            true_value,
                                        );

                                        let continue_block =
                                            self.create_subject_block(subject_count);
                                        let failure_values: Vec<_> =
                                            subjects[..failure_block_arg_count].to_vec();
                                        let success_values = subjects.clone();
                                        let _ = self.builder.ins().brif(
                                            is_equal,
                                            continue_block,
                                            success_values.as_slice(),
                                            failure_block,
                                            &failure_values,
                                        );
                                        self.seal_block(current_block);
                                        self.builder.switch_to_block(continue_block);
                                        current_block = continue_block;
                                        subjects =
                                            self.builder.block_params(current_block).to_vec();
                                    }
                                    ListConstructorCondition::EmptyList => {
                                        let nil_func = self.declare_runtime_nil(module)?;
                                        let nil_ref = module
                                            .declare_func_in_func(nil_func, self.builder.func);
                                        let nil_call = self.builder.ins().call(nil_ref, &[]);
                                        let nil_value = self.expect_result(nil_call, "call result");
                                        let is_nil =
                                            self.builder.ins().icmp(IntCC::Equal, value, nil_value);

                                        let continue_block =
                                            self.create_subject_block(subject_count);
                                        let failure_values: Vec<_> =
                                            subjects[..failure_block_arg_count].to_vec();
                                        let success_values = subjects.clone();
                                        let _ = self.builder.ins().brif(
                                            is_nil,
                                            continue_block,
                                            success_values.as_slice(),
                                            failure_block,
                                            &failure_values,
                                        );
                                        self.seal_block(current_block);
                                        self.builder.switch_to_block(continue_block);
                                        current_block = continue_block;
                                        subjects =
                                            self.builder.block_params(current_block).to_vec();
                                    }
                                }

                                if let Some(slot) = storage_slot {
                                    let offset = (stored * pointer_bytes) as i32;
                                    let _ = self.builder.ins().stack_store(value, slot, offset);
                                }
                                stored += 1;
                            }
                        }
                        if extras_iter.next().is_some() {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "unexpected extra constructor capture values in native list pattern lowering"
                                        .into(),
                            });
                        }
                    }
                    ListHeadMatch::Tuple(info) => {
                        let mut extras_iter = head_extras.into_iter();
                        for capture_flag in info.capture_flags.iter() {
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
                        if extras_iter.next().is_some() {
                            return Err(crate::Error::NativeCodegen {
                                message:
                                    "unexpected extra tuple capture values in native list pattern lowering"
                                        .into(),
                            });
                        }
                    }
                }
            }

            subjects[subject_index] = tail;
            current_subject = tail;
        }

        if ensure_exact {
            if self.builder.current_block() != Some(current_block) {
                #[cfg(debug_assertions)]
                self.assert_block_open(current_block, "ensuring exact list tail");
                self.builder.switch_to_block(current_block);
            }

            let nil_func = self.declare_runtime_nil(module)?;
            let nil_ref = module.declare_func_in_func(nil_func, self.builder.func);
            let nil_call = self.builder.ins().call(nil_ref, &[]);
            let nil_value = self.expect_result(nil_call, "call result");

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
            let failure_values: Vec<_> = subjects[..failure_block_arg_count].to_vec();
            let _ = self.builder.ins().brif(
                is_nil,
                exact_block,
                &subjects,
                failure_block,
                &failure_values,
            );
            self.seal_block(current_block);

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
        self.seal_block(current_block);
        self.builder.switch_to_block(success_block);
        #[cfg(debug_assertions)]
        self.assert_block_open(success_block, "before reading list success params");
        let params = self.builder.block_params(success_block).to_vec();
        self.seal_block(success_block);
        let new_subjects = params[..subject_count].to_vec();
        let extras = params[subject_count..].to_vec();
        Ok((success_block, new_subjects, extras))
    }

    pub(super) fn lower_nested_list<'pattern>(
        &mut self,
        module: &mut ObjectModule,
        pattern_block: &mut ir::Block,
        pattern_subjects: &mut Vec<Value>,
        head_value: Value,
        info: &NestedListInfo<'pattern>,
        failure_block: ir::Block,
        bindings: &mut Vec<(EcoString, BindingSource)>,
    ) -> Result<()> {
        let mut nested_args = pattern_subjects.clone();
        nested_args.push(head_value);
        let nested_index = nested_args.len() - 1;

        let failure_arg_count = self.builder.block_params(failure_block).len();
        let (block, params, extras) = self.branch_on_list_pattern(
            module,
            *pattern_block,
            nested_index,
            info.pattern.capture_heads.as_slice(),
            info.pattern.head_matches.as_slice(),
            info.pattern.capture_tail,
            info.pattern.ensure_exact,
            failure_block,
            nested_args.as_slice(),
            failure_arg_count,
            nested_args.len(),
        )?;
        *pattern_block = block;
        let mut new_subjects = params;
        let _ = new_subjects.pop();
        *pattern_subjects = new_subjects;

        if self.builder.current_block() != Some(*pattern_block) {
            self.builder.switch_to_block(*pattern_block);
        }

        let mut extra_iter = extras.into_iter();
        for index in 0..info.pattern.capture_heads.len() {
            let capture = info.pattern.capture_heads[index];
            let name = &info.pattern.head_names[index];
            let field_bindings = &info.pattern.head_field_bindings[index];
            let extra_aliases = &info.pattern.head_extra_bindings[index];
            let nested = &info.pattern.head_nested_lists[index];
            if capture {
                let Some(value) = extra_iter.next() else {
                    return Err(crate::Error::NativeCodegen {
                        message: "missing captured nested list head value in native case lowering"
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
                    self.lower_nested_list(
                        module,
                        pattern_block,
                        pattern_subjects,
                        value,
                        nested_info,
                        failure_block,
                        bindings,
                    )?;
                    if self.builder.current_block() != Some(*pattern_block) {
                        self.builder.switch_to_block(*pattern_block);
                    }
                }
            } else if nested.is_some() {
                return Err(crate::Error::NativeCodegen {
                    message: "nested list pattern requires head capture in native case lowering"
                        .into(),
                });
            }

            if let Some(binding_names) = field_bindings {
                let Some(info) = info.pattern.head_matches[index].as_ref() else {
                    return Err(crate::Error::NativeCodegen {
                        message:
                            "missing nested head match info for list bindings in native lowering"
                                .into(),
                    });
                };

                if info.capture_flags().len() != binding_names.len() {
                    return Err(crate::Error::NativeCodegen {
                        message:
                            "nested list head binding length mismatch in native list pattern lowering"
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
                                    "missing captured nested constructor field in native case lowering"
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

        if info.pattern.capture_tail {
            if let Some(name) = &info.pattern.tail_name {
                let Some(value) = extra_iter.next() else {
                    return Err(crate::Error::NativeCodegen {
                        message: "missing captured nested list tail value in native case lowering"
                            .into(),
                    });
                };
                bindings.push((name.clone(), BindingSource::Value(value)));
            }
        }

        if extra_iter.next().is_some() {
            return Err(crate::Error::NativeCodegen {
                message:
                    "unexpected extra values after lowering nested list pattern in native backend"
                        .into(),
            });
        }

        Ok(())
    }

    pub(super) fn lower_nested_constructor(
        &mut self,
        _module: &mut ObjectModule,
        pattern_block: &mut ir::Block,
        pattern_subjects: &mut Vec<Value>,
        constructor_value: Value,
        info: &NestedConstructorInfo<'_>,
        failure_block: ir::Block,
        bindings: &mut Vec<(EcoString, BindingSource)>,
    ) -> Result<()> {
        let mut nested_subjects = pattern_subjects.clone();
        nested_subjects.push(constructor_value);
        let failure_block_arg_count = self.builder.block_params(failure_block).len();
        let alias_counts = vec![0; info.capture_flags.len()];
        let subject_count = nested_subjects.len();
        let (block, params, extras) = self.branch_on_constructor_pattern(
            *pattern_block,
            constructor_value,
            info.constructor,
            info.type_,
            info.capture_flags.as_slice(),
            alias_counts.as_slice(),
            failure_block,
            nested_subjects.as_slice(),
            failure_block_arg_count,
            subject_count,
        )?;
        *pattern_block = block;
        let mut new_subjects = params;
        let _ = new_subjects.pop();
        *pattern_subjects = new_subjects;

        if self.builder.current_block() != Some(*pattern_block) {
            self.builder.switch_to_block(*pattern_block);
        }

        let mut extras_iter = extras.into_iter();
        for (capture_flag, binding_name) in info.capture_flags.iter().zip(info.binding_names.iter())
        {
            if *capture_flag {
                let Some(value) = extras_iter.next() else {
                    return Err(crate::Error::NativeCodegen {
                        message: "missing nested constructor capture in native case lowering"
                            .into(),
                    });
                };
                if let Some(name) = binding_name {
                    bindings.push((name.clone(), BindingSource::Value(value)));
                }
            }
        }

        if extras_iter.next().is_some() {
            return Err(crate::Error::NativeCodegen {
                message:
                    "unexpected extra values after lowering nested constructor pattern in native backend"
                        .into(),
            });
        }

        Ok(())
    }

    pub(super) fn ensure_zero_arity_constructor(
        &mut self,
        module: &mut ObjectModule,
        pattern_block: &mut ir::Block,
        pattern_subjects: &mut Vec<Value>,
        value: Value,
        constructor: &PatternConstructor,
        type_: &Arc<Type>,
        failure_block: ir::Block,
    ) -> Result<()> {
        let expected = if type_.is_bool() {
            let bool_value = match constructor.name.as_str() {
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
            self.bool_constant(module, bool_value)?
        } else {
            self.zero_arity_record_constant(
                module,
                &constructor.module,
                constructor.constructor_index,
            )?
        };

        let is_equal = self.builder.ins().icmp(IntCC::Equal, value, expected);
        let subject_count = pattern_subjects.len();
        let continue_block = self.create_subject_block(subject_count);
        let failure_args = pattern_subjects.clone();
        let success_args = pattern_subjects.clone();
        let _ = self.builder.ins().brif(
            is_equal,
            continue_block,
            success_args.as_slice(),
            failure_block,
            &failure_args,
        );
        self.seal_block(*pattern_block);
        self.builder.switch_to_block(continue_block);
        *pattern_block = continue_block;
        *pattern_subjects = self.builder.block_params(continue_block).to_vec();
        Ok(())
    }

    pub(super) fn lower_nested_constructor_assignment(
        &mut self,
        module: &mut ObjectModule,
        pattern_block: &mut ir::Block,
        pattern_subjects: &mut Vec<Value>,
        constructor_value: Value,
        info: &NestedConstructorInfo<'_>,
        failure_block: ir::Block,
    ) -> Result<Vec<(EcoString, Value)>> {
        let mut nested_bindings = Vec::new();
        self.lower_nested_constructor(
            module,
            pattern_block,
            pattern_subjects,
            constructor_value,
            info,
            failure_block,
            &mut nested_bindings,
        )?;

        let mut results = Vec::with_capacity(nested_bindings.len());
        for (name, source) in nested_bindings {
            let value = match source {
                BindingSource::Value(value) => value,
                BindingSource::BlockParam { value, .. } => value,
                BindingSource::Subject(index) => pattern_subjects
                    .get(index)
                    .copied()
                    .ok_or_else(|| crate::Error::NativeCodegen {
                        message:
                            "subject index out of bounds while lowering nested constructor assignment"
                                .into(),
                    })?,
            };
            results.push((name, value));
        }

        Ok(results)
    }

    pub(super) fn ensure_int_value(
        &mut self,
        pattern_block: &mut ir::Block,
        pattern_subjects: &mut Vec<Value>,
        value: Value,
        int_value: &BigInt,
        failure_block: ir::Block,
    ) -> Result<()> {
        let number = int_value
            .to_i64()
            .ok_or_else(|| crate::Error::NativeCodegen {
                message: format!("integer literal out of range for Gleam immediate: {int_value}"),
            })?;
        let encoded = encode_small_int(number)?;
        let expected = self.builder.ins().iconst(self.pointer_type, encoded);
        let is_equal = self.builder.ins().icmp(IntCC::Equal, value, expected);

        let subject_count = pattern_subjects.len();
        let continue_block = self.create_subject_block(subject_count);
        let failure_args = pattern_subjects.clone();
        let success_args = pattern_subjects.clone();
        let _ = self.builder.ins().brif(
            is_equal,
            continue_block,
            success_args.as_slice(),
            failure_block,
            &failure_args,
        );
        self.seal_block(*pattern_block);
        self.builder.switch_to_block(continue_block);
        *pattern_block = continue_block;
        *pattern_subjects = self.builder.block_params(continue_block).to_vec();
        Ok(())
    }

    pub(super) fn ensure_string_value(
        &mut self,
        module: &mut ObjectModule,
        pattern_block: &mut ir::Block,
        pattern_subjects: &mut Vec<Value>,
        value: Value,
        string_literal: &str,
        failure_block: ir::Block,
    ) -> Result<()> {
        let literal = self.string_constant(module, string_literal)?;
        let (block, params) = self.branch_on_string_pattern(
            module,
            *pattern_block,
            value,
            literal,
            failure_block,
            pattern_subjects.as_slice(),
            pattern_subjects.len(),
        )?;
        *pattern_block = block;
        *pattern_subjects = params;
        if self.builder.current_block() != Some(*pattern_block) {
            self.builder.switch_to_block(*pattern_block);
        }
        Ok(())
    }

    pub(super) fn lower_constructor_tuple(
        &mut self,
        module: &mut ObjectModule,
        pattern_block: &mut ir::Block,
        pattern_subjects: &mut Vec<Value>,
        tuple_value: Value,
        info: &ConstructorTupleInfo,
        failure_block: ir::Block,
        bindings: &mut Vec<(EcoString, BindingSource)>,
    ) -> Result<()> {
        let subject_count = pattern_subjects.len();
        let failure_block_arg_count = self.builder.block_params(failure_block).len();
        if failure_block_arg_count > subject_count {
            return Err(crate::Error::NativeCodegen {
                message: format!(
                    "constructor tuple pattern requested {failure_block_arg_count} failure arguments, got {subject_count}"
                )
                .into(),
            });
        }
        let (block, params, extras) = self.branch_on_tuple_pattern(
            *pattern_block,
            tuple_value,
            info.capture_flags.len(),
            info.capture_flags.as_slice(),
            failure_block,
            &pattern_subjects[..failure_block_arg_count],
            failure_block_arg_count,
            subject_count,
        )?;
        *pattern_block = block;
        *pattern_subjects = params;

        if self.builder.current_block() != Some(*pattern_block) {
            self.builder.switch_to_block(*pattern_block);
        }

        let mut extra_iter = extras.into_iter();
        for ((capture, name), condition) in info
            .capture_flags
            .iter()
            .zip(info.binding_names.iter())
            .zip(info.conditions.iter())
        {
            if *capture {
                let Some(value) = extra_iter.next() else {
                    return Err(crate::Error::NativeCodegen {
                        message: "missing tuple constructor element capture in native lowering"
                            .into(),
                    });
                };

                match condition {
                    ConstructorTupleCondition::None => {}
                    ConstructorTupleCondition::String(expected) => {
                        let expected_value = self.string_constant(module, expected.as_str())?;
                        let func_id = self.declare_runtime_string_equal(module)?;
                        let func_ref = module.declare_func_in_func(func_id, self.builder.func);
                        let call = self.builder.ins().call(func_ref, &[value, expected_value]);
                        let result = self.expect_result(call, "call result");
                        let true_value = self.bool_constant(module, true)?;
                        let is_equal = self.builder.ins().icmp(IntCC::Equal, result, true_value);

                        let continue_block = self.create_subject_block(pattern_subjects.len());
                        let failure_args = pattern_subjects.clone();
                        let success_args = pattern_subjects.clone();
                        let _ = self.builder.ins().brif(
                            is_equal,
                            continue_block,
                            success_args.as_slice(),
                            failure_block,
                            &failure_args,
                        );
                        self.seal_block(*pattern_block);
                        self.builder.switch_to_block(continue_block);
                        *pattern_block = continue_block;
                        *pattern_subjects = self.builder.block_params(continue_block).to_vec();
                    }
                }

                if let Some(name) = name {
                    bindings.push((name.clone(), BindingSource::Value(value)));
                }
            }
        }

        if extra_iter.next().is_some() {
            return Err(crate::Error::NativeCodegen {
                message: "unexpected extra tuple constructor capture values in native lowering"
                    .into(),
            });
        }

        Ok(())
    }

    pub(super) fn tuple_element(&mut self, tuple: Value, index: u64) -> Result<Value> {
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
        let pointer_subject = self.expect_block_param(pointer_block, 0, "block param");
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
        self.seal_block(pointer_block);

        self.builder.switch_to_block(tuple_block);
        let tuple_ptr = self.expect_block_param(tuple_block, 0, "block param");
        let tuple_header = self.expect_block_param(tuple_block, 1, "block param");
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
        self.seal_block(tuple_block);

        self.builder.switch_to_block(failure_block);
        let _ = self.builder.ins().trap(TrapCode::User(0));
        self.seal_block(failure_block);

        self.builder.switch_to_block(element_block);
        let tuple_ptr = self.expect_block_param(element_block, 0, "block param");
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
        self.seal_block(element_block);
        Ok(element)
    }

    pub(super) fn record_field(&mut self, record: Value, index: u64) -> Result<Value> {
        let pointer_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(pointer_block, self.pointer_type);
        let failure_block = self.builder.create_block();

        let mem_flags = MemFlags::trusted();
        let value_tag_mask = self.builder.ins().iconst(self.pointer_type, VALUE_TAG_MASK);
        let boxed_check = self.builder.ins().band(record, value_tag_mask);
        let zero = self.builder.ins().iconst(self.pointer_type, 0);
        let is_boxed = self.builder.ins().icmp(IntCC::Equal, boxed_check, zero);

        let _ = self
            .builder
            .ins()
            .brif(is_boxed, pointer_block, &[record], failure_block, &[]);

        self.builder.switch_to_block(pointer_block);
        let record_ptr = self.expect_block_param(pointer_block, 0, "block param");
        let header = self
            .builder
            .ins()
            .load(self.pointer_type, mem_flags, record_ptr, 0);
        let header_mask = self
            .builder
            .ins()
            .iconst(self.pointer_type, HEADER_FIELD_MASK);
        let header_tag = self.builder.ins().band(header, header_mask);
        let record_tag = self.builder.ins().iconst(self.pointer_type, TAG_RECORD);
        let is_record = self
            .builder
            .ins()
            .icmp(IntCC::Equal, header_tag, record_tag);

        let record_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(record_block, self.pointer_type);
        let _ = self
            .builder
            .append_block_param(record_block, self.pointer_type);
        let _ = self.builder.ins().brif(
            is_record,
            record_block,
            &[record_ptr, header],
            failure_block,
            &[],
        );
        self.seal_block(pointer_block);

        self.builder.switch_to_block(record_block);
        let record_ptr = self.expect_block_param(record_block, 0, "block param");
        let record_header = self.expect_block_param(record_block, 1, "block param");
        let header_mask = self
            .builder
            .ins()
            .iconst(self.pointer_type, HEADER_FIELD_MASK);
        let arity_shifted = self
            .builder
            .ins()
            .ushr_imm(record_header, HEADER_ARITY_SHIFT);
        let arity_value = self.builder.ins().band(arity_shifted, header_mask);

        let index_i64 = i64::try_from(index).map_err(|_| crate::Error::NativeCodegen {
            message: "record index exceeds native backend limits".into(),
        })?;
        let index_value = self.builder.ins().iconst(self.pointer_type, index_i64);
        let in_bounds = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, index_value, arity_value);

        let field_block = self.builder.create_block();
        let _ = self
            .builder
            .append_block_param(field_block, self.pointer_type);
        let _ = self
            .builder
            .ins()
            .brif(in_bounds, field_block, &[record_ptr], failure_block, &[]);
        self.seal_block(record_block);

        self.builder.switch_to_block(failure_block);
        let _ = self.builder.ins().trap(TrapCode::User(0));
        self.seal_block(failure_block);

        self.builder.switch_to_block(field_block);
        let record_ptr = self.expect_block_param(field_block, 0, "block param");
        let index_usize = usize::try_from(index).map_err(|_| crate::Error::NativeCodegen {
            message: "record index exceeds native backend limits".into(),
        })?;
        let index_i32 = i32::try_from(index_usize).map_err(|_| crate::Error::NativeCodegen {
            message: "record field offset exceeds native backend limits".into(),
        })?;
        let pointer_stride = self.pointer_bytes() as i32;
        let base_offset = HEADER_SIZE + (2 * size_of::<u32>() as i32);
        let offset = base_offset + index_i32 * pointer_stride;
        let field = self
            .builder
            .ins()
            .load(self.pointer_type, mem_flags, record_ptr, offset);
        self.seal_block(field_block);
        Ok(field)
    }

    pub(super) fn declare_runtime_gleeunit_main(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn declare_runtime_gleeunit_do_main(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
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

    pub(super) fn pointer_bytes(&self) -> usize {
        self.pointer_bytes as usize
    }
}

fn constant_string_value(constant: &TypedConstant) -> Option<String> {
    match constant {
        Constant::String { value, .. } => Some(value.to_string()),
        Constant::StringConcatenation { left, right, .. } => {
            let mut left_value = constant_string_value(left)?;
            let right_value = constant_string_value(right)?;
            left_value.push_str(&right_value);
            Some(left_value)
        }
        Constant::Var {
            constructor: Some(constructor),
            ..
        } => match &constructor.variant {
            ValueConstructorVariant::ModuleConstant { literal, .. } => {
                constant_string_value(literal)
            }
            ValueConstructorVariant::LocalConstant { literal } => constant_string_value(literal),
            _ => None,
        },
        _ => None,
    }
}

/// Encodes an i63 immediate using the runtime tagging scheme.
pub(super) fn encode_small_int(value: i64) -> Result<i64, crate::Error> {
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
