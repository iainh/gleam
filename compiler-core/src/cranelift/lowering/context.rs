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
    ConstructorTupleCondition, ConstructorTupleInfo, ListHeadMatch, NestedListInfo,
};
use super::{
    BOOLEAN_FALSE_ARITY, BOOLEAN_TRUE_ARITY, BindingSource, FLOAT_HEADER, FunctionIdMap,
    HEADER_ARITY_SHIFT, HEADER_FIELD_MASK, HEADER_SIZE, TAG_BOOLEAN, TAG_FLOAT, TAG_LIST,
    TAG_RECORD, TAG_TUPLE, VALUE_TAG_MASK,
};

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
    pub(super) runtime_string_utf8_bits: Option<FuncId>,
    pub(super) runtime_bit_array_utf8_split: Option<FuncId>,
    pub(super) runtime_bit_array_bit_size: Option<FuncId>,
    pub(super) runtime_bit_array_to_int: Option<FuncId>,
    pub(super) runtime_bit_array_pop_byte: Option<FuncId>,
    pub(super) runtime_bit_array_split_bits: Option<FuncId>,
    pub(super) runtime_bool_true: Option<FuncId>,
    pub(super) runtime_bool_false: Option<FuncId>,
    pub(super) runtime_list_cons: Option<FuncId>,
    pub(super) runtime_int_negate: Option<FuncId>,
    pub(super) runtime_float_from_f64: Option<FuncId>,
    pub(super) runtime_alloc_closure: Option<FuncId>,
    pub(super) runtime_apply_closure: Option<FuncId>,
    pub(super) runtime_alloc_record: Option<FuncId>,
    pub(super) runtime_gleeunit_main: Option<FuncId>,
    pub(super) runtime_gleeunit_do_main: Option<FuncId>,
    pub(super) pointer_bytes: u8,
    pub(super) functions: &'a FunctionIdMap,
    pub(super) module_name: &'a EcoString,
    pub(super) sealed_blocks: HashSet<ir::Block>,
}

impl<'a, 'b, 'c> LoweringContext<'a, 'b, 'c> {
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
            runtime_string_utf8_bits: None,
            runtime_bit_array_utf8_split: None,
            runtime_bit_array_bit_size: None,
            runtime_bit_array_to_int: None,
            runtime_bit_array_pop_byte: None,
            runtime_bit_array_split_bits: None,
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
        let apply_ref = module.declare_func_in_func(apply_func, &mut self.builder.func);
        let call = self
            .builder
            .ins()
            .call(apply_ref, &[fun_value, args_ptr, argc]);
        let results = self.builder.inst_results(call);
        Ok(results[0])
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
                    let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[]);
                    let results = self.builder.inst_results(call);
                    return Ok(results[0]);
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
                let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
                let call = self.builder.ins().call(func_ref, &[base_ptr, len_value]);
                let results = self.builder.inst_results(call);
                Ok(results[0])
            }
            Constant::List { elements, .. } => {
                let mut values = Vec::with_capacity(elements.len());
                for element in elements {
                    values.push(self.lower_constant(module, element)?);
                }

                let mut current = {
                    let func_id = self.declare_runtime_nil(module)?;
                    let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
                    let call = self.builder.ins().call(func_ref, &[]);
                    self.builder.inst_results(call)[0]
                };

                if !values.is_empty() {
                    let func_id = self.declare_runtime_list_cons(module)?;
                    let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
                    for value in values.into_iter().rev() {
                        let call = self.builder.ins().call(func_ref, &[value, current]);
                        let results = self.builder.inst_results(call);
                        current = results[0];
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

        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
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

        let gv = module.declare_data_in_func(data_id, &mut self.builder.func);
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

    pub(super) fn declare_runtime_string_utf8_bits(
        &mut self,
        module: &mut ObjectModule,
    ) -> Result<FuncId> {
        if let Some(id) = self.runtime_string_utf8_bits {
            return Ok(id);
        }

        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(self.pointer_type));
        signature.returns.push(ir::AbiParam::new(self.pointer_type));

        let id = module
            .declare_function("string_to_utf8_bits", Linkage::Import, &signature)
            .map_err(|err| crate::Error::NativeCodegen {
                message: err.to_string(),
            })?;
        self.runtime_string_utf8_bits = Some(id);
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
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let size_value = self.builder.inst_results(call)[0];

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
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let tuple = self.builder.inst_results(call)[0];

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
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let result = self.builder.inst_results(call)[0];

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
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject, size_value]);
        let result = self.builder.inst_results(call)[0];

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
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
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

        let gv = module.declare_data_in_func(data_id, &mut self.builder.func);
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
        let eq_ref = module.declare_func_in_func(eq_func, &mut self.builder.func);
        let compare_call = self.builder.ins().call(eq_ref, &[subject, string_value]);
        let compare_value = self.builder.inst_results(compare_call)[0];

        let true_func = self.declare_runtime_bool_true(module)?;
        let true_ref = module.declare_func_in_func(true_func, &mut self.builder.func);
        let true_call = self.builder.ins().call(true_ref, &[]);
        let true_value = self.builder.inst_results(true_call)[0];

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
        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
        let call = self.builder.ins().call(func_ref, &[subject]);
        let result = self.builder.inst_results(call)[0];

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
        self.seal_block(current_block);

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
        self.seal_block(pointer_block);

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
                failure_args,
            );
            self.seal_block(tag_block);

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
        self.seal_block(current_block);

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
        self.seal_block(pointer_block);

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
                self.builder.switch_to_block(current_block);
            }

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
            if self.builder.current_block() != Some(current_block) {
                self.builder.switch_to_block(current_block);
            }

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
            pattern_subjects.len(),
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
        let (block, params, extras) = self.branch_on_tuple_pattern(
            *pattern_block,
            tuple_value,
            info.capture_flags.len(),
            info.capture_flags.as_slice(),
            failure_block,
            pattern_subjects.as_slice(),
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
                        let func_ref = module.declare_func_in_func(func_id, &mut self.builder.func);
                        let call = self.builder.ins().call(func_ref, &[value, expected_value]);
                        let result = self.builder.inst_results(call)[0];
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
        self.seal_block(pointer_block);

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
        self.seal_block(tuple_block);

        self.builder.switch_to_block(failure_block);
        let _ = self.builder.ins().trap(TrapCode::User(0));
        self.seal_block(failure_block);

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
        self.seal_block(element_block);
        Ok(element)
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
