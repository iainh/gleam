//! Converts typed Gleam AST into Cranelift IR.

use std::collections::HashMap;

use cranelift_codegen::ir::Value;
use cranelift_module::FuncId;
use ecow::EcoString;

pub(super) use super::config::ModuleConfig;

pub(super) const VALUE_TAG_MASK: i64 = 0b11;
pub(super) const HEADER_FIELD_MASK: i64 = 0xFFFF;
pub(super) const HEADER_ARITY_SHIFT: i64 = 16;
pub(super) const HEADER_SIZE: i32 = 8;
pub(super) const TAG_FLOAT: u64 = 1;
pub(super) const TAG_LIST: i64 = 4;
pub(super) const TAG_TUPLE: i64 = 5;
pub(super) const TAG_RECORD: i64 = 6;
pub(super) const TAG_BOOLEAN: i64 = 12;
pub(super) const BOOLEAN_FALSE_ARITY: i64 = 0;
pub(super) const BOOLEAN_TRUE_ARITY: i64 = 1;
pub(super) const FLOAT_HEADER: u64 = (1u64 << 32) | TAG_FLOAT;

pub(super) type FunctionIdMap = HashMap<(EcoString, usize), FuncId>;

/// Describes the origin of a value bound during pattern lowering so it can be
/// materialised later.
#[derive(Clone, Copy)]
pub(super) enum BindingSource {
    Subject(usize),
    /// References a block parameter by stable index so captures survive block rewrites
    BlockParam {
        index: usize,
        value: Value,
    },
    Value(Value),
}

mod context;
mod expressions;
mod patterns;

pub(crate) use expressions::lower_module_functions;
