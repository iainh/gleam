//! Pattern analysis structures shared between list and constructor lowering.

use crate::{
    Result,
    ast::Pattern,
    type_::{PatternConstructor, Type},
};
use ecow::EcoString;
use std::sync::Arc;

/// Metadata about a list constructor pattern used to drive lowering.
#[derive(Debug)]
pub(super) struct ListConstructorInfo<'a> {
    pub(super) constructor: &'a PatternConstructor,
    pub(super) type_: &'a Arc<Type>,
    pub(super) capture_flags: Vec<bool>,
    pub(super) conditions: Vec<ListConstructorCondition>,
}

/// Describes a tuple pattern matched against a list head.
#[derive(Debug)]
pub(super) struct ListTupleInfo {
    pub(super) arity: usize,
    pub(super) capture_flags: Vec<bool>,
}

/// Represents the possible matchers that can consume a list head.
#[derive(Debug)]
pub(super) enum ListHeadMatch<'a> {
    Constructor(ListConstructorInfo<'a>),
    Tuple(ListTupleInfo),
}

impl<'a> ListHeadMatch<'a> {
    pub(super) fn capture_flags(&self) -> &[bool] {
        match self {
            ListHeadMatch::Constructor(info) => &info.capture_flags,
            ListHeadMatch::Tuple(info) => &info.capture_flags,
        }
    }
}

/// Additional runtime guards that must hold for a list head match.
#[derive(Debug)]
pub(super) enum ListConstructorCondition {
    None,
    String(EcoString),
    EmptyList,
}

/// Aggregated information about an entire list pattern.
#[derive(Debug)]
pub(super) struct ListPatternInfo<'a> {
    pub(super) capture_heads: Vec<bool>,
    pub(super) head_names: Vec<Option<EcoString>>,
    pub(super) head_matches: Vec<Option<ListHeadMatch<'a>>>,
    pub(super) head_field_bindings: Vec<Option<Vec<Option<EcoString>>>>,
    pub(super) head_extra_bindings: Vec<Vec<EcoString>>,
    pub(super) head_nested_lists: Vec<Option<NestedListInfo<'a>>>,
    pub(super) capture_tail: bool,
    pub(super) tail_name: Option<EcoString>,
    pub(super) ensure_exact: bool,
}

/// Recursive detail for nested list patterns.
#[derive(Debug)]
pub(super) struct NestedListInfo<'a> {
    pub(super) pattern: Box<ListPatternInfo<'a>>,
}

/// Metadata for tuple constructors used within patterns.
#[derive(Debug)]
pub(super) struct ConstructorTupleInfo {
    pub(super) capture_flags: Vec<bool>,
    pub(super) binding_names: Vec<Option<EcoString>>,
    pub(super) conditions: Vec<ConstructorTupleCondition>,
}

/// Guards applied to constructor tuple matches.
#[derive(Debug)]
pub(super) enum ConstructorTupleCondition {
    None,
    String(EcoString),
}

/// Captures nested constructor details to reuse matches.
#[derive(Debug)]
pub(super) struct NestedConstructorInfo<'a> {
    pub(super) constructor: &'a PatternConstructor,
    pub(super) type_: &'a Arc<Type>,
    pub(super) capture_flags: Vec<bool>,
    pub(super) binding_names: Vec<Option<EcoString>>,
}

/// Analyses a list pattern to drive the lowering of pattern matching logic.
pub(super) fn collect_list_pattern_info<'a>(
    elements: &'a [Pattern<Arc<Type>>],
    tail: Option<&'a Pattern<Arc<Type>>>,
) -> Result<ListPatternInfo<'a>> {
    fn analyse_list_element<'a>(
        element: &'a Pattern<Arc<Type>>,
        capture_heads: &mut Vec<bool>,
        head_names: &mut Vec<Option<EcoString>>,
        head_matches: &mut Vec<Option<ListHeadMatch<'a>>>,
        head_field_bindings: &mut Vec<Option<Vec<Option<EcoString>>>>,
        head_extra_bindings: &mut Vec<Vec<EcoString>>,
        head_nested_lists: &mut Vec<Option<NestedListInfo<'a>>>,
    ) -> Result<()> {
        match element {
            Pattern::Assign { name, pattern, .. } => {
                let before = capture_heads.len();
                analyse_list_element(
                    pattern,
                    capture_heads,
                    head_names,
                    head_matches,
                    head_field_bindings,
                    head_extra_bindings,
                    head_nested_lists,
                )?;
                if capture_heads.len() == before {
                    return Err(crate::Error::NativeCodegen {
                        message: "assign pattern did not produce list head in native lowering"
                            .into(),
                    });
                }
                let index = capture_heads.len() - 1;
                if !capture_heads[index] {
                    capture_heads[index] = true;
                    head_names[index] = Some(name.clone());
                } else if head_names[index].is_none() {
                    head_names[index] = Some(name.clone());
                } else {
                    head_extra_bindings[index].push(name.clone());
                }
                Ok(())
            }
            Pattern::Variable { name, .. } => {
                capture_heads.push(true);
                head_names.push(Some(name.clone()));
                head_matches.push(None);
                head_field_bindings.push(None);
                head_extra_bindings.push(Vec::new());
                head_nested_lists.push(None);
                Ok(())
            }
            Pattern::Discard { .. } => {
                capture_heads.push(false);
                head_names.push(None);
                head_matches.push(None);
                head_field_bindings.push(None);
                head_extra_bindings.push(Vec::new());
                head_nested_lists.push(None);
                Ok(())
            }
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
                let mut conditions = Vec::with_capacity(arguments.len());

                for argument in arguments {
                    match &argument.value {
                        Pattern::Variable { name, .. } => {
                            capture_flags.push(true);
                            binding_names.push(Some(name.clone()));
                            conditions.push(ListConstructorCondition::None);
                        }
                        Pattern::Discard { .. } => {
                            capture_flags.push(false);
                            binding_names.push(None);
                            conditions.push(ListConstructorCondition::None);
                        }
                        Pattern::String { value, .. } => {
                            capture_flags.push(true);
                            binding_names.push(None);
                            conditions.push(ListConstructorCondition::String(value.clone()));
                        }
                        Pattern::List { elements, tail, .. }
                            if elements.is_empty() && tail.is_none() =>
                        {
                            capture_flags.push(true);
                            binding_names.push(None);
                            conditions.push(ListConstructorCondition::EmptyList);
                        }
                        other => {
                            return Err(crate::Error::NativeCodegen {
                                message: format!(
                                    "constructor list head argument `{other:?}` is not yet supported in native functions"
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
                    conditions,
                })));
                head_field_bindings.push(Some(binding_names));
                head_extra_bindings.push(Vec::new());
                head_nested_lists.push(None);
                Ok(())
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
                head_extra_bindings.push(Vec::new());
                head_nested_lists.push(None);
                Ok(())
            }
            Pattern::List { elements, tail, .. } => {
                let info = collect_list_pattern_info(elements, tail.as_deref())?;
                capture_heads.push(true);
                head_names.push(None);
                head_matches.push(None);
                head_field_bindings.push(None);
                head_extra_bindings.push(Vec::new());
                head_nested_lists.push(Some(NestedListInfo {
                    pattern: Box::new(info),
                }));
                Ok(())
            }
            other => Err(crate::Error::NativeCodegen {
                message: format!(
                    "list pattern element `{other:?}` is not yet supported in native functions"
                ),
            }),
        }
    }

    let mut capture_heads = Vec::with_capacity(elements.len());
    let mut head_names = Vec::with_capacity(elements.len());
    let mut head_matches: Vec<Option<ListHeadMatch<'a>>> = Vec::with_capacity(elements.len());
    let mut head_field_bindings: Vec<Option<Vec<Option<EcoString>>>> =
        Vec::with_capacity(elements.len());
    let mut head_extra_bindings: Vec<Vec<EcoString>> = Vec::with_capacity(elements.len());
    let mut head_nested_lists: Vec<Option<NestedListInfo<'a>>> = Vec::with_capacity(elements.len());

    for element in elements {
        analyse_list_element(
            element,
            &mut capture_heads,
            &mut head_names,
            &mut head_matches,
            &mut head_field_bindings,
            &mut head_extra_bindings,
            &mut head_nested_lists,
        )?;
    }

    let (capture_tail, tail_name) = match tail {
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

    Ok(ListPatternInfo {
        capture_heads,
        head_names,
        head_matches,
        head_field_bindings,
        head_extra_bindings,
        head_nested_lists,
        capture_tail,
        tail_name,
        ensure_exact: tail.is_none(),
    })
}
