//! Global atom registry shared by the runtime and generated code.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::value::Value;

#[derive(Debug, Default)]
struct AtomTableInner {
    map: HashMap<Arc<str>, u32>,
    values: Vec<Arc<str>>,
}

/// Global atom registry used by tagged immediates.
#[derive(Debug, Default)]
pub struct AtomTable {
    inner: RwLock<AtomTableInner>,
}

impl AtomTable {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(AtomTableInner::default()),
        }
    }

    pub fn intern(&self, text: &str) -> Value {
        if let Some(index) = self.lookup_index(text) {
            return Value::atom(index);
        }

        let mut guard = self.inner.write().expect("atom table poisoned");
        if let Some(index) = guard.map.get(text).copied() {
            return Value::atom(index);
        }

        let storage: Arc<str> = Arc::from(text);
        let index = guard.values.len() as u32;
        guard.map.insert(storage.clone(), index);
        guard.values.push(storage);
        Value::atom(index)
    }

    pub fn resolve(&self, index: u32) -> Option<Arc<str>> {
        let guard = self.inner.read().ok()?;
        guard.values.get(index as usize).cloned()
    }

    fn lookup_index(&self, text: &str) -> Option<u32> {
        let guard = self.inner.read().ok()?;
        guard.map.get(text).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_same_value() {
        let table = AtomTable::new();
        let first = table.intern("hello");
        let second = table.intern("hello");
        assert_eq!(first.to_raw(), second.to_raw());
    }
}
