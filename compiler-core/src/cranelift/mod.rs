mod config;
mod emit;
mod lowering;

pub use config::ModuleConfig;
pub(crate) use config::module_contains_public_main;
pub use emit::emit_object;
