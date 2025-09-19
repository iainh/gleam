@external(native, "runtime_cranelift", "io_print")
@external(erlang, "gleam_stdlib", "print")
@external(javascript, "../gleam_stdlib.mjs", "print")
pub fn print(string: String) -> Nil

@external(native, "runtime_cranelift", "io_println")
@external(erlang, "gleam_stdlib", "println")
@external(javascript, "../gleam_stdlib.mjs", "console_log")
pub fn println(string: String) -> Nil

@external(native, "runtime_cranelift", "io_print_error")
@external(erlang, "gleam_stdlib", "print_error")
@external(javascript, "../gleam_stdlib.mjs", "print_error")
pub fn print_error(string: String) -> Nil

@external(native, "runtime_cranelift", "io_println_error")
@external(erlang, "gleam_stdlib", "println_error")
@external(javascript, "../gleam_stdlib.mjs", "console_error")
pub fn println_error(string: String) -> Nil
