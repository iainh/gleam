import gleam/io
import gleam/dynamic.{type Dynamic}

@external(cranelift, "facil_io", "facil_io_run")
fn facil_io_run() -> Nil

@external(cranelift, "facil_io", "facil_io_send_body")
fn send_body(request: Dynamic, body: String) -> Nil

pub fn handle_request(request: Dynamic) -> Nil {
  send_body(request, "Hello from facil.io + Gleam!")
}

pub fn main() {
  io.println("Starting facil.io HTTP server on http://localhost:3000")
  facil_io_run()
}
