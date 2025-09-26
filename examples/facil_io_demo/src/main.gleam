import gleam/io

@external(cranelift, "facil_io_demo", "facil_io_demo_run")
fn facil_io_demo_run() -> Nil

pub fn main() {
  io.println("Starting facil.io HTTP server on http://localhost:3000")
  facil_io_demo_run()
}
