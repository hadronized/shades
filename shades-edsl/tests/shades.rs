use shades::{
  env::Environment,
  input::Inputs,
  output::Outputs,
  stage::{Stage, VS},
};
use shades_edsl::shades;

#[derive(Debug)]
struct TestInput;

impl Inputs for TestInput {
  type In = ();

  fn input() -> Self::In {
    ()
  }

  fn input_set() -> Vec<(u16, shades::types::Type)> {
    Vec::new()
  }
}

#[derive(Debug)]
struct TestOutput;

impl Outputs for TestOutput {
  type Out = ();

  fn output() -> Self::Out {
    ()
  }

  fn output_set() -> Vec<(u16, shades::types::Type)> {
    Vec::new()
  }
}

#[derive(Debug)]
struct TestEnv;

impl Environment for TestEnv {
  type Env = ();

  fn env() -> Self::Env {
    ()
  }

  fn env_set() -> Vec<(u16, shades::types::Type)> {
    Vec::new()
  }
}

/// Test the main shades! macro.
#[test]
fn test_shades() {
  let _stage: Stage<VS, TestInput, TestOutput, TestEnv> = shades! { |_input, _output, _env| {
    fn _add(a: i32, b: i32) -> i32 {
      let x = 3;
      let y = 2;
      let z = 10;

      x = 10;
      y *= z;

      a + b
    }

    fn main() {}
  }};
}
