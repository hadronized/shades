use shades::{env::Environment, input::Inputs, output::Outputs};
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
  let stage = shades! { vertex |input: TestInput, output: #TestOutput, env: #TestEnv| {
    fn add(a: i32, b: i32) -> i32 {
      return a + b;
    }

    fn main() -> () {}
  }};

  panic!("stage is:\n{:#?}", stage);
}
