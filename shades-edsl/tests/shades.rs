use shades::{
  env::Environment, expr::Expr, input::Inputs, output::Outputs, stage::VS, types::ToType,
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

struct TestEnv {
  t: Expr<f32>,
}

impl Environment for TestEnv {
  type Env = TestEnv;

  fn env() -> Self::Env {
    TestEnv {
      t: Expr::new_env("t"),
    }
  }

  fn env_set() -> Vec<(String, shades::types::Type)> {
    vec![("t".to_owned(), <f32 as ToType>::ty())]
  }
}

/// Test the main shades! macro.
#[test]
fn test_shades() {
  let _stage = shades! { VS |_input: TestInput, _output: TestOutput, env: TestEnv| {
    fn _add(a: f32, b: f32) -> f32 {
      let x = 3.;
      let y = 2.;
      let z = 10.;

      x = 10.;
      y *= z;

      (a + b * 2. * a) * env.t
    }

    fn main() {}
  }};
}
