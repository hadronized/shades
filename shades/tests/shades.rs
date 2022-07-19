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
  let stage = shades! { VS |_input: TestInput, _output: TestOutput, env: TestEnv| {
    fn add(a: f32, b: f32) -> f32 {
      let x = 3.;
      let y = 2.;
      let z = 10.;
      let w = [1., 2.];

      x = 10.;
      y *= z;

      while true {
        x += 1.;
      }

      (a + b * 2. * a) * env.t + w[1]
    }

    fn main() {
      let _x = add(1., 2.);
    }
  }};

  let expected = r#"uniform float t;

float fun_0(float arg_0, float arg_1) {
  float var_0_0 = 3.;
  float var_0_1 = 2.;
  float var_0_2 = 10.;
  float[2] var_0_3 = float[2](1.,2.);
  var_0_0 = 10.;
  var_0_1 *= var_0_2;
  while (true) {
    var_0_0 += 1.;
  }
  return (((arg_0 + ((arg_1 * 2.) * arg_0)) * t) + var_0_3[1]);

}

void main() {
  float var_0_0 = fun_0(1., 2.);
}"#;

  assert_eq!(
    shades::writer::glsl::write_shader_to_str(&stage).unwrap(),
    expected
  );
}
