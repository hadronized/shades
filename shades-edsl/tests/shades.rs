use shades_edsl::shades;

struct TestInput;
struct TestOutput;
struct TestEnv;

/// Test the main shades! macro.
#[test]
fn test_shades() {
  let stage = shades! { vertex |input: TestInput, output: #TestOutput, env: #TestEnv| {
    fn add(a: i32, b: i32) -> i32 {
      return a + b;
    }
  }};

  panic!("stage is:\n{:#?}", stage);
}
