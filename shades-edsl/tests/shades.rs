use shades_edsl::shades;

struct TestInput;
struct TestOutput;
struct TestEnv;

/// Test the main shades! macro.
#[test]
fn test_shades() {
  shades! { |input: TestInput, output: #TestOutput, env: #TestEnv| {
    const X: i32 = 3;
    const Y: i32 = 1;
    const PI: f32 = std::f32::consts::PI;

    fn add(a: i32, b: i32) -> i32 {
       return a + b;
    }

    fn foo(test: #Test) -> () {
      return input + output * test;
    }
  }};
}
