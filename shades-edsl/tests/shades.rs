use shades_edsl::shades;

struct TestInput;
struct TestOutput;
struct TestEnv;

/// Test the main shades! macro.
#[test]
fn test_shades() {
  shades! { vertex |input: TestInput, output: #TestOutput, env: #TestEnv| {
    const X: i32 = 3;
    const Y: i32 = X + 1;
    const PI: f32 = std::f32::consts::PI;

    fn add(a: i32, b: i32) -> i32 {
      if (a < 1) {
        return a;
      } else if (b < 10) {
        return a + b;
      } else {
        return 10;
      }
    }

    fn foo(test: #Test) -> () {
      let x: i32 = 3;
      return input + output * test + x;
    }
  }};
}
