use shades_edsl::shades;

struct TestInput;
struct TestOutput;
struct TestEnv;

// TODO: We need to inject shades::expr::Expr::from() where we have a syn::Expr, whenever implementing quote::ToTokens.
// TODO: This is required to ensure we lift all expressions and sub-expressions, i.e. 1 + 2 must become Expr::from(1) +
// TODO: Expr::from(2) and not Expr::from(1 + 2).
/// Test the main shades! macro.
#[test]
fn test_shades() {
  shades! { vertex |input: TestInput, output: #TestOutput, env: #TestEnv| {
    const X: i32 = 3;
    const Y: i32 = 1;
    const PI: f32 = std::f32::consts::PI;

    fn add(a: i32, b: i32) -> i32 {
      if (a < 10) {
        return a;
      }

      return a + b;
    }

    fn foo(test: #Test) -> () {
      return input + output * test;
    }
  }};
}
