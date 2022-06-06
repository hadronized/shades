use shades_edsl::shades;

/// Test the main shades! macro.
#[test]
fn test_shades() {
  shades! { |input, output, my_env| {
    const X: i32 = 3;
    const Y: i32 = 1;
    const PI: f32 = std::f32::consts::PI;

    fn add(a: i32, b: i32) -> i32 {
       return a + b;
    }
  }};
}

/* #[test]
fn test_vertex_shader() {
  shades_fn! { |input, output, env| {

  }}
} */
