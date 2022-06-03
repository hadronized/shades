use shades_edsl::shades;

/// Test the main shades! macro.
#[test]
fn test_shades() {
  shades! {
    use foo::bar;
    const X: i32 = 3;
    const Y: i32 = 1;
    const PI: f32 = std::f32::consts::PI;

    fn add(a: i32, b: i32) -> i32 {
       a + b
    }
  };
}

#[test]
fn test_vertex_shader() {
  shades_fn! { |input, output, env| {

  }}
}
