use shades_edsl::shades;

/// Test the main shades! macro.
#[test]
fn test_shades() {
  shades! {
    use foo::bar;
    const X: i32 = 3;
    const Y: i32 = 1;
    const PI: f32 = std::f32::consts::PI;
  };
}
