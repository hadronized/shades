use shades::{lit, Scope, Shader, Var, POSITION, V, V3};

fn main() {
  let mut vertex_shader: Shader<V> = Shader::new();
  let pos: Var<V, V3<f32>> = vertex_shader.input();

  vertex_shader.main_fun(|s: &mut Scope<V, ()>| {
    s.set(POSITION, lit![0., 0.1, 1., -1.]);
  });

  let output = shades::writer::glsl::write_shader_to_str(&vertex_shader).unwrap();
  println!("{}", output);
}
