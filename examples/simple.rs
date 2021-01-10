use shades::{lit, Expr, Scope, Shader, Var, L, POSITION, V, V3};

fn main() {
  let mut vertex_shader: Shader<V> = Shader::new();
  let pos: Var<V, V3<f32>> = vertex_shader.input();

  let add = vertex_shader.fun(|s: &mut Scope<V, Expr<V, i32>>, a: Expr<L, _>| a + lit!(1));
  vertex_shader.main_fun(|s: &mut Scope<V, ()>| {
    let x = add(lit!(1));
    s.set(POSITION, lit![0., 0.1, 1., -1.]);
  });

  let output = shades::writer::glsl::write_shader_to_str(&vertex_shader).unwrap();
  println!("{}", output);
}
