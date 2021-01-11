use shades::{lit, Expr, Scope, Shader};

fn main() {
  let vertex_shader = Shader::new_vertex_shader(|shader, vertex| {
    let increment = shader.fun(|s: &mut Scope<Expr<f32>>, a: Expr<f32>| a + lit!(1.));

    shader.main_fun(|s: &mut Scope<()>| {
      let x = s.var(lit!(1.));
      s.set(&vertex.position, lit![0., 0.1, 1., -1.]);
    });
  });

  let output = shades::writer::glsl::write_shader_to_str(&vertex_shader).unwrap();
  println!("{}", output);
}
