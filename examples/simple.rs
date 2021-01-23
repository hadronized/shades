use shades::{lit, Expr, Scope, ShaderBuilder, V2};

fn main() {
  let vertex_shader = ShaderBuilder::new_vertex_shader(|mut shader: ShaderBuilder, vertex| {
    let increment = shader.fun(|_: &mut Scope<Expr<f32>>, a: Expr<f32>| a + lit!(1.));

    shader.fun(|_: &mut Scope<()>, _: Expr<[[V2<f32>; 2]; 15]>| ());

    shader.main_fun(|s: &mut Scope<()>| {
      let x = s.var(1.);
      let _ = s.var([1, 2]);
      s.set(vertex.clip_distance.at(0), increment(x.into()));
      s.set(&vertex.position, lit![0., 0.1, 1., -1.]);
    })
  });

  let output = shades::writer::glsl::write_shader_to_str(&vertex_shader).unwrap();
  println!("{}", output);
}
