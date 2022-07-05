use shades::{
  expr::Expr,
  input::Inputs,
  lit,
  scope::{Conditional as _, Scope},
  stage::ModBuilder,
  types::{ToType, Type, V2, V3, V4},
  vec4,
};

pub struct MyVertex {
  pos: Expr<V3<f32>>,

  #[allow(dead_code)]
  color: Expr<V4<f32>>,
}

impl Inputs for MyVertex {
  type In = Self;

  fn input() -> Self::In {
    Self {
      pos: Expr::new_input(0),
      color: Expr::new_input(1),
    }
  }

  fn input_set() -> Vec<(usize, Type)> {
    vec![
      (0, <V3<f32> as ToType>::ty()),
      (1, <V4<f32> as ToType>::ty()),
    ]
  }
}

fn main() {
  let vertex_shader = ModBuilder::new_vertex_shader(
    |mut shader: ModBuilder<MyVertex, (), ()>, input, output| {
      let increment = shader.fun(|_: &mut Scope<Expr<f32>>, a: Expr<f32>| a + lit!(1.));

      shader.fun(|_: &mut Scope<()>, _: Expr<[[V2<f32>; 2]; 15]>| ());

      shader.main_fun(|s: &mut Scope<()>| {
        let x = s.var(1.);
        let _ = s.var([1, 2]);
        s.set(output.clip_distance.at(0), increment(x.clone()));
        s.set(
          &output.position,
          vec4!(input.pos, 1.) * lit![0., 0.1, 1., -1.],
        );

        s.loop_while(true, |s| {
          s.when(x.clone().eq(1.), |s| {
            s.loop_break();
            s.abort();
          });
        });
      })
    },
  );

  let output = shades::writer::glsl::write_shader_to_str(&vertex_shader).unwrap();
  println!("{}", output);
}
