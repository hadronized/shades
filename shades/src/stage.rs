use std::marker::PhantomData;

use crate::{
  env::Environment,
  expr::{ErasedExpr, Expr},
  fun::{ErasedFunHandle, FunHandle, ToFun},
  input::{
    FragmentShaderInputs, GeometryShaderInputs, Inputs, TessCtrlShaderInputs, TessEvalShaderInputs,
    VertexShaderInputs,
  },
  output::{
    FragmentShaderOutputs, GeometryShaderOutputs, Outputs, TessCtrlShaderOutputs,
    TessEvalShaderOutputs, VertexShaderOutputs,
  },
  scope::ScopedHandle,
  shader::ShaderDecl,
  types::ToType,
};

/// A fully built shader stage as represented in Rust, obtained by adding the `main` function to a [`StageBuilder`].
#[derive(Debug)]
pub struct Stage<I, O, E> {
  pub(crate) builder: StageBuilder<I, O, E>,
}

/// A shader stage builder.
///
/// This opaque type is the representation of a shader stage in Rust. It contains constants, uniforms, inputs, outputs and
/// functions declarations. Such a type is used to build a shader stage and is fully built when the `main` function is
/// present in its code. See [`StageBuilder::main_fun`] for further details.
#[derive(Debug)]
pub struct StageBuilder<I, O, E> {
  pub(crate) decls: Vec<ShaderDecl>,
  next_fun_handle: u16,
  next_global_handle: u16,
  _phantom: PhantomData<(I, O, E)>,
}

impl<I, O, E> StageBuilder<I, O, E>
where
  I: Inputs,
  O: Outputs,
  E: Environment,
{
  /// Create a new _vertex shader_.
  ///
  /// This method creates a [`Stage`] that can be used as _vertex shader_. This is enforced by the fact only this
  /// method allows to build a vertex [`Stage`] by using the [`VertexShaderEnv`] argument passed to the input
  /// closure.
  ///
  /// That closure takes as first argument a [`StageBuilder`] and a [`VertexShaderEnv`] as
  /// second argument. The [`VertexShaderEnv`] allows you to access to vertex attributes found in any invocation of
  /// a vertex shader. Those are expressions (read-only) and variables (read-write) valid only in vertex shaders.
  ///
  /// # Return
  ///
  /// This method returns the fully built [`Stage`], which cannot be mutated anymore once it has been built,
  /// and can be passed to various [`writers`](crate::writer) to generate actual code for target shading languages.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{Scope, StageBuilder, V3, inputs, vec4};
  ///
  /// let vertex_shader = StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   inputs!(s, position: V3<f32>);
  ///
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.set(vertex.position, vec4!(position, 1.));
  ///   })
  /// });
  /// ```
  pub fn new_vertex_shader(
    f: impl FnOnce(Self, VertexShaderInputs<I::In>, VertexShaderOutputs<O::Out>) -> Stage<I, O, E>,
  ) -> Stage<I, O, E> {
    f(
      Self::new(),
      VertexShaderInputs::new(I::input()),
      VertexShaderOutputs::new(O::output()),
    )
  }

  /// Create a new _tessellation control shader_.
  ///
  /// This method creates a [`Stage`] that can be used as _tessellation control shader_. This is enforced by the
  /// fact only this method authorized to build a tessellation control [`Stage`] by using the [`TessCtrlShaderEnv`]
  /// argument passed to the input closure.
  ///
  /// That closure takes as first argument a mutable reference on a [`StageBuilder`] and a [`TessCtrlShaderEnv`] as
  /// second argument. The [`TessCtrlShaderEnv`] allows you to access to tessellation control attributes found in any
  /// invocation of a tessellation control shader. Those are expressions (read-only) and variables (read-write) valid
  /// only in tessellation control shaders.
  ///
  /// # Return
  ///
  /// This method returns the fully built [`Stage`], which cannot be mutated anymore once it has been built,
  /// and can be passed to various [`writers`](crate::writer) to generate actual code for target shading languages.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{Scope, StageBuilder, V3, vec4};
  ///
  /// let tess_ctrl_shader = StageBuilder::new_tess_ctrl_shader(|mut s, patch| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.set(patch.tess_level_outer.at(0), 0.1);
  ///   })
  /// });
  /// ```
  pub fn new_tess_ctrl_shader(
    f: impl FnOnce(Self, TessCtrlShaderInputs<I::In>, TessCtrlShaderOutputs<O::Out>) -> Stage<I, O, E>,
  ) -> Stage<I, O, E> {
    f(
      Self::new(),
      TessCtrlShaderInputs::new(I::input()),
      TessCtrlShaderOutputs::new(O::output()),
    )
  }

  /// Create a new _tessellation evaluation shader_.
  ///
  /// This method creates a [`Stage`] that can be used as _tessellation evaluation shader_. This is enforced by the
  /// fact only this method authorized to build a tessellation evaluation [`Stage`] by using the [`TessEvalShaderEnv`]
  /// argument passed to the input closure.
  ///
  /// That closure takes as first argument a mutable reference on a [`StageBuilder`] and a [`TessEvalShaderEnv`] as
  /// second argument. The [`TessEvalShaderEnv`] allows you to access to tessellation evaluation attributes found in
  /// any invocation of a tessellation evaluation shader. Those are expressions (read-only) and variables (read-write)
  /// valid only in tessellation evaluation shaders.
  ///
  /// # Return
  ///
  /// This method returns the fully built [`Stage`], which cannot be mutated anymore once it has been built,
  /// and can be passed to various [`writers`](crate::writer) to generate actual code for target shading languages.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{Scope, StageBuilder, V3, inputs, vec4};
  ///
  /// let tess_eval_shader = StageBuilder::new_tess_eval_shader(|mut s, patch| {
  ///   inputs!(s, position: V3<f32>);
  ///
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.set(patch.position, vec4!(position, 1.));
  ///   })
  /// });
  /// ```
  pub fn new_tess_eval_shader(
    f: impl FnOnce(Self, TessEvalShaderInputs<I::In>, TessEvalShaderOutputs<O::Out>) -> Stage<I, O, E>,
  ) -> Stage<I, O, E> {
    f(
      Self::new(),
      TessEvalShaderInputs::new(I::input()),
      TessEvalShaderOutputs::new(O::output()),
    )
  }

  /// Create a new _geometry shader_.
  ///
  /// This method creates a [`Stage`] that can be used as _geometry shader_. This is enforced by the fact only this
  /// method authorized to build a geometry [`Stage`] by using the [`GeometryShaderEnv`] argument passed to the input
  /// closure.
  ///
  /// That closure takes as first argument a mutable reference on a [`StageBuilder`] and a [`GeometryShaderEnv`] as
  /// second argument. The [`GeometryShaderEnv`] allows you to access to geometry attributes found in any invocation of
  /// a geometry shader. Those are expressions (read-only) and variables (read-write) valid only in geometry shaders.
  ///
  /// # Return
  ///
  /// This method returns the fully built [`Stage`], which cannot be mutated anymore once it has been built,
  /// and can be passed to various [`writers`](crate::writer) to generate actual code for target shading languages.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{LoopScope, Scope, StageBuilder, V3, vec4};
  ///
  /// let geo_shader = StageBuilder::new_geometry_shader(|mut s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.loop_for(0, |i| i.lt(3), |i| i + 1, |s: &mut LoopScope<()>, i| {
  ///       s.set(vertex.position, vertex.input.at(i).position());
  ///     });
  ///   })
  /// });
  /// ```
  pub fn new_geometry_shader(
    f: impl FnOnce(Self, GeometryShaderInputs<I::In>, GeometryShaderOutputs<O::Out>) -> Stage<I, O, E>,
  ) -> Stage<I, O, E> {
    f(
      Self::new(),
      GeometryShaderInputs::new(I::input()),
      GeometryShaderOutputs::new(O::output()),
    )
  }

  /// Create a new _fragment shader_.
  ///
  /// This method creates a [`Stage`] that can be used as _fragment shader_. This is enforced by the fact only this
  /// method authorized to build a [`StageBuilder`] by using the [`FragmentShaderEnv`] argument passed to the input
  /// closure.
  ///
  /// That closure takes as first argument a mutable reference on a [`StageBuilder`] and a [`FragmentShaderEnv`] as
  /// second argument. The [`FragmentShaderEnv`] allows you to access to fragment attributes found in any invocation of
  /// a fragment shader. Those are expressions (read-only) and variables (read-write) valid only in fragment shaders.
  ///
  /// # Return
  ///
  /// This method returns the fully built [`Stage`], which cannot be mutated anymore once it has been built,
  /// and can be passed to various [`writers`](crate::writer) to generate actual code for target shading languages.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{Geometry as _, Scope, StageBuilder, V4, outputs, vec4};
  ///
  /// let geo_shader = StageBuilder::new_fragment_shader(|mut s, fragment| {
  ///   outputs!(s, color: V4<f32>);
  ///
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.set(color, fragment.frag_coord.normalize());
  ///   })
  /// });
  /// ```
  pub fn new_fragment_shader(
    f: impl FnOnce(Self, FragmentShaderInputs<I::In>, FragmentShaderOutputs<O::Out>) -> Stage<I, O, E>,
  ) -> Stage<I, O, E> {
    f(
      Self::new(),
      FragmentShaderInputs::new(I::input()),
      FragmentShaderOutputs::new(O::output()),
    )
  }

  /// Create a new empty shader.
  pub(crate) fn new() -> Self {
    Self {
      decls: Vec::new(),
      next_fun_handle: 0,
      next_global_handle: 0,
      _phantom: PhantomData,
    }
  }

  /// Create a new function in the shader and get its handle for future use.
  ///
  /// This method requires to pass a closure encoding the argument(s) and return type of the function to create. The
  /// closure’s body encodes the body of the function to create. The number of arguments will directly impact the
  /// number of arguments the created function will have. The return type can be [`()`](unit) if the function doesn’t
  /// return anything or [`Expr<T>`] if it does return something.
  ///
  /// The first argument of the closure is a mutable reference on a [`Scope`]. Its type parameter must be set to the
  /// return type. The scope allows you to add instructions to the function body of the generated function. As in
  /// vanilla Rust, the last expression in a function is assumed as return value, if the function returns a value.
  /// However, unlike Rust, if your function returns something, it **cannot `return` it: it has to use the
  /// expression-as-last-instruction syntax**. It means that even if you don’t use the [`Scope`] within the last
  /// expression of your function body, the returned expression will still be part of the function as special returned
  /// expression:
  ///
  /// ```
  /// # use shades::StageBuilder;
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{Expr, Scope};
  ///
  /// let f = s.fun(|s: &mut Scope<Expr<f32>>, a: Expr<f32>| a + 1.);
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  ///
  /// However, as mentioned above, you cannot `return` the last expression (`leave`), as this is not accepted by the
  /// EDSL:
  ///
  /// ```compile_fail
  /// # use shades::StageBuilder;
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{Expr, Scope};
  ///
  /// let f = s.fun(|s: &mut Scope<Expr<f32>>, a: Expr<f32>| {
  ///   s.leave(a + 1.);
  /// });
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  ///
  /// Please refer to the [`Scope`] documentation for a complete list of the instructions you can record.
  ///
  /// # Caveats
  ///
  /// On a last note, you can still use the `return` keyword from Rust, but it is highly discouraged, as returning with
  /// `return` cannot be captured by the EDSL. It means that you will not get the shader code you expect.
  ///
  /// ```
  /// # use shades::{Expr, Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{Expr, Scope};
  ///
  /// let f = s.fun(|s: &mut Scope<Expr<f32>>, a: Expr<f32>| {
  ///   return a + 1.;
  /// });
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  ///
  /// An example of a broken shader is when you use the Rust `return` keyword inside a conditional statement or looping
  /// statement:
  ///
  /// ```
  /// # use shades::StageBuilder;
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{CanEscape as _, EscapeScope, Expr, Scope};
  ///
  /// // don’t do this.
  /// let f = s.fun(|s: &mut Scope<Expr<f32>>, a: Expr<f32>| {
  ///   s.when(a.lt(10.), |s: &mut EscapeScope<Expr<f32>>| {
  ///     // /!\ problem here /!\
  ///     return;
  ///   });
  ///
  ///   a + 1.
  /// });
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  ///
  /// This snippet will create a GLSL function testing whether its `a` argument is less than `10.` and if it’s the case,
  /// does nothing inside of it (the `return` is not captured by the EDSL).
  ///
  /// # Return
  ///
  /// This method returns a _function handle_, [`FunHandle<R, A>`], where `R` is the return type and `A` the argument
  /// list of the function. This handle can be used in various positions in the EDSL but the most interesting place is
  /// in [`Expr<T>`] and [`Var<T>`], when calling the function to, respectively, combine it with other expressions or
  /// assign it to a variable.
  ///
  /// ## Nightly-only: call syntax
  ///
  /// On the current version of stable `rustc` (1.49), it is not possible to use a [`FunHandle<R, A>`] as you would use
  /// a normal Rust function: you have to use the [`FunHandle::call`] method, which is not really elegant nor ergonomic.
  ///
  /// To fix this problem, enable the `fun-call` feature gate.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{Exponential as _, Expr, Scope, StageBuilder, lit};
  ///
  /// let shader = StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   // create a function taking a floating-point number and returning its square
  ///   let square = s.fun(|s: &mut Scope<Expr<f32>>, a: Expr<f32>| {
  ///     a.pow(2.)
  ///   });
  ///
  ///   // `square` can now be used to square floats!
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     // if you use the nightly compiler
  /// #   #[cfg(feature = "fun-call")]
  ///     let nine = s.var(square(lit!(3.)));
  ///
  ///     // if you’d rather use stable
  ///     let nine = s.var(square.call(lit!(3.)));
  ///   })
  /// });
  /// ```
  pub fn fun<F, R, A>(&mut self, f: F) -> FunHandle<R, A>
  where
    F: ToFun<R, A>,
  {
    let fundef = f.build_fn();
    let handle = self.next_fun_handle;
    self.next_fun_handle += 1;

    self.decls.push(ShaderDecl::FunDef(handle, fundef.erased));

    FunHandle::new(ErasedFunHandle::UserDefined(handle as _))
  }

  /// Declare the `main` function of the shader stage.
  ///
  /// This method is very similar to [`StageBuilder::fun`] in the sense it declares a function. However, it declares the special
  /// `main` entry-point of a shader stage, which doesn’t have any argument and returns nothing, and is the only
  /// way to finalize the building of a [`Stage`].
  ///
  /// The input closure must take a single argument: a mutable reference on a `Scope<()>`, as the `main` function
  /// cannot return anything.
  ///
  /// # Return
  ///
  /// The fully built [`Stage`], which cannot be altered anymore.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{Scope, StageBuilder};
  ///
  /// let shader = StageBuilder::new_vertex_shader(|s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     // …
  ///   })
  /// });
  /// ```
  pub fn main_fun<F, R>(mut self, f: F) -> Stage<I, O, E>
  where
    F: ToFun<R, ()>,
  {
    let fundef = f.build_fn();

    self.decls.push(ShaderDecl::Main(fundef.erased));

    Stage { builder: self }
  }

  /// Declare a new constant, shared between all functions and constants that come next.
  ///
  /// The input argument is any object that can be transformed [`Into`] an [`Expr<T>`]. At this level in the
  /// shader, pretty much nothing but literals and other constants are accepted here.
  ///
  /// # Return
  ///
  /// An [`Expr<T>`] representing the constant passed as input.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{EscapeScope, Expr, Scope, StageBuilder, lit};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// // don’t do this.
  /// let illum_coefficient: Expr<f32> = s.constant(10.);
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn constant<T>(&mut self, expr: Expr<T>) -> Expr<T>
  where
    T: ToType,
  {
    let handle = self.next_global_handle;
    self.next_global_handle += 1;

    self
      .decls
      .push(ShaderDecl::Const(handle, T::ty(), expr.erased));

    Expr::new(ErasedExpr::Var(ScopedHandle::global(handle)))
  }
}
