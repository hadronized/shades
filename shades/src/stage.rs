use std::marker::PhantomData;

use crate::{
  env::Environment,
  expr::{ErasedExpr, Expr},
  fun::{ErasedFunHandle, FunDef, FunHandle},
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
pub struct Stage<S, I, O, E>
where
  S: ?Sized,
{
  pub(crate) builder: ModBuilder<S, I, O, E>,
}

/// Shader module.
///
/// A shader module is defined by its type, which leads to the possible kinds ofinputs and outputs.
///
/// Shader modules that define a `main` functions are usually called “shader stages”, and the ones that don’t are
/// referred to as simple “modules.”
pub trait ShaderModule<I, O> {
  type Inputs;
  type Outputs;

  /// Create a new shader stage.
  ///
  /// This method creates a [`Stage`] that can be used only by the kind of flavour defined by the `S` type variable.
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
  fn new_shader_module<E>(
    f: impl FnOnce(
      ModBuilder<Self, I, O, E>,
      Self::Inputs,
      Self::Outputs,
      E::Env,
    ) -> Stage<Self, I, O, E>,
  ) -> Stage<Self, I, O, E>
  where
    E: Environment;
}

/// Vertex shader module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VS;

impl<I, O> ShaderModule<I, O> for VS
where
  I: Inputs,
  O: Outputs,
{
  type Inputs = VertexShaderInputs<I::In>;
  type Outputs = VertexShaderOutputs<O::Out>;

  fn new_shader_module<E>(
    f: impl FnOnce(
      ModBuilder<Self, I, O, E>,
      Self::Inputs,
      Self::Outputs,
      E::Env,
    ) -> Stage<Self, I, O, E>,
  ) -> Stage<Self, I, O, E>
  where
    E: Environment,
  {
    f(
      ModBuilder::new(),
      VertexShaderInputs::new(I::input()),
      VertexShaderOutputs::new(O::output()),
      E::env(),
    )
  }
}

/// Tessellation control shader module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TCS;

impl<I, O> ShaderModule<I, O> for TCS
where
  I: Inputs,
  O: Outputs,
{
  type Inputs = TessCtrlShaderInputs<I::In>;
  type Outputs = TessCtrlShaderOutputs<O::Out>;

  fn new_shader_module<E>(
    f: impl FnOnce(
      ModBuilder<Self, I, O, E>,
      Self::Inputs,
      Self::Outputs,
      E::Env,
    ) -> Stage<Self, I, O, E>,
  ) -> Stage<Self, I, O, E>
  where
    E: Environment,
  {
    f(
      ModBuilder::new(),
      TessCtrlShaderInputs::new(I::input()),
      TessCtrlShaderOutputs::new(O::output()),
      E::env(),
    )
  }
}

/// Tessellation evaluation shader module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TES;

impl<I, O> ShaderModule<I, O> for TES
where
  I: Inputs,
  O: Outputs,
{
  type Inputs = TessEvalShaderInputs<I::In>;
  type Outputs = TessEvalShaderOutputs<O::Out>;

  fn new_shader_module<E>(
    f: impl FnOnce(
      ModBuilder<Self, I, O, E>,
      Self::Inputs,
      Self::Outputs,
      E::Env,
    ) -> Stage<Self, I, O, E>,
  ) -> Stage<Self, I, O, E>
  where
    E: Environment,
  {
    f(
      ModBuilder::new(),
      TessEvalShaderInputs::new(I::input()),
      TessEvalShaderOutputs::new(O::output()),
      E::env(),
    )
  }
}

/// Geometry shader module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GS;

impl<I, O> ShaderModule<I, O> for GS
where
  I: Inputs,
  O: Outputs,
{
  type Inputs = GeometryShaderInputs<I::In>;
  type Outputs = GeometryShaderOutputs<O::Out>;

  fn new_shader_module<E>(
    f: impl FnOnce(
      ModBuilder<Self, I, O, E>,
      Self::Inputs,
      Self::Outputs,
      E::Env,
    ) -> Stage<Self, I, O, E>,
  ) -> Stage<Self, I, O, E>
  where
    E: Environment,
  {
    f(
      ModBuilder::new(),
      GeometryShaderInputs::new(I::input()),
      GeometryShaderOutputs::new(O::output()),
      E::env(),
    )
  }
}

/// Fragment shader module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FS;

impl<I, O> ShaderModule<I, O> for FS
where
  I: Inputs,
  O: Outputs,
{
  type Inputs = FragmentShaderInputs<I::In>;
  type Outputs = FragmentShaderOutputs<O::Out>;

  fn new_shader_module<E>(
    f: impl FnOnce(
      ModBuilder<Self, I, O, E>,
      Self::Inputs,
      Self::Outputs,
      E::Env,
    ) -> Stage<Self, I, O, E>,
  ) -> Stage<Self, I, O, E>
  where
    E: Environment,
  {
    f(
      ModBuilder::new(),
      FragmentShaderInputs::new(I::input()),
      FragmentShaderOutputs::new(O::output()),
      E::env(),
    )
  }
}

/// A shader module builder.
///
/// This opaque type is the representation of a shader module in Rust. It contains constants, uniforms, inputs, outputs and
/// functions declarations.
///
/// It can also be sude to create shareable modules.
#[derive(Debug)]
pub struct ModBuilder<S, I, O, E>
where
  S: ?Sized,
{
  pub(crate) decls: Vec<ShaderDecl>,
  next_fun_handle: u16,
  next_global_handle: u16,
  _phantom: PhantomData<(*const S, I, O, E)>,
}

impl<S, I, O, E> ModBuilder<S, I, O, E>
where
  S: ?Sized,
  I: Inputs,
  O: Outputs,
  E: Environment,
{
  fn new() -> Self {
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
  pub fn fun<R, A>(&mut self, fundef: FunDef<R, A>) -> FunHandle<R, A> {
    let handle = self.next_fun_handle;
    self.next_fun_handle += 1;

    self.decls.push(ShaderDecl::FunDef(handle, fundef.erased));

    FunHandle::new(ErasedFunHandle::UserDefined(handle as _))
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

impl<S, I, O, E> ModBuilder<S, I, O, E>
where
  S: ShaderModule<I, O>,
  I: Inputs,
  O: Outputs,
  E: Environment,
{
  /// Declare a new shader stage.
  pub fn new_stage(
    f: impl FnOnce(ModBuilder<S, I, O, E>, S::Inputs, S::Outputs, E::Env) -> Stage<S, I, O, E>,
  ) -> Stage<S, I, O, E> {
    S::new_shader_module(f)
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
  pub fn main_fun(mut self, fundef: FunDef<(), ()>) -> Stage<S, I, O, E> {
    self.decls.push(ShaderDecl::Main(fundef.erased));
    Stage { builder: self }
  }
}
