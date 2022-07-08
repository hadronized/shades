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
  /// # Return
  ///
  /// This method returns a _function handle_, [`FunHandle<R, A>`], where `R` is the return type and `A` the argument
  /// list of the function. This handle can be used in various positions in the EDSL but the most interesting place is
  /// in [`Expr<T>`] and [`Var<T>`], when calling the function to, respectively, combine it with other expressions or
  /// assign it to a variable.
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
  pub fn main_fun(mut self, fundef: FunDef<(), ()>) -> Stage<S, I, O, E> {
    self.decls.push(ShaderDecl::Main(fundef.erased));
    Stage { builder: self }
  }
}
