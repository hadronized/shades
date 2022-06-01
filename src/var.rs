use crate::{expr::{Expr, ErasedExpr}, scope::ScopedHandle};
use std::ops;

/// Mutable variable.
///
/// A [`Var<T>`] is akin to an [`Expr<T>`] that can be mutated. You can go from a [`Var<T>`] to an [`Expr<T>`] via
/// either the [`From`] or [`Var::to_expr`] method.
///
/// Variables, because they allow mutations, allow to write more complicated shader functions. Also, lots of graphics
/// pipelines’ properties are variables you will have to write to, such as [`VertexShaderEnv::position`].
#[derive(Debug)]
pub struct Var<T>(pub Expr<T>)
where
  T: ?Sized;

impl<'a, T> From<&'a Var<T>> for Var<T>
where
  T: ?Sized,
{
  fn from(v: &'a Self) -> Self {
    Var(v.0.clone())
  }
}

impl<T> From<Var<T>> for Expr<T>
where
  T: ?Sized,
{
  fn from(v: Var<T>) -> Self {
    v.0
  }
}

impl<'a, T> From<&'a Var<T>> for Expr<T>
where
  T: ?Sized,
{
  fn from(v: &'a Var<T>) -> Self {
    v.0.clone()
  }
}

impl<T> Var<T>
where
  T: ?Sized,
{
  /// Create a new [`Var<T>`] from a [`ScopedHandle`].
  pub(crate) const fn new(handle: ScopedHandle) -> Self {
    Self(Expr::new(ErasedExpr::Var(handle)))
  }

  /// Coerce [`Var<T>`] into [`Expr<T>`].
  ///
  /// Remember that doing so will move the [`Var<T>`]. `clone` it if you want to preserve the source variable.
  ///
  /// > Note: use this function only when necessary. Lots of functions will accept both [`Expr<T>`] and [`Var<T>`],
  /// > performing the coercion for you automatically.
  ///
  /// # Return
  ///
  /// The expression representation of [`Var<T>`], allowing to pass the variable to functions or expressions that don’t
  /// easily coerce it automatically to [`Expr<T>`] already.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// let v = s.var(123); // Var<i32>
  /// let e = v.to_expr(); // Expr<i32>
  /// #   })
  /// # });
  /// ```
  pub fn to_expr(&self) -> Expr<T> {
    self.0.clone()
  }
}

impl<T> Var<[T]> {
  pub fn at(&self, index: impl Into<Expr<i32>>) -> Var<T> {
    Var(self.to_expr().at(index))
  }
}

impl<T, const N: usize> Var<[T; N]> {
  pub fn at(&self, index: impl Into<Expr<i32>>) -> Var<T> {
    Var(self.to_expr().at(index))
  }
}

impl<T> ops::Deref for Var<T>
where
  T: ?Sized,
{
  type Target = Expr<T>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
