//! EDSL expressions.
//!
//! Expressions are read-only items representing different kind of objects, such as literal values, operations between
//! expressions (unary, binary, function calls, etc.), arrays of expressions, etc. etc.
//!
//! There is an important relationship between expressions ([`Expr`]) and variables [`Var`]. [`Var`] are basically
//! read/write expressions and can coerce to expressions; the other way around is not possible.
use crate::{
  builtin::BuiltIn,
  erased::Erased,
  fun::ErasedFunHandle,
  scope::ScopedHandle,
  swizzle::Swizzle,
  types::{ToType, Type, M22, M33, M44, V2, V3, V4},
  var::Var,
};
use std::{marker::PhantomData, ops};

/// Representation of an expression.
#[derive(Clone, Debug, PartialEq)]
pub enum ErasedExpr {
  // scalars
  LitInt(i32),
  LitUInt(u32),
  LitFloat(f32),
  LitBool(bool),
  // vectors
  LitInt2([i32; 2]),
  LitUInt2([u32; 2]),
  LitFloat2([f32; 2]),
  LitBool2([bool; 2]),
  LitInt3([i32; 3]),
  LitUInt3([u32; 3]),
  LitFloat3([f32; 3]),
  LitBool3([bool; 3]),
  LitInt4([i32; 4]),
  LitUInt4([u32; 4]),
  LitFloat4([f32; 4]),
  LitBool4([bool; 4]),
  // matrices
  LitM22(M22),
  // LitM23(M23),
  // LitM24(M24),
  // LitM32(M32),
  LitM33(M33),
  // LitM34(M34),
  // LitM42(M42),
  // LitM43(M43),
  LitM44(M44),
  // arrays
  Array(Type, Vec<ErasedExpr>),
  // var
  Var(ScopedHandle),
  // built-in functions and operators
  Not(Box<Self>),
  And(Box<Self>, Box<Self>),
  Or(Box<Self>, Box<Self>),
  Xor(Box<Self>, Box<Self>),
  BitOr(Box<Self>, Box<Self>),
  BitAnd(Box<Self>, Box<Self>),
  BitXor(Box<Self>, Box<Self>),
  Neg(Box<Self>),
  Add(Box<Self>, Box<Self>),
  Sub(Box<Self>, Box<Self>),
  Mul(Box<Self>, Box<Self>),
  Div(Box<Self>, Box<Self>),
  Rem(Box<Self>, Box<Self>),
  Shl(Box<Self>, Box<Self>),
  Shr(Box<Self>, Box<Self>),
  Eq(Box<Self>, Box<Self>),
  Neq(Box<Self>, Box<Self>),
  Lt(Box<Self>, Box<Self>),
  Lte(Box<Self>, Box<Self>),
  Gt(Box<Self>, Box<Self>),
  Gte(Box<Self>, Box<Self>),
  // function call
  FunCall(ErasedFunHandle, Vec<Self>),
  // swizzle
  Swizzle(Box<Self>, Swizzle),
  // field expression, as in a struct Foo { float x; }, foo.x is an Expr representing the x field on object foo
  Field { object: Box<Self>, field: Box<Self> },
  ArrayLookup { object: Box<Self>, index: Box<Self> },
}

impl ErasedExpr {
  pub(crate) const fn new_builtin(builtin: BuiltIn) -> Self {
    ErasedExpr::Var(ScopedHandle::builtin(builtin))
  }
}

/// Expression representation.
///
/// An expression is anything that carries a (typed) value and that can be combined in various ways with other
/// expressions. A literal, a constant or a variable are all expressions. The sum (as in `a + b`) of two expressions is
/// also an expression. A function call returning an expression is also an expression, as in `a * sin(b)`. Accessing an
/// element in an array (which is an expression as it carries items) via an index (an expression) is also an
/// expression — e.g. `levels[y * HEIGHT + x] * size`. The same thing applies to field access, swizzling, etc. etc.
///
/// On a general note, expressions are pretty central to the EDSL, as they are the _lower level concept_ you will be
/// able to manipulate. Expressions are side effect free, so a variable, for instance, can either be considered as an
/// expression or not. If `x` is a variable (see [`Var`]), then `x * 10` is an expression, but using `x` to mutate its
/// content does not make a use of `x` as an expression. It means that expressions are read-only and even though you
/// can go from higher constructs (like variables) to expressions, the opposite direction is forbidden.
///
/// # Literals
///
/// The most obvious kind of expression is a literal — e.g. `1`, `false`, `3.14`, etc. Any type `T` that defines an
/// implementor `From<T> for Expr<T>` can be used as literal. You can then use, for instance, `1.into()`,
/// `Expr::from(1)`, etc.
///
/// > That is a concern of yours only if you do not use [shades-edsl].
///
/// # Expressions from side-effects
///
/// Some side-effects will create expressions, such as creating a variable or a constant. Most of the time, you
/// shouldn’t have to worry about the type of the expression as it should be inferred based on the side-effect.
///
/// # Expression macros
///
/// Some macros will create expressions for you, such as [`vec2!`](vec2), [`vec3!`](vec3) and [`vec4!`](vec4) or the
/// [`sw!`](sw) macros.
///
/// [shades-edsl]: https://crates.io/crates/shades-edsl
#[derive(Debug)]
pub struct Expr<T>
where
  T: ?Sized,
{
  pub(crate) erased: ErasedExpr,
  _phantom: PhantomData<T>,
}

impl<T> Clone for Expr<T>
where
  T: ?Sized,
{
  fn clone(&self) -> Self {
    Self::new(self.erased.clone())
  }
}

impl<T> Erased for Expr<T> {
  type Erased = ErasedExpr;

  fn to_erased(self) -> Self::Erased {
    self.erased
  }

  fn erased(&self) -> &Self::Erased {
    &self.erased
  }

  fn erased_mut(&mut self) -> &mut Self::Erased {
    &mut self.erased
  }
}

impl<T> Expr<T>
where
  T: ?Sized,
{
  /// Type an [`ErasedExpr`] and return it wrapped in [`Expr<T>`].
  pub const fn new(erased: ErasedExpr) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }

  /// Create a new input.
  ///
  /// You should use this function when implementing [`Inputs::input()`].
  pub const fn new_input(handle: u16) -> Self {
    Self::new(ErasedExpr::Var(ScopedHandle::Input(handle)))
  }

  /// Create a new output.
  ///
  /// You should use this function when implementing [`Outputs::output()`].
  pub fn new_output(handle: u16) -> Self {
    Self::new(ErasedExpr::Var(ScopedHandle::Output(handle)))
  }

  /// Create a new environment variable.
  ///
  /// You should use this function when implementing [`Environment::env()`].
  pub fn new_env(name: impl Into<String>) -> Self {
    Self::new(ErasedExpr::Var(ScopedHandle::Env(name.into())))
  }

  /// Create a new function argument.
  ///
  /// You shouldn’t have to use that function, as its main purpose is to be used by [shades-edsl].
  ///
  /// [shades-edsl]: https://crates.io/crates/shades-edsl
  pub const fn new_fun_arg(handle: u16) -> Self {
    Self::new(ErasedExpr::Var(ScopedHandle::FunArg(handle)))
  }

  /// Equality expression.
  ///
  /// This method builds an expression representing the equality between two expressions.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing the equality between the two input expressions.
  pub fn eq(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Eq(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Inequality expression.
  ///
  /// This method builds an expression representing the inequality between two expressions.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing the inequality between the two input expressions.
  pub fn neq(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Neq(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

impl<T> Expr<T>
where
  T: PartialOrd,
{
  /// Less-than expression.
  ///
  /// This method builds an expression representing the binary operation `a < b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a < b`.
  pub fn lt(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lt(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Less-than-or-equal expression.
  ///
  /// This method builds an expression representing the binary operation `a <= b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a <= b`.
  pub fn lte(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lte(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Greater-than expression.
  ///
  /// This method builds an expression representing the binary operation `a > b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a > b`.
  pub fn gt(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gt(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Less-than-or-equal expression.
  ///
  /// This method builds an expression representing the binary operation `a <= b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a <= b`.
  pub fn gte(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gte(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

impl Expr<bool> {
  /// Logical _and_ expression.
  ///
  /// This method builds an expression representing the logical operation `a AND b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a AND b`.
  pub fn and(&self, rhs: impl Into<Expr<bool>>) -> Expr<bool> {
    Expr::new(ErasedExpr::And(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Logical _or_ expression.
  ///
  /// This method builds an expression representing the logical operation `a OR b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a OR b`.
  pub fn or(&self, rhs: impl Into<Expr<bool>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Or(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Logical _exclusive or_ expression.
  ///
  /// This method builds an expression representing the logical operation `a XOR b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a XOR b`.
  pub fn xor(&self, rhs: impl Into<Expr<bool>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Xor(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

impl<T> Expr<[T]> {
  /// Array lookup.
  ///
  /// The expression `a.at(i)` represents an _array lookup_, where `a` is an array — which type must be either
  /// [`Expr<[T]>`](Expr) or [`Expr<[T; N]>`](Expr) – and `i` is an [`Expr<i32>`].
  ///
  /// # Return
  ///
  /// The resulting [`Expr<T>`] represents the array lookup in `a` at index `i`.
  pub fn at(&self, index: Expr<i32>) -> Expr<T> {
    Expr::new(ErasedExpr::ArrayLookup {
      object: Box::new(self.erased.clone()),
      index: Box::new(index.erased),
    })
  }
}

impl<T, const N: usize> Expr<[T; N]> {
  /// Array lookup.
  ///
  /// The expression `a.at(i)` represents an _array lookup_, where `a` is an array — which type must be either
  /// [`Expr<[T]>`](Expr) or [`Expr<[T; N]>`](Expr) – and `i` is an [`Expr<i32>`].
  ///
  /// # Return
  ///
  /// The resulting [`Expr<T>`] represents the array lookup in `a` at index `i`.
  pub fn at(&self, index: impl Into<Expr<i32>>) -> Expr<T> {
    Expr::new(ErasedExpr::ArrayLookup {
      object: Box::new(self.erased.clone()),
      index: Box::new(index.into().erased),
    })
  }
}

// not
macro_rules! impl_Not_Expr {
  ($t:ty) => {
    impl ops::Not for Expr<$t> {
      type Output = Self;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.erased)))
      }
    }

    impl ops::Not for Var<$t> {
      type Output = Expr<$t>;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.0.erased)))
      }
    }
  };
}

impl_Not_Expr!(bool);
impl_Not_Expr!(V2<bool>);
impl_Not_Expr!(V3<bool>);
impl_Not_Expr!(V4<bool>);

// neg
macro_rules! impl_Neg {
  ($t:ty) => {
    impl ops::Neg for Expr<$t> {
      type Output = Self;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.erased)))
      }
    }

    impl ops::Neg for Var<$t> {
      type Output = Expr<$t>;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.0.erased)))
      }
    }
  };
}

impl_Neg!(i32);
impl_Neg!(V2<i32>);
impl_Neg!(V3<i32>);
impl_Neg!(V4<i32>);

impl_Neg!(u32);
impl_Neg!(V2<u32>);
impl_Neg!(V3<u32>);
impl_Neg!(V4<u32>);

impl_Neg!(f32);
impl_Neg!(V2<f32>);
impl_Neg!(V3<f32>);
impl_Neg!(V4<f32>);

// binary arithmetic and logical (+, -, *, /, %)
// binop
macro_rules! impl_binop_Expr {
  ($op:ident, $meth_name:ident, $a:ty, $b:ty) => {
    impl_binop_Expr!($op, $meth_name, $a, $b, $a);
  };

  ($op:ident, $meth_name:ident, $a:ty, $b:ty, $r:ty) => {
    // expr OP expr
    impl<'a> ops::$op<Expr<$b>> for Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }
  };
}

// or
impl_binop_Expr!(BitOr, bitor, bool, bool);
impl_binop_Expr!(BitOr, bitor, V2<bool>, V2<bool>);
impl_binop_Expr!(BitOr, bitor, V2<bool>, bool);
impl_binop_Expr!(BitOr, bitor, V3<bool>, V3<bool>);
impl_binop_Expr!(BitOr, bitor, V3<bool>, bool);
impl_binop_Expr!(BitOr, bitor, V4<bool>, V4<bool>);
impl_binop_Expr!(BitOr, bitor, V4<bool>, bool);

// and
impl_binop_Expr!(BitAnd, bitand, bool, bool);
impl_binop_Expr!(BitAnd, bitand, V2<bool>, V2<bool>);
impl_binop_Expr!(BitAnd, bitand, V2<bool>, bool);
impl_binop_Expr!(BitAnd, bitand, V3<bool>, V3<bool>);
impl_binop_Expr!(BitAnd, bitand, V3<bool>, bool);
impl_binop_Expr!(BitAnd, bitand, V4<bool>, V4<bool>);
impl_binop_Expr!(BitAnd, bitand, V4<bool>, bool);

// xor
impl_binop_Expr!(BitXor, bitxor, bool, bool);
impl_binop_Expr!(BitXor, bitxor, V2<bool>, V2<bool>);
impl_binop_Expr!(BitXor, bitxor, V2<bool>, bool);
impl_binop_Expr!(BitXor, bitxor, V3<bool>, V3<bool>);
impl_binop_Expr!(BitXor, bitxor, V3<bool>, bool);
impl_binop_Expr!(BitXor, bitxor, V4<bool>, V4<bool>);
impl_binop_Expr!(BitXor, bitxor, V4<bool>, bool);

/// Run a macro on all supported types to generate the impl for them
///
/// The macro has to have to take two `ty` as argument and yield a `std::ops` trait implementor.
macro_rules! impl_binarith_Expr {
  ($op:ident, $meth_name:ident) => {
    impl_binop_Expr!($op, $meth_name, i32, i32);
    impl_binop_Expr!($op, $meth_name, V2<i32>, V2<i32>);
    impl_binop_Expr!($op, $meth_name, V2<i32>, i32);
    impl_binop_Expr!($op, $meth_name, V3<i32>, V3<i32>);
    impl_binop_Expr!($op, $meth_name, V3<i32>, i32);
    impl_binop_Expr!($op, $meth_name, V4<i32>, V4<i32>);
    impl_binop_Expr!($op, $meth_name, V4<i32>, i32);

    impl_binop_Expr!($op, $meth_name, u32, u32);
    impl_binop_Expr!($op, $meth_name, V2<u32>, V2<u32>);
    impl_binop_Expr!($op, $meth_name, V2<u32>, u32);
    impl_binop_Expr!($op, $meth_name, V3<u32>, V3<u32>);
    impl_binop_Expr!($op, $meth_name, V3<u32>, u32);
    impl_binop_Expr!($op, $meth_name, V4<u32>, V4<u32>);
    impl_binop_Expr!($op, $meth_name, V4<u32>, u32);

    impl_binop_Expr!($op, $meth_name, f32, f32);
    impl_binop_Expr!($op, $meth_name, V2<f32>, V2<f32>);
    impl_binop_Expr!($op, $meth_name, V2<f32>, f32);
    impl_binop_Expr!($op, $meth_name, V3<f32>, V3<f32>);
    impl_binop_Expr!($op, $meth_name, V3<f32>, f32);
    impl_binop_Expr!($op, $meth_name, V4<f32>, V4<f32>);
    impl_binop_Expr!($op, $meth_name, V4<f32>, f32);
  };
}

impl_binarith_Expr!(Add, add);
impl_binarith_Expr!(Sub, sub);
impl_binarith_Expr!(Mul, mul);
impl_binarith_Expr!(Div, div);

impl_binop_Expr!(Rem, rem, f32, f32);
impl_binop_Expr!(Rem, rem, V2<f32>, V2<f32>);
impl_binop_Expr!(Rem, rem, V2<f32>, f32);
impl_binop_Expr!(Rem, rem, V3<f32>, V3<f32>);
impl_binop_Expr!(Rem, rem, V3<f32>, f32);
impl_binop_Expr!(Rem, rem, V4<f32>, V4<f32>);
impl_binop_Expr!(Rem, rem, V4<f32>, f32);

impl_binop_Expr!(Mul, mul, M22, M22);
impl_binop_Expr!(Mul, mul, M22, V2<f32>, V2<f32>);
impl_binop_Expr!(Mul, mul, V2<f32>, M22, M22);
impl_binop_Expr!(Mul, mul, M33, M33);
impl_binop_Expr!(Mul, mul, M33, V3<f32>, V3<f32>);
impl_binop_Expr!(Mul, mul, V3<f32>, M33, M33);
impl_binop_Expr!(Mul, mul, M44, M44);
impl_binop_Expr!(Mul, mul, M44, V4<f32>, V4<f32>);
impl_binop_Expr!(Mul, mul, V4<f32>, M44, M44);

macro_rules! impl_binshift_Expr {
  ($op:ident, $meth_name:ident, $ty:ty) => {
    // expr OP expr
    impl ops::$op<Expr<u32>> for Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }
  };
}

/// Binary shift generating macro.
macro_rules! impl_binshifts_Expr {
  ($op:ident, $meth_name:ident) => {
    impl_binshift_Expr!($op, $meth_name, i32);
    impl_binshift_Expr!($op, $meth_name, V2<i32>);
    impl_binshift_Expr!($op, $meth_name, V3<i32>);
    impl_binshift_Expr!($op, $meth_name, V4<i32>);

    impl_binshift_Expr!($op, $meth_name, u32);
    impl_binshift_Expr!($op, $meth_name, V2<u32>);
    impl_binshift_Expr!($op, $meth_name, V3<u32>);
    impl_binshift_Expr!($op, $meth_name, V4<u32>);

    impl_binshift_Expr!($op, $meth_name, f32);
    impl_binshift_Expr!($op, $meth_name, V2<f32>);
    impl_binshift_Expr!($op, $meth_name, V3<f32>);
    impl_binshift_Expr!($op, $meth_name, V4<f32>);
  };
}

impl_binshifts_Expr!(Shl, shl);
impl_binshifts_Expr!(Shr, shr);

macro_rules! impl_From_Expr_scalar {
  ($t:ty, $q:ident) => {
    impl From<$t> for Expr<$t> {
      fn from(a: $t) -> Self {
        Self::new(ErasedExpr::$q(a))
      }
    }
  };
}

impl_From_Expr_scalar!(i32, LitInt);
impl_From_Expr_scalar!(u32, LitUInt);
impl_From_Expr_scalar!(f32, LitFloat);
impl_From_Expr_scalar!(bool, LitBool);

macro_rules! impl_From_Expr_vn {
  ($t:ty, $q:ident) => {
    impl From<$t> for Expr<$t> {
      fn from(a: $t) -> Self {
        Self::new(ErasedExpr::$q(a.0))
      }
    }
  };
}

impl_From_Expr_vn!(V2<i32>, LitInt2);
impl_From_Expr_vn!(V2<u32>, LitUInt2);
impl_From_Expr_vn!(V2<f32>, LitFloat2);
impl_From_Expr_vn!(V2<bool>, LitBool2);
impl_From_Expr_vn!(V3<i32>, LitInt3);
impl_From_Expr_vn!(V3<u32>, LitUInt3);
impl_From_Expr_vn!(V3<f32>, LitFloat3);
impl_From_Expr_vn!(V3<bool>, LitBool3);
impl_From_Expr_vn!(V4<i32>, LitInt4);
impl_From_Expr_vn!(V4<u32>, LitUInt4);
impl_From_Expr_vn!(V4<f32>, LitFloat4);
impl_From_Expr_vn!(V4<bool>, LitBool4);

impl<T, const N: usize> From<[T; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: Clone + ToType,
{
  fn from(array: [T; N]) -> Self {
    let array = array
      .iter()
      .cloned()
      .map(|t| Expr::from(t).erased)
      .collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: Clone + ToType,
{
  fn from(array: &'a [T; N]) -> Self {
    let array = array
      .iter()
      .cloned()
      .map(|t| Expr::from(t).erased)
      .collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

impl<T, const N: usize> From<[Expr<T>; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: ToType,
{
  fn from(array: [Expr<T>; N]) -> Self {
    let array = array.iter().cloned().map(|e| e.erased).collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

impl<'a, T, const N: usize> From<&'a [Expr<T>; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: ToType,
{
  fn from(array: &'a [Expr<T>; N]) -> Self {
    let array = array.iter().cloned().map(|e| e.erased).collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

/// Create 2D scalar vectors via different forms.
///
/// This macro allows to create 2D ([`V2`]) scalar vectors from two forms:
///
/// - `vec2!(xy)`, which acts as the cast operator. Only types `T` satisfying [`Vec2`] are castable.
/// - `vec2!(x, y)`, which builds a [`V2<T>`] for `x: T` and `y: T`.
#[macro_export]
macro_rules! vec2 {
  ($a:expr) => {
    todo!("vec2 cast operator missing");
  };

  ($xy:expr, $z:expr) => {{
    use $crate::types::Vec2 as _;
    $crate::expr::Expr::vec2(($crate::expr::Expr::from($xy), $crate::expr::Expr::from($z)))
  }};
}

/// Create 3D scalar vectors via different forms.
///
/// This macro allows to create 3D ([`V3`]) scalar vectors from several forms:
///
/// - `vec3!(xyz)`, which acts as the cast operator. Only types `T` satisfying [`Vec3`] are castable.
/// - `vec3!(xy, z)`, which builds a [`V3<T>`] with `xy` a value that can be turned into a `Expr<V2<T>>` and `z: T`
/// - `vec3!(x, y, z)`, which builds a [`V3<T>`] for `x: T`, `y: T` and `z: T`.
#[macro_export]
macro_rules! vec3 {
  ($a:expr) => {
    todo!("vec3 cast operator missing");
  };

  ($xy:expr, $z:expr) => {{
    use $crate::types::Vec3 as _;
    $crate::expr::Expr::vec3(($crate::expr::Expr::from($xy), $crate::expr::Expr::from($z)))
  }};

  ($x:expr, $y:expr, $z:expr) => {{
    use $crate::types::Vec3 as _;
    $crate::expr::Expr::vec3((
      $crate::expr::Expr::from($x),
      $crate::expr::Expr::from($y),
      $crate::expr::Expr::from($z),
    ))
  }};
}

/// Create 4D scalar vectors via different forms.
///
/// This macro allows to create 4D ([`V4`]) scalar vectors from several forms:
///
/// - `vec4!(xyzw)`, which acts as the cast operator. Only types `T` satisfying [`Vec4`] are castable.
/// - `vec4!(xyz, w)`, which builds a [`V4<T>`] with `xyz` a value that can be turned into a `Expr<V3<T>>` and `w: T`.
/// - `vec4!(xy, zw)`, which builds a [`V4<T>`] with `xy` and `zw` values that can be turned into `Expr<V3<T>>`.
/// - `vec4!(xy, z, w)`, which builds a [`V4<T>`] with `xy`, `z: T` and `w: T`.
/// - `vec4!(x, y, z, w)`, which builds a [`V3<T>`] for `x: T`, `y: T` and `z: T`.
#[macro_export]
macro_rules! vec4 {
  ($a:expr) => {
    todo!("vec4 cast operator missing");
  };

  ($xy:expr, $zw:expr) => {{
    use $crate::types::Vec4 as _;
    $crate::expr::Expr::vec4(($crate::expr::Expr::from($xy), $crate::expr::Expr::from($zw)))
  }};

  ($xy:expr, $z:expr, $w:expr) => {{
    use $crate::types::Vec4 as _;
    $crate::expr::Expr::vec4((
      $crate::expr::Expr::from($xy),
      $crate::expr::Expr::from($z),
      $crate::expr::Expr::from($w),
    ))
  }};

  ($x:expr, $y:expr, $z:expr, $w:expr) => {{
    use $crate::types::Vec4 as _;
    $crate::expr::Expr::vec4((
      $crate::expr::Expr::from($x),
      $crate::expr::Expr::from($y),
      $crate::expr::Expr::from($z),
      $crate::expr::Expr::from($w),
    ))
  }};
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    scope::{Scope, ScopeInstr},
    stdlib::Bounded as _,
    types::{Dim, PrimType},
  };

  #[test]
  fn unary() {
    let mut scope = Scope::<()>::new(0);

    let a = !Expr::from(true);
    let b = -Expr::from(3i32);
    let c = scope.var(a.clone());

    assert_eq!(
      a.erased,
      ErasedExpr::Not(Box::new(ErasedExpr::LitBool(true)))
    );
    assert_eq!(b.erased, ErasedExpr::Neg(Box::new(ErasedExpr::LitInt(3))));
    assert_eq!(c.erased, ErasedExpr::Var(ScopedHandle::fun_var(0, 0)));
  }

  #[test]
  fn binary() {
    let a = Expr::from(1i32) + Expr::from(2);

    assert_eq!(
      a.erased,
      ErasedExpr::Add(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2)),
      )
    );

    let a = Expr::from(1i32) - Expr::from(2);

    assert_eq!(
      a.erased,
      ErasedExpr::Sub(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2)),
      )
    );

    let a = Expr::from(1i32) * Expr::from(2);

    assert_eq!(
      a.erased,
      ErasedExpr::Mul(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2)),
      )
    );

    let a = Expr::from(1i32) / Expr::from(2);

    assert_eq!(
      a.erased,
      ErasedExpr::Div(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2)),
      )
    );
  }

  #[test]
  fn clone() {
    let a = Expr::from(1i32);
    let b = a.clone() + Expr::from(1);
    let c = a + Expr::from(1);

    assert_eq!(b.erased, c.erased);
  }

  #[test]
  fn var() {
    let mut scope = Scope::<()>::new(0);

    let x = scope.var(Expr::from(0));
    let y = scope.var(Expr::from(1u32));
    let z = scope.var(vec3![false, true, false]);

    assert_eq!(x.erased, ErasedExpr::Var(ScopedHandle::fun_var(0, 0)));
    assert_eq!(y.erased, ErasedExpr::Var(ScopedHandle::fun_var(0, 1)));
    assert_eq!(
      z.erased,
      ErasedExpr::Var(ScopedHandle::fun_var(0, 2).into())
    );
    assert_eq!(scope.erased.instructions.len(), 3);
    assert_eq!(
      scope.erased.instructions[0],
      ScopeInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Int(Dim::Scalar),
          array_dims: Vec::new(),
        },
        handle: ScopedHandle::fun_var(0, 0),
        init_value: ErasedExpr::LitInt(0),
      }
    );
    assert_eq!(
      scope.erased.instructions[1],
      ScopeInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::UInt(Dim::Scalar),
          array_dims: Vec::new(),
        },
        handle: ScopedHandle::fun_var(0, 1),
        init_value: ErasedExpr::LitUInt(1),
      }
    );
    assert_eq!(
      scope.erased.instructions[2],
      ScopeInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Bool(Dim::D3),
          array_dims: Vec::new(),
        },
        handle: ScopedHandle::fun_var(0, 2),
        init_value: ErasedExpr::FunCall(
          ErasedFunHandle::Vec3,
          vec![
            Expr::from(false).erased,
            Expr::from(true).erased,
            Expr::from(false).erased
          ]
        )
      }
    );
  }

  #[test]
  fn min_max_clamp() {
    let a = Expr::from(1i32);
    let b = Expr::from(2);
    let c = Expr::from(3);

    assert_eq!(
      a.clone().min(b.clone()).erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Min,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)],
      )
    );

    assert_eq!(
      a.clone().max(b.clone()).erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Max,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)],
      )
    );

    assert_eq!(
      a.clone().clamp(b, c).erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Clamp,
        vec![
          ErasedExpr::LitInt(1),
          ErasedExpr::LitInt(2),
          ErasedExpr::LitInt(3)
        ],
      )
    );
  }

  #[test]
  fn array_creation() {
    let _ = Expr::from([1, 2, 3]);
    let _ = Expr::from(&[1, 2, 3]);
    let _ = Expr::from([vec2!(1., 2.)]);
    let two_d = Expr::from([[1, 2], [3, 4]]);

    assert_eq!(
      two_d.erased,
      ErasedExpr::Array(
        <[[i32; 2]; 2] as ToType>::ty(),
        vec![
          ErasedExpr::Array(
            <[i32; 2] as ToType>::ty(),
            vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)]
          ),
          ErasedExpr::Array(
            <[i32; 2] as ToType>::ty(),
            vec![ErasedExpr::LitInt(3), ErasedExpr::LitInt(4)]
          )
        ]
      )
    );
  }

  #[test]
  fn vec3_ctor() {
    let xy = vec2![1., 2.];
    let xyz2 = vec3![xy.clone(), 3.];
    let xyz3 = vec3![1., 2., 3.];

    assert_eq!(
      xyz2.erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Vec3,
        vec![xy.erased, Expr::from(3.).erased]
      )
    );

    assert_eq!(
      xyz3.erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Vec3,
        vec![
          Expr::from(1.).erased,
          Expr::from(2.).erased,
          Expr::from(3.).erased
        ]
      )
    );
  }

  #[test]
  fn vec4_ctor() {
    let xy = vec2![1., 2.];
    let xyzw22 = vec4!(xy.clone(), xy.clone());
    let xyzw211 = vec4!(xy.clone(), 3., 4.);
    let xyzw31 = vec4!(vec3!(1., 2., 3.), 4.);
    let xyzw4 = vec4!(1., 2., 3., 4.);

    assert_eq!(
      xyzw22.erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Vec4,
        vec![xy.clone().erased, xy.clone().erased]
      )
    );

    assert_eq!(
      xyzw211.erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Vec4,
        vec![
          xy.clone().erased,
          Expr::from(3.).erased,
          Expr::from(4.).erased
        ]
      )
    );

    assert_eq!(
      xyzw31.erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Vec4,
        vec![vec3!(1., 2., 3.).erased, Expr::from(4.).erased]
      )
    );

    assert_eq!(
      xyzw4.erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Vec4,
        vec![
          Expr::from(1.).erased,
          Expr::from(2.).erased,
          Expr::from(3.).erased,
          Expr::from(4.).erased
        ]
      )
    );
  }
}
