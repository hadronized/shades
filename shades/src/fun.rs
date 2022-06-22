use std::marker::PhantomData;

use crate::{
  expr::{ErasedExpr, Expr},
  scope::ErasedScope,
  types::{ToType, Type},
};

/// Function return.
///
/// This type represents a function return and is used to annotate values that can be returned from functions (i.e.
/// expressions).
#[derive(Clone, Debug, PartialEq)]
pub struct Return {
  pub(crate) erased: ErasedReturn,
}

/// Erased return.
///
/// Either `Void` (i.e. `void`) or an expression. The type of the expression is also present for convenience.
#[derive(Clone, Debug, PartialEq)]
pub enum ErasedReturn {
  Void,
  Expr(Type, ErasedExpr),
}

impl From<()> for Return {
  fn from(_: ()) -> Self {
    Return {
      erased: ErasedReturn::Void,
    }
  }
}

impl<T> From<Expr<T>> for Return
where
  T: ToType,
{
  fn from(expr: Expr<T>) -> Self {
    Return {
      erased: ErasedReturn::Expr(T::ty(), expr.erased),
    }
  }
}

/// An opaque function handle, used to call user-defined functions.
///
/// Function handles are created with the [`StageBuilder::fun`] function, introducing new functions in the EDSL. You
/// can then call the functions in the context of generating new expressions, returning them or creating variables.
///
/// Injecting a function call in the EDSL is done via two current mechanisms:
///
/// - Either call the [`FunHandle::call`] method:
///   - It is a function without argument if the represented function doesn’t have any argument.
///   - It is a unary function if the represented function has a single argument.
///   - It takes a tuple encapsulating the arguments for a n-ary represented function.
/// - **On nightly only**, you can enable the `fun-call` feature-gate and calling the function will do the same thing
///   as [`FunHandle::call`]. However, because of how the various [`FnOnce`], [`FnMut`] and [`Fn`] traits are made,
///   functions taking several arguments take them as separate arguments as if it was a regular Rust function (they
///   don’t take a tuple as with the [`FunHandle::call`] method).
///
/// # Examples
///
/// A unary function squaring its argument, the regular way:
///
/// ```
/// # use shades::StageBuilder;
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{Expr, FunHandle, Scope, lit, vec2};
///
/// let square = s.fun(|s: &mut Scope<Expr<i32>>, a: Expr<i32>| {
///   &a * &a
/// });
///
/// s.main_fun(|s: &mut Scope<()>| {
///   // call square with 3 and bind the result to a variable
///   let squared = s.var(square.call(lit!(3)));
/// })
/// # });
/// ```
///
/// The same function but with the `fun-call` feature-gate enabled:
///
/// ```
/// # use shades::StageBuilder;
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{Expr, Scope, lit};
///
/// let square = s.fun(|s: &mut Scope<Expr<i32>>, a: Expr<i32>| {
///   &a * &a
/// });
///
/// s.main_fun(|s: &mut Scope<()>| {
///   // call square with 3 and bind the result to a variable
///   # #[cfg(feature = "fun-call")]
///   let squared = s.var(square(lit!(3)));
/// })
/// # });
/// ```
///
/// A function taking two 3D vectors and a floating scalar and returning their linear interpolation, called with
/// three arguments:
///
/// ```
/// # use shades::StageBuilder;
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{Expr, Mix as _, Scope, V3, lit, vec3};
///
/// let lerp = s.fun(|s: &mut Scope<Expr<V3<f32>>>, a: Expr<V3<f32>>, b: Expr<V3<f32>>, t: Expr<f32>| {
///   a.mix(b, t)
/// });
///
/// s.main_fun(|s: &mut Scope<()>| {
///   # #[cfg(feature = "fun-call")]
///   let a = vec3!(0., 0., 0.);
///   let b = vec3!(1., 1., 1.);
///
///   // call lerp here and bind it to a local variable
/// # #[cfg(feature = "fun-call")]
///   let result = s.var(lerp(a, b, lit!(0.75)));
/// })
/// # });
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct FunHandle<R, A> {
  pub(crate) erased: ErasedFunHandle,
  _phantom: PhantomData<(R, A)>,
}

impl<R, A> FunHandle<R, A> {
  pub(crate) fn new(erased: ErasedFunHandle) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

impl<R> FunHandle<Expr<R>, ()> {
  /// Create an expression representing a function call to this function.
  ///
  /// See the documentation of [`FunHandle`] for examples.
  pub fn call(&self) -> Expr<R> {
    Expr::new(ErasedExpr::FunCall(self.erased.clone(), Vec::new()))
  }
}

#[cfg(feature = "fun-call")]
impl<R> FnOnce<()> for FunHandle<Expr<R>, ()> {
  type Output = Expr<R>;

  extern "rust-call" fn call_once(self, _: ()) -> Self::Output {
    self.call()
  }
}

#[cfg(feature = "fun-call")]
impl<R> FnMut<()> for FunHandle<Expr<R>, ()> {
  extern "rust-call" fn call_mut(&mut self, _: ()) -> Self::Output {
    self.call()
  }
}

#[cfg(feature = "fun-call")]
impl<R> Fn<()> for FunHandle<Expr<R>, ()> {
  extern "rust-call" fn call(&self, _: ()) -> Self::Output {
    self.call()
  }
}

impl<R, A> FunHandle<Expr<R>, Expr<A>> {
  /// Create an expression representing a function call to this function.
  ///
  /// See the documentation of [`FunHandle`] for examples.
  pub fn call(&self, a: Expr<A>) -> Expr<R> {
    Expr::new(ErasedExpr::FunCall(self.erased.clone(), vec![a.erased]))
  }
}

#[cfg(feature = "fun-call")]
impl<R, A> FnOnce<(Expr<A>,)> for FunHandle<Expr<R>, Expr<A>> {
  type Output = Expr<R>;

  extern "rust-call" fn call_once(self, a: (Expr<A>,)) -> Self::Output {
    self.call(a.0)
  }
}

#[cfg(feature = "fun-call")]
impl<R, A> FnMut<(Expr<A>,)> for FunHandle<Expr<R>, Expr<A>> {
  extern "rust-call" fn call_mut(&mut self, a: (Expr<A>,)) -> Self::Output {
    self.call(a.0)
  }
}

#[cfg(feature = "fun-call")]
impl<R, A> Fn<(Expr<A>,)> for FunHandle<Expr<R>, Expr<A>> {
  extern "rust-call" fn call(&self, a: (Expr<A>,)) -> Self::Output {
    self.call(a.0)
  }
}

// the first stage must be named S0
macro_rules! impl_FunCall {
  ( $( ( $arg_name:ident, $arg_ty:ident ) ),*) => {
    impl<R, $($arg_ty),*> FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      /// Create an expression representing a function call to this function.
      ///
      /// See the documentation of [`FunHandle`] for examples.
      pub fn call(&self, $($arg_name : Expr<$arg_ty>),*) -> Expr<R> {
        Expr::new(ErasedExpr::FunCall(self.erased.clone(), vec![$($arg_name.erased),*]))
      }
    }

    #[cfg(feature = "fun-call")]
    impl<R, $($arg_ty),*> FnOnce<($(Expr<$arg_ty>),*)> for FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      type Output = Expr<R>;

      extern "rust-call" fn call_once(self, ($($arg_name),*): ($(Expr<$arg_ty>),*)) -> Self::Output {
        self.call($($arg_name),*)
      }
    }

    #[cfg(feature = "fun-call")]
    impl<R, $($arg_ty),*> FnMut<($(Expr<$arg_ty>),*)> for FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      extern "rust-call" fn call_mut(&mut self, ($($arg_name),*): ($(Expr<$arg_ty>),*)) -> Self::Output {
        self.call($($arg_name),*)
      }
    }

    #[cfg(feature = "fun-call")]
    impl<R, $($arg_ty),*> Fn<($(Expr<$arg_ty>),*)> for FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      extern "rust-call" fn call(&self, ($($arg_name),*): ($(Expr<$arg_ty>),*)) -> Self::Output {
        self.call($($arg_name),*)
      }
    }
  };
}

// implement function calls for Expr up to 16 arguments
macro_rules! impl_FunCall_rec {
  ( ( $a:ident, $b:ident ) , ( $x:ident, $y:ident )) => {
    impl_FunCall!(($a, $b), ($x, $y));
  };

  ( ( $a:ident, $b:ident ) , ( $x: ident, $y: ident ) , $($r:tt)* ) => {
    impl_FunCall_rec!(($a, $b), $($r)*);
    impl_FunCall!(($a, $b), ($x, $y), $($r)*);
  };
}
impl_FunCall_rec!(
  (a, A),
  (b, B),
  (c, C),
  (d, D),
  (e, E),
  (f, F),
  (g, G),
  (h, H),
  (i, I),
  (j, J),
  (k, K),
  (l, L),
  (m, M),
  (n, N),
  (o, O),
  (p, P)
);

/// Erased function handle.
#[derive(Clone, Debug, PartialEq)]
pub enum ErasedFunHandle {
  // cast operators
  Vec2,
  Vec3,
  Vec4,
  // trigonometry
  Radians,
  Degrees,
  Sin,
  Cos,
  Tan,
  ASin,
  ACos,
  ATan,
  SinH,
  CosH,
  TanH,
  ASinH,
  ACosH,
  ATanH,
  // exponential
  Pow,
  Exp,
  Exp2,
  Log,
  Log2,
  Sqrt,
  InverseSqrt,
  // common
  Abs,
  Sign,
  Floor,
  Trunc,
  Round,
  RoundEven,
  Ceil,
  Fract,
  Min,
  Max,
  Clamp,
  Mix,
  Step,
  SmoothStep,
  IsNan,
  IsInf,
  FloatBitsToInt,
  IntBitsToFloat,
  UIntBitsToFloat,
  FMA,
  Frexp,
  Ldexp,
  // floating-point pack and unpack functions
  PackUnorm2x16,
  PackSnorm2x16,
  PackUnorm4x8,
  PackSnorm4x8,
  UnpackUnorm2x16,
  UnpackSnorm2x16,
  UnpackUnorm4x8,
  UnpackSnorm4x8,
  PackHalf2x16,
  UnpackHalf2x16,
  // geometry functions
  Length,
  Distance,
  Dot,
  Cross,
  Normalize,
  FaceForward,
  Reflect,
  Refract,
  // matrix functions
  // TODO
  // vector relational functions
  VLt,
  VLte,
  VGt,
  VGte,
  VEq,
  VNeq,
  VAny,
  VAll,
  VNot,
  // integer functions
  UAddCarry,
  USubBorrow,
  UMulExtended,
  IMulExtended,
  BitfieldExtract,
  BitfieldInsert,
  BitfieldReverse,
  BitCount,
  FindLSB,
  FindMSB,
  // texture functions
  // TODO
  // geometry shader functions
  EmitStreamVertex,
  EndStreamPrimitive,
  EmitVertex,
  EndPrimitive,
  // fragment processing functions
  DFDX,
  DFDY,
  DFDXFine,
  DFDYFine,
  DFDXCoarse,
  DFDYCoarse,
  FWidth,
  FWidthFine,
  FWidthCoarse,
  InterpolateAtCentroid,
  InterpolateAtSample,
  InterpolateAtOffset,
  // shader invocation control functions
  Barrier,
  MemoryBarrier,
  MemoryBarrierAtomic,
  MemoryBarrierBuffer,
  MemoryBarrierShared,
  MemoryBarrierImage,
  GroupMemoryBarrier,
  // shader invocation group functions
  AnyInvocation,
  AllInvocations,
  AllInvocationsEqual,
  UserDefined(u16),
}

/// A function definition.
///
/// Function definitions contain the information required to know how to represent a function’s arguments, return type
/// and its body.
#[derive(Debug)]
pub struct FunDef<R, A> {
  pub(crate) erased: ErasedFun,
  _phantom: PhantomData<(R, A)>,
}

impl<R, A> FunDef<R, A> {
  pub fn new(erased: ErasedFun) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

/// Erased function definition.
#[derive(Debug)]
pub struct ErasedFun {
  pub(crate) args: Vec<Type>,
  pub(crate) ret_ty: Option<Type>,
  pub(crate) scope: ErasedScope,
}

impl ErasedFun {
  pub fn new(args: Vec<Type>, ret_ty: Option<Type>, scope: ErasedScope) -> Self {
    Self {
      args,
      ret_ty,
      scope,
    }
  }
}
