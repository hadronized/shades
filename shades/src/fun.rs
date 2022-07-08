//! Function definition, arguments, return and body.

use crate::{
  expr::{ErasedExpr, Expr},
  scope::ErasedScope,
  types::{ToType, Type},
};
use std::marker::PhantomData;

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

impl<R, A> FunHandle<Expr<R>, Expr<A>> {
  /// Create an expression representing a function call to this function.
  ///
  /// See the documentation of [`FunHandle`] for examples.
  pub fn call(&self, a: Expr<A>) -> Expr<R> {
    Expr::new(ErasedExpr::FunCall(self.erased.clone(), vec![a.erased]))
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
