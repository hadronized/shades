use crate::{types::{V2, V3, V4}, fun::ErasedFunHandle, expr::{ErasedExpr, Expr}};

// standard library
pub trait Trigonometry {
  fn radians(&self) -> Self;

  fn degrees(&self) -> Self;

  fn sin(&self) -> Self;

  fn cos(&self) -> Self;

  fn tan(&self) -> Self;

  fn asin(&self) -> Self;

  fn acos(&self) -> Self;

  fn atan(&self) -> Self;

  fn sinh(&self) -> Self;

  fn cosh(&self) -> Self;

  fn tanh(&self) -> Self;

  fn asinh(&self) -> Self;

  fn acosh(&self) -> Self;

  fn atanh(&self) -> Self;
}

macro_rules! impl_Trigonometry {
  ($t:ty) => {
    impl Trigonometry for Expr<$t> {
      fn radians(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Radians,
          vec![self.erased.clone()],
        ))
      }

      fn degrees(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Degrees,
          vec![self.erased.clone()],
        ))
      }

      fn sin(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Sin,
          vec![self.erased.clone()],
        ))
      }

      fn cos(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Cos,
          vec![self.erased.clone()],
        ))
      }

      fn tan(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Tan,
          vec![self.erased.clone()],
        ))
      }

      fn asin(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::ASin,
          vec![self.erased.clone()],
        ))
      }

      fn acos(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::ACos,
          vec![self.erased.clone()],
        ))
      }

      fn atan(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::ATan,
          vec![self.erased.clone()],
        ))
      }

      fn sinh(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::SinH,
          vec![self.erased.clone()],
        ))
      }

      fn cosh(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::CosH,
          vec![self.erased.clone()],
        ))
      }

      fn tanh(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::TanH,
          vec![self.erased.clone()],
        ))
      }

      fn asinh(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::ASinH,
          vec![self.erased.clone()],
        ))
      }

      fn acosh(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::ACosH,
          vec![self.erased.clone()],
        ))
      }

      fn atanh(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::ATanH,
          vec![self.erased.clone()],
        ))
      }
    }
  };
}

impl_Trigonometry!(f32);
impl_Trigonometry!(V2<f32>);
impl_Trigonometry!(V3<f32>);
impl_Trigonometry!(V4<f32>);

pub trait Exponential: Sized {
  fn pow(&self, p: impl Into<Self>) -> Self;

  fn exp(&self) -> Self;

  fn exp2(&self) -> Self;

  fn log(&self) -> Self;

  fn log2(&self) -> Self;

  fn sqrt(&self) -> Self;

  fn isqrt(&self) -> Self;
}

macro_rules! impl_Exponential {
  ($t:ty) => {
    impl Exponential for Expr<$t> {
      fn pow(&self, p: impl Into<Self>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Pow,
          vec![self.erased.clone(), p.into().erased],
        ))
      }

      fn exp(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Exp,
          vec![self.erased.clone()],
        ))
      }

      fn exp2(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Exp2,
          vec![self.erased.clone()],
        ))
      }

      fn log(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Log,
          vec![self.erased.clone()],
        ))
      }

      fn log2(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Log2,
          vec![self.erased.clone()],
        ))
      }

      fn sqrt(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Sqrt,
          vec![self.erased.clone()],
        ))
      }

      fn isqrt(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::InverseSqrt,
          vec![self.erased.clone()],
        ))
      }
    }
  };
}

impl_Exponential!(f32);
impl_Exponential!(V2<f32>);
impl_Exponential!(V3<f32>);
impl_Exponential!(V4<f32>);

pub trait Relative {
  fn abs(&self) -> Self;

  fn sign(&self) -> Self;
}

macro_rules! impl_Relative {
  ($t:ty) => {
    impl Relative for Expr<$t> {
      fn abs(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Abs,
          vec![self.erased.clone()],
        ))
      }

      fn sign(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Sign,
          vec![self.erased.clone()],
        ))
      }
    }
  };
}

impl_Relative!(i32);
impl_Relative!(V2<i32>);
impl_Relative!(V3<i32>);
impl_Relative!(V4<i32>);
impl_Relative!(f32);
impl_Relative!(V2<f32>);
impl_Relative!(V3<f32>);
impl_Relative!(V4<f32>);

pub trait Floating {
  fn floor(&self) -> Self;

  fn trunc(&self) -> Self;

  fn round(&self) -> Self;

  fn ceil(&self) -> Self;

  fn fract(&self) -> Self;
}

macro_rules! impl_Floating {
  ($t:ty) => {
    impl Floating for Expr<$t> {
      fn floor(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Floor,
          vec![self.erased.clone()],
        ))
      }

      fn trunc(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Trunc,
          vec![self.erased.clone()],
        ))
      }

      fn round(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Round,
          vec![self.erased.clone()],
        ))
      }

      fn ceil(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Ceil,
          vec![self.erased.clone()],
        ))
      }

      fn fract(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Fract,
          vec![self.erased.clone()],
        ))
      }
    }
  };
}

impl_Floating!(f32);
impl_Floating!(V2<f32>);
impl_Floating!(V3<f32>);
impl_Floating!(V4<f32>);

pub trait Bounded: Sized {
  fn min(&self, rhs: impl Into<Self>) -> Self;

  fn max(&self, rhs: impl Into<Self>) -> Self;

  fn clamp(&self, min_value: impl Into<Self>, max_value: impl Into<Self>) -> Self;
}

macro_rules! impl_Bounded {
  ($t:ty) => {
    impl Bounded for Expr<$t> {
      fn min(&self, rhs: impl Into<Self>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Min,
          vec![self.erased.clone(), rhs.into().erased],
        ))
      }

      fn max(&self, rhs: impl Into<Self>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Max,
          vec![self.erased.clone(), rhs.into().erased],
        ))
      }

      fn clamp(&self, min_value: impl Into<Self>, max_value: impl Into<Self>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Clamp,
          vec![
            self.erased.clone(),
            min_value.into().erased,
            max_value.into().erased,
          ],
        ))
      }
    }
  };
}

impl_Bounded!(i32);
impl_Bounded!(V2<i32>);
impl_Bounded!(V3<i32>);
impl_Bounded!(V4<i32>);

impl_Bounded!(u32);
impl_Bounded!(V2<u32>);
impl_Bounded!(V3<u32>);
impl_Bounded!(V4<u32>);

impl_Bounded!(f32);
impl_Bounded!(V2<f32>);
impl_Bounded!(V3<f32>);
impl_Bounded!(V4<f32>);

impl_Bounded!(bool);
impl_Bounded!(V2<bool>);
impl_Bounded!(V3<bool>);
impl_Bounded!(V4<bool>);

pub trait Mix<RHS>: Sized {
  fn mix(&self, y: impl Into<Self>, a: RHS) -> Self;

  fn step(&self, edge: RHS) -> Self;

  fn smooth_step(&self, edge_a: RHS, edge_b: RHS) -> Self;
}

macro_rules! impl_Mix {
  ($t:ty, $q:ty) => {
    impl Mix<Expr<$q>> for Expr<$t> {
      fn mix(&self, y: impl Into<Self>, a: Expr<$q>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Mix,
          vec![self.erased.clone(), y.into().erased, a.erased],
        ))
      }

      fn step(&self, edge: Expr<$q>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Step,
          vec![self.erased.clone(), edge.erased],
        ))
      }

      fn smooth_step(&self, edge_a: Expr<$q>, edge_b: Expr<$q>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::SmoothStep,
          vec![self.erased.clone(), edge_a.erased, edge_b.erased],
        ))
      }
    }
  };
}

impl_Mix!(f32, f32);
impl_Mix!(V2<f32>, f32);
impl_Mix!(V2<f32>, V2<f32>);

impl_Mix!(V3<f32>, f32);
impl_Mix!(V3<f32>, V3<f32>);

impl_Mix!(V4<f32>, f32);
impl_Mix!(V4<f32>, V4<f32>);

pub trait FloatingExt {
  type BoolExpr;

  fn is_nan(&self) -> Self::BoolExpr;

  fn is_inf(&self) -> Self::BoolExpr;
}

macro_rules! impl_FloatingExt {
  ($t:ty, $bool_expr:ty) => {
    impl FloatingExt for Expr<$t> {
      type BoolExpr = Expr<$bool_expr>;

      fn is_nan(&self) -> Self::BoolExpr {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::IsNan,
          vec![self.erased.clone()],
        ))
      }

      fn is_inf(&self) -> Self::BoolExpr {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::IsInf,
          vec![self.erased.clone()],
        ))
      }
    }
  };
}

impl_FloatingExt!(f32, bool);
impl_FloatingExt!(V2<f32>, V2<bool>);
impl_FloatingExt!(V3<f32>, V3<bool>);
impl_FloatingExt!(V4<f32>, V4<bool>);

pub trait Geometry: Sized {
  type LengthExpr;

  fn length(&self) -> Self::LengthExpr;

  fn distance(&self, other: impl Into<Self>) -> Self::LengthExpr;

  fn dot(&self, other: impl Into<Self>) -> Self::LengthExpr;

  fn cross(&self, other: impl Into<Self>) -> Self;

  fn normalize(&self) -> Self;

  fn face_forward(&self, normal: impl Into<Self>, reference: impl Into<Self>) -> Self;

  fn reflect(&self, normal: impl Into<Self>) -> Self;

  fn refract(&self, normal: impl Into<Self>, eta: impl Into<Expr<f32>>) -> Self;
}

macro_rules! impl_Geometry {
  ($t:ty, $l:ty) => {
    impl Geometry for Expr<$t> {
      type LengthExpr = Expr<$l>;

      fn length(&self) -> Self::LengthExpr {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Length,
          vec![self.erased.clone()],
        ))
      }

      fn distance(&self, other: impl Into<Self>) -> Self::LengthExpr {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Distance,
          vec![self.erased.clone(), other.into().erased],
        ))
      }

      fn dot(&self, other: impl Into<Self>) -> Self::LengthExpr {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Dot,
          vec![self.erased.clone(), other.into().erased],
        ))
      }

      fn cross(&self, other: impl Into<Self>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Cross,
          vec![self.erased.clone(), other.into().erased],
        ))
      }

      fn normalize(&self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Normalize,
          vec![self.erased.clone()],
        ))
      }

      fn face_forward(&self, normal: impl Into<Self>, reference: impl Into<Self>) -> Self {
        // note: this function call is super weird as the normal and incident (i.e. self) arguments are swapped
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::FaceForward,
          vec![
            normal.into().erased,
            self.erased.clone(),
            reference.into().erased,
          ],
        ))
      }

      fn reflect(&self, normal: impl Into<Self>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Reflect,
          vec![self.erased.clone(), normal.into().erased],
        ))
      }

      fn refract(&self, normal: impl Into<Self>, eta: impl Into<Expr<f32>>) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Refract,
          vec![self.erased.clone(), normal.into().erased, eta.into().erased],
        ))
      }
    }
  };
}

impl_Geometry!(V2<f32>, f32);
impl_Geometry!(V3<f32>, f32);
impl_Geometry!(V4<f32>, f32);
