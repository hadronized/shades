use crate::{
  expr::{ErasedExpr, Expr},
  fun::ErasedFunHandle,
};
use std::iter::once;

macro_rules! make_vn {
  ($t:ident, $dim:expr) => {
    /// Scalar vectors.
    ///
    /// Scalar vectors come into three flavors, based on the dimension used:
    ///
    /// - Two dimensions (2D): [`V2<T>`].
    /// - Three dimensions (3D): [`V3<T>`].
    /// - Four dimensions (4D): [`V4<T>`].
    ///
    /// Each type implements the [`From`] trait for sized array. For instance, if you want to make a `V3<f32>` from
    /// constants / literals, you can simply use the implementor `From<[f32; 3]> for V3<f32>`.
    ///
    /// A builder macro version exists for each flavor:
    ///
    /// - [`vec2`]: build [`V2<T>`].
    /// - [`vec3`]: build [`V3<T>`].
    /// - [`vec4`]: build [`V4<T>`].
    ///
    /// Those three macros can also be used with literals.
    #[derive(Clone, Debug, PartialEq)]
    pub struct $t<T>(pub [T; $dim]);

    impl<T> From<[T; $dim]> for $t<T> {
      fn from(a: [T; $dim]) -> Self {
        Self(a)
      }
    }
  };
}

make_vn!(V2, 2);
make_vn!(V3, 3);
make_vn!(V4, 4);

/// Matrix wrapper.
///
/// This type represents a matrix of a given dimension, deduced from the wrapped type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Matrix<T>(pub T);

impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<[[T; N]; M]> {
  fn from(a: [[T; N]; M]) -> Self {
    Matrix(a)
  }
}

macro_rules! make_mat_ty {
  ($t:ident, $lit:ident, $m:expr, $n:expr, $mdim:ident) => {
    pub type $t = Matrix<[[f32; $n]; $m]>;

    impl ToPrimType for Matrix<[[f32; $n]; $m]> {
      const PRIM_TYPE: PrimType = PrimType::Matrix(MatrixDim::$mdim);
    }

    impl From<Matrix<[[f32; $n]; $m]>> for Expr<Matrix<[[f32; $n]; $m]>> {
      fn from(matrix: Matrix<[[f32; $n]; $m]>) -> Self {
        Self::new(ErasedExpr::$lit(matrix))
      }
    }
  };
}

make_mat_ty!(M22, LitM22, 2, 2, D22);
// make_mat_ty!(M23, LitM23, 2, 3, D23);
// make_mat_ty!(M24, LitM24, 2, 4, D24);
// make_mat_ty!(M32, LitM32, 3, 2, D32);
make_mat_ty!(M33, LitM33, 3, 3, D33);
// make_mat_ty!(M34, LitM34, 3, 4, D34);
// make_mat_ty!(M42, LitM42, 4, 2, D42);
// make_mat_ty!(M43, LitM43, 4, 3, D43);
make_mat_ty!(M44, LitM44, 4, 4, D44);

/// Matrix dimension.
///
/// Matrices can have several dimensions. Most of the time, you will be interested in squared dimensions, e.g. 2×2, 3×3
/// and 4×4. However, other dimensions exist.
///
/// > Note: matrices are expressed in column-major.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MatrixDim {
  /// Squared 2 dimension.
  D22,
  /// 2×3 dimension.
  D23,
  /// 2×4 dimension.
  D24,
  /// 3×2 dimension.
  D32,
  /// Squared 3 dimension.
  D33,
  /// 3×4 dimension.
  D34,
  /// 4×2 dimension.
  D42,
  /// 4×3 dimension.
  D43,
  /// Squared 4 dimension.
  D44,
}

/// Dimension of a primitive type.
///
/// Primitive types currently can have one of four dimension:
///
/// - [`Dim::Scalar`]: designates a scalar value.
/// - [`Dim::D2`]: designates a 2D vector.
/// - [`Dim::D3`]: designates a 3D vector.
/// - [`Dim::D4`]: designates a 4D vector.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Dim {
  /// Scalar value.
  Scalar,

  /// 2D vector.
  D2,

  /// 3D vector.
  D3,

  /// 4D vector.
  D4,
}

/// Type representation — akin to [`PrimType`] glued with array dimensions, if any.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Type {
  /// Primitive type, representing a type without array dimensions.
  pub(crate) prim_ty: PrimType,

  /// Array dimensions, if any.
  ///
  /// Dimensions are sorted from outer to inner; i.e. `[[i32; N]; M]`’s dimensions is encoded as `vec![M, N]`.
  pub(crate) array_dims: Vec<usize>,
}

/// Primitive supported types.
///
/// Types without array dimensions are known as _primitive types_ and are exhaustively constructed thanks to
/// [`PrimType`].
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PrimType {
  /// An integral type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  Int(Dim),

  /// An unsigned integral type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  UInt(Dim),

  /// An floating type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  Float(Dim),

  /// A boolean type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  Bool(Dim),

  /// A N×M floating matrix.
  ///
  /// The [`MatrixDim`] provides the information required to know the exact dimension of the matrix.
  Matrix(MatrixDim),
}

/// Class of types that are recognized by the EDSL.
///
/// Any type implementing this type family is _representable_ in the EDSL.
pub trait ToPrimType {
  /// Mapped primitive type.
  const PRIM_TYPE: PrimType;
}

macro_rules! impl_ToPrimType {
  ($t:ty, $q:ident, $d:ident) => {
    impl ToPrimType for $t {
      const PRIM_TYPE: PrimType = PrimType::$q(Dim::$d);
    }
  };
}

impl_ToPrimType!(i32, Int, Scalar);
impl_ToPrimType!(u32, UInt, Scalar);
impl_ToPrimType!(f32, Float, Scalar);
impl_ToPrimType!(bool, Bool, Scalar);
impl_ToPrimType!(V2<i32>, Int, D2);
impl_ToPrimType!(V2<u32>, UInt, D2);
impl_ToPrimType!(V2<f32>, Float, D2);
impl_ToPrimType!(V2<bool>, Bool, D2);
impl_ToPrimType!(V3<i32>, Int, D3);
impl_ToPrimType!(V3<u32>, UInt, D3);
impl_ToPrimType!(V3<f32>, Float, D3);
impl_ToPrimType!(V3<bool>, Bool, D3);
impl_ToPrimType!(V4<i32>, Int, D4);
impl_ToPrimType!(V4<u32>, UInt, D4);
impl_ToPrimType!(V4<f32>, Float, D4);
impl_ToPrimType!(V4<bool>, Bool, D4);

/// Represent a type (primitive type and array dimension) in the EDSL.
///
/// Any type implementing [`ToType`] is representable in the EDSL. Any type implementing [`ToPrimType`] automatically
/// also implements [`ToType`].
pub trait ToType {
  fn ty() -> Type;
}

impl<T> ToType for T
where
  T: ToPrimType,
{
  fn ty() -> Type {
    Type {
      prim_ty: T::PRIM_TYPE,
      array_dims: Vec::new(),
    }
  }
}

impl<T, const N: usize> ToType for [T; N]
where
  T: ToType,
{
  fn ty() -> Type {
    let Type {
      prim_ty,
      array_dims,
    } = T::ty();
    let array_dims = once(N).chain(array_dims).collect();

    Type {
      prim_ty,
      array_dims,
    }
  }
}

/// Trait allowing to create 2D scalar vector ([`V2`])constructors.
/// 2D scalar vectors can be created from either two sole scalars or a single 2D scalar vector (identity function).
///
///
/// The `A` type variable represents the arguments type. In the case of several arguments, tuples are used.
///
/// You are advised to use the [`vec2!`](vec2) macro instead as the interface of this function is not really
/// user-friendly.
pub trait Vec2<A> {
  /// Make a [`V2`] from `A`.
  fn vec2(args: A) -> Self;
}

impl<T> Vec2<(Expr<T>, Expr<T>)> for Expr<V2<T>> {
  fn vec2(args: (Expr<T>, Expr<T>)) -> Self {
    let (x, y) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec2,
      vec![x.erased, y.erased],
    ))
  }
}

/// Trait allowing to create 3D scalar vector ([`V3`])constructors.
///
/// 3D scalar vectors can be created from either three sole scalars, a single 2D scalar vector with a single scalar or
/// a single 3D scalar vector (identity function).
///
/// The `A` type variable represents the arguments type. In the case of several arguments, tuples are used.
///
/// You are advised to use the [`vec3!`](vec3) macro instead as the interface of this function is not really
/// user-friendly.
pub trait Vec3<A> {
  /// Make a [`V3`] from `A`.
  fn vec3(args: A) -> Self;
}

impl<T> Vec3<(Expr<V2<T>>, Expr<T>)> for Expr<V3<T>> {
  fn vec3(args: (Expr<V2<T>>, Expr<T>)) -> Self {
    let (xy, z) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec3,
      vec![xy.erased, z.erased],
    ))
  }
}

impl<T> Vec3<(Expr<T>, Expr<T>, Expr<T>)> for Expr<V3<T>> {
  fn vec3(args: (Expr<T>, Expr<T>, Expr<T>)) -> Self {
    let (x, y, z) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec3,
      vec![x.erased, y.erased, z.erased],
    ))
  }
}

/// Trait allowing to create 4D scalar vector ([`V4`])constructors.
///
/// 4D scalar vectors can be created from either four sole scalars, a single 3D scalar vector with a single scalar,
/// two 2D scalar vectors, a 2D scalar vector and a sole scalar or a single 4D scalar vector (identity function).
///
/// The `A` type variable represents the arguments type. In the case of several arguments, tuples are used.
///
/// You are advised to use the [`vec4!`](vec4) macro instead as the interface of this function is not really
/// user-friendly.
pub trait Vec4<A> {
  /// Make a [`V4`] from `A`.
  fn vec4(args: A) -> Self;
}

impl<T> Vec4<(Expr<V3<T>>, Expr<T>)> for Expr<V4<T>> {
  fn vec4(args: (Expr<V3<T>>, Expr<T>)) -> Self {
    let (xyz, w) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![xyz.erased, w.erased],
    ))
  }
}

impl<T> Vec4<(Expr<V2<T>>, Expr<V2<T>>)> for Expr<V4<T>> {
  fn vec4(args: (Expr<V2<T>>, Expr<V2<T>>)) -> Self {
    let (xy, zw) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![xy.erased, zw.erased],
    ))
  }
}

impl<'a, T> Vec4<(Expr<V2<T>>, Expr<T>, Expr<T>)> for Expr<V4<T>> {
  fn vec4(args: (Expr<V2<T>>, Expr<T>, Expr<T>)) -> Self {
    let (xy, z, w) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![xy.erased, z.erased, w.erased],
    ))
  }
}

impl<'a, T> Vec4<(Expr<T>, Expr<T>, Expr<T>, Expr<T>)> for Expr<V4<T>> {
  fn vec4(args: (Expr<T>, Expr<T>, Expr<T>, Expr<T>)) -> Self {
    let (x, y, z, w) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![x.erased, y.erased, z.erased, w.erased],
    ))
  }
}
