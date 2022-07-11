use crate::{
  expr::{ErasedExpr, Expr},
  types::{V2, V3, V4},
};

/// Select a channel to extract from into a swizzled expession.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwizzleSelector {
  /// Select the `.x` (or `.r`) channel.
  X,

  /// Select the `.y` (or `.g`) channel.
  Y,

  /// Select the `.z` (or `.b`) channel.
  Z,

  /// Select the `.w` (or `.a`) channel.
  W,
}

/// Swizzle channel selector.
///
/// This type gives the dimension of the target expression (output) and dimension of the source expression (input). The
/// [`SwizzleSelector`] also to select a specific channel in the input expression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Swizzle {
  /// Create a one-channel expression.
  D1(SwizzleSelector),

  /// Create a two-channel expression.
  D2(SwizzleSelector, SwizzleSelector),

  /// Create a three-channel expression.
  D3(SwizzleSelector, SwizzleSelector, SwizzleSelector),

  /// Create a four-channel expression.
  D4(
    SwizzleSelector,
    SwizzleSelector,
    SwizzleSelector,
    SwizzleSelector,
  ),
}

/// Interface to implement to swizzle an expression.
///
/// If you plan to use your implementor with the [`sw!`](sw) macro, `S` must be one of the following types:
///
/// - [`SwizzleSelector`]: to implement `sw!(.x)`.
/// - [[`SwizzleSelector`]; 2]: to implement `sw!(.xx)`.
/// - [[`SwizzleSelector`]; 3]: to implement `sw!(.xxx)`.
/// - [[`SwizzleSelector`]; 4]: to implement `sw!(.xxxx)`.
pub trait Swizzlable<S> {
  type Output;

  fn swizzle(&self, sw: S) -> Self::Output;
}

// 2D
impl<T> Swizzlable<SwizzleSelector> for Expr<V2<T>> {
  type Output = Expr<T>;

  fn swizzle(&self, x: SwizzleSelector) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 2]> for Expr<V2<T>> {
  type Output = Self;

  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

// 3D
impl<T> Swizzlable<SwizzleSelector> for Expr<V3<T>> {
  type Output = Expr<T>;

  fn swizzle(&self, x: SwizzleSelector) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 2]> for Expr<V3<T>> {
  type Output = Expr<V2<T>>;

  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 3]> for Expr<V3<T>> {
  type Output = Self;

  fn swizzle(&self, [x, y, z]: [SwizzleSelector; 3]) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D3(x, y, z),
    ))
  }
}

// 4D
impl<T> Swizzlable<SwizzleSelector> for Expr<V4<T>> {
  type Output = Expr<T>;

  fn swizzle(&self, x: SwizzleSelector) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 2]> for Expr<V4<T>> {
  type Output = Expr<V2<T>>;

  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 3]> for Expr<V4<T>> {
  type Output = Expr<V3<T>>;

  fn swizzle(&self, [x, y, z]: [SwizzleSelector; 3]) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D3(x, y, z),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 4]> for Expr<V4<T>> {
  type Output = Self;

  fn swizzle(&self, [x, y, z, w]: [SwizzleSelector; 4]) -> Self::Output {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D4(x, y, z, w),
    ))
  }
}

/// Expressions having a `x` or `r` coordinate.
///
/// Akin to swizzling with `.x` or `.r`, but easier.
pub trait HasX {
  type Output;

  fn x(&self) -> Self::Output;
  fn r(&self) -> Self::Output {
    self.x()
  }
}

/// Expressions having a `y` or `g` coordinate.
///
/// Akin to swizzling with `.y` or `.g`, but easier.
pub trait HasY {
  type Output;

  fn y(&self) -> Self::Output;
  fn g(&self) -> Self::Output {
    self.y()
  }
}

/// Expressions having a `z` or `b` coordinate.
///
/// Akin to swizzling with `.z` or `.b`, but easier.
pub trait HasZ {
  type Output;

  fn z(&self) -> Self::Output;
  fn b(&self) -> Self::Output {
    self.z()
  }
}

/// Expressions having a `w` or `a` coordinate.
///
/// Akin to swizzling with `.w` or `.a`, but easier.
pub trait HasW {
  type Output;

  fn w(&self) -> Self::Output;
  fn a(&self) -> Self::Output {
    self.w()
  }
}

macro_rules! impl_has_k {
  ($trait:ident, $name:ident, $selector:ident, $t:ident) => {
    impl<T> $trait for Expr<$t<T>> {
      type Output = Expr<T>;

      fn $name(&self) -> Self::Output {
        self.swizzle(SwizzleSelector::$selector)
      }
    }
  };
}

impl_has_k!(HasX, x, X, V2);
impl_has_k!(HasX, x, X, V3);
impl_has_k!(HasX, x, X, V4);

impl_has_k!(HasY, y, Y, V2);
impl_has_k!(HasY, y, Y, V3);
impl_has_k!(HasY, y, Y, V4);

impl_has_k!(HasZ, z, Z, V3);
impl_has_k!(HasZ, z, Z, V4);

impl_has_k!(HasW, w, W, V4);

/// Swizzle macro.
///
/// This macro allows to swizzle expressions to yield expressions reorganizing the vector attributes. For instance,
/// `sw!(color, .rgbr)` will take a 4D color and will output a 4D color for which the alpha channel is overridden with
/// the red channel.
///
/// The current syntax allows to extract and construct from a lot of types. Have a look at [`Swizzlable`] for a
/// comprehensive list of what you can do.
#[macro_export]
macro_rules! sw {
  ($e:expr, . $a:tt) => {
    $e.swizzle($crate::sw_extract!($a))
  };

  ($e:expr, . $a:tt . $b:tt) => {
    $e.swizzle([$crate::sw_extract!($a), $crate::sw_extract!($b)])
  };

  ($e:expr, . $a:tt . $b:tt . $c:tt) => {
    $e.swizzle([
      $crate::sw_extract!($a),
      $crate::sw_extract!($b),
      $crate::sw_extract!($c),
    ])
  };

  ($e:expr, . $a:tt . $b:tt . $c:tt . $d:tt) => {
    $e.swizzle([
      $crate::sw_extract!($a),
      $crate::sw_extract!($b),
      $crate::sw_extract!($c),
      $crate::sw_extract!($d),
    ])
  };
}

#[doc(hidden)]
#[macro_export]
macro_rules! sw_extract {
  (x) => {
    $crate::swizzle::SwizzleSelector::X
  };

  (r) => {
    $crate::swizzle::SwizzleSelector::X
  };

  (y) => {
    $crate::swizzle::SwizzleSelector::Y
  };

  (g) => {
    $crate::swizzle::SwizzleSelector::Y
  };

  (z) => {
    $crate::swizzle::SwizzleSelector::Z
  };

  (b) => {
    $crate::swizzle::SwizzleSelector::Z
  };

  (w) => {
    $crate::swizzle::SwizzleSelector::W
  };

  (a) => {
    $crate::swizzle::SwizzleSelector::W
  };
}

#[cfg(test)]
mod test {
  use crate::{
    scope::{Scope, ScopedHandle},
    vec2, vec4,
  };

  use super::*;

  #[test]
  fn swizzling() {
    let mut scope = Scope::<()>::new(0);
    let foo = scope.var(vec2![1, 2]);
    let foo_xy: Expr<V2<_>> = sw!(foo, .x.y);
    let foo_xx: Expr<V2<_>> = sw!(foo, .x.x);

    assert_eq!(
      foo_xy.erased,
      ErasedExpr::Swizzle(
        Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
        Swizzle::D2(SwizzleSelector::X, SwizzleSelector::Y),
      )
    );

    assert_eq!(
      foo_xx.erased,
      ErasedExpr::Swizzle(
        Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
        Swizzle::D2(SwizzleSelector::X, SwizzleSelector::X),
      )
    );
  }

  #[test]
  fn has_x_y_z_w() {
    let xyzw: Expr<V4<i32>> = vec4![1, 2, 3, 4];
    let x: Expr<i32> = sw!(xyzw, .x);
    let y: Expr<i32> = sw!(xyzw, .y);
    let z: Expr<i32> = sw!(xyzw, .z);
    let w: Expr<i32> = sw!(xyzw, .w);

    assert_eq!(xyzw.x().erased, x.erased);
    assert_eq!(xyzw.y().erased, y.erased);
    assert_eq!(xyzw.z().erased, z.erased);
    assert_eq!(xyzw.w().erased, w.erased);
  }
}
