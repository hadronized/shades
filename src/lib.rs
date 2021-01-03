use std::{marker::PhantomData, ops};

#[derive(Debug)]
pub struct V;

#[derive(Debug)]
pub struct TC;

#[derive(Debug)]
pub struct TE;

#[derive(Debug)]
pub struct G;

#[derive(Debug)]
pub struct F;

#[derive(Debug)]
pub struct L;

pub unsafe trait CompatibleStage<S> {
  type Intersect;
}

unsafe impl<T> CompatibleStage<L> for T {
  type Intersect = T;
}

macro_rules! impl_CompatibleStage {
  ($t:ty) => {
    unsafe impl CompatibleStage<$t> for $t {
      type Intersect = $t;
    }
  };
}

impl_CompatibleStage!(V);
impl_CompatibleStage!(TC);
impl_CompatibleStage!(TE);
impl_CompatibleStage!(G);
impl_CompatibleStage!(F);

#[derive(Debug)]
pub struct Shader<S> {
  decls: Vec<ShaderDecl>,
  _phantom: PhantomData<S>,
}

impl<S> Shader<S> {
  pub fn new() -> Self {
    Self {
      decls: Vec::new(),
      _phantom: PhantomData,
    }
  }

  pub fn fun<F, R, A>(&mut self, f: F) -> FunHandle<S, R, A>
  where
    F: ToFun<S, R, A>,
  {
    let fundef = f.build_fn();
    let handle = self.decls.len();

    self.decls.push(ShaderDecl::FunDef(fundef.erased));

    FunHandle {
      erased: ErasedFunHandle::UserDefined(handle as _),
      _phantom: PhantomData,
    }
  }

  pub fn constant<T>(&mut self, expr: Expr<S, T>) -> Var<S, T>
  where
    T: ToType,
  {
    let n = self.decls.len() as u16;
    self.decls.push(ShaderDecl::Const(expr.erased));

    Var(Expr::new(ErasedExpr::Var(ScopedHandle::global(n))))
  }

  pub fn input<T>(&mut self) -> Var<S, T>
  where
    T: ToType,
  {
    let n = self.decls.len() as u16;
    self.decls.push(ShaderDecl::In(T::TYPE, n));

    Var(Expr::new(ErasedExpr::Var(ScopedHandle::global(n))))
  }

  pub fn output<T>(&mut self) -> Var<S, T>
  where
    T: ToType,
  {
    let n = self.decls.len() as u16;
    self.decls.push(ShaderDecl::Out(T::TYPE, n));

    Var(Expr::new(ErasedExpr::Var(ScopedHandle::global(n))))
  }
}

#[derive(Debug)]
enum ShaderDecl {
  FunDef(ErasedFun),
  Const(ErasedExpr),
  In(Type, u16),
  Out(Type, u16),
}

macro_rules! make_vn {
  ($t:ident, $dim:expr) => {
    #[derive(Clone, Debug, PartialEq)]
    pub struct $t<T>([T; $dim]);

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
}

#[derive(Debug)]
pub struct Expr<S, T>
where
  T: ?Sized,
{
  erased: ErasedExpr,
  _phantom: PhantomData<(S, T)>,
}

impl<S, T> From<&'_ Self> for Expr<S, T> {
  fn from(e: &Self) -> Self {
    Self::new(e.erased.clone())
  }
}

impl<S, T> Clone for Expr<S, T> {
  fn clone(&self) -> Self {
    Self::new(self.erased.clone())
  }
}

impl<S, T> Expr<S, T>
where
  T: ?Sized,
{
  const fn new(erased: ErasedExpr) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }

  pub fn eq<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Eq(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  pub fn neq<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Neq(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

impl<S, T> Expr<S, T>
where
  T: PartialOrd,
{
  pub fn lt<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Lt(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  pub fn lte<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Lte(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  pub fn gt<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Gt(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  pub fn gte<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Gte(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

impl<S> Expr<S, bool> {
  pub fn and<Q>(&self, rhs: impl Into<Expr<Q, bool>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::And(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  pub fn or<Q>(&self, rhs: impl Into<Expr<Q, bool>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Or(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  pub fn xor<Q>(&self, rhs: impl Into<Expr<Q, bool>>) -> Expr<S::Intersect, bool>
  where
    S: CompatibleStage<Q>,
  {
    Expr::new(ErasedExpr::Xor(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

trait Bounded<S, T>: Sized {
  fn min<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, T>
  where
    S: CompatibleStage<Q>;
  fn max<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, T>
  where
    S: CompatibleStage<Q>;

  fn clamp<Q, R>(
    &self,
    min_value: impl Into<Expr<R, T>>,
    max_value: impl Into<Expr<Q, T>>,
  ) -> Expr<<S::Intersect as CompatibleStage<R>>::Intersect, T>
  where
    S: CompatibleStage<Q>,
    S::Intersect: CompatibleStage<R>,
    Expr<S::Intersect, T>: Bounded<S::Intersect, T>,
  {
    self.min(max_value).max(min_value)
  }
}

macro_rules! impl_Bounded {
  ($t:ty) => {
    impl<S, T> Bounded<S, T> for Expr<S, $t> {
      fn min<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, T>
      where
        S: CompatibleStage<Q>,
      {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Min,
          vec![self.erased.clone(), rhs.into().erased],
        ))
      }

      fn max<Q>(&self, rhs: impl Into<Expr<Q, T>>) -> Expr<S::Intersect, T>
      where
        S: CompatibleStage<Q>,
      {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Max,
          vec![self.erased.clone(), rhs.into().erased],
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

// not
macro_rules! impl_Not_Expr {
  ($t:ty) => {
    impl<S> ops::Not for Expr<S, $t> {
      type Output = Self;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.erased)))
      }
    }

    impl<'a, S> ops::Not for &'a Expr<S, $t> {
      type Output = Expr<S, $t>;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.erased.clone())))
      }
    }
  };
}

impl_Not_Expr!(bool);
impl_Not_Expr!(V2<bool>);
impl_Not_Expr!(V3<bool>);
impl_Not_Expr!(V4<bool>);

// neg
macro_rules! impl_Neg_Expr {
  ($t:ty) => {
    impl<S> ops::Neg for Expr<S, $t> {
      type Output = Self;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.erased)))
      }
    }

    impl<'a, S> ops::Neg for &'a Expr<S, $t> {
      type Output = Expr<S, $t>;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.erased.clone())))
      }
    }
  };
}

impl_Neg_Expr!(i32);
impl_Neg_Expr!(V2<i32>);
impl_Neg_Expr!(V3<i32>);
impl_Neg_Expr!(V4<i32>);

impl_Neg_Expr!(u32);
impl_Neg_Expr!(V2<u32>);
impl_Neg_Expr!(V3<u32>);
impl_Neg_Expr!(V4<u32>);

impl_Neg_Expr!(f32);
impl_Neg_Expr!(V2<f32>);
impl_Neg_Expr!(V3<f32>);
impl_Neg_Expr!(V4<f32>);

// binary arithmetic and logical (+, -, *, /)
// binop
macro_rules! impl_binop_Expr {
  ($op:ident, $meth_name:ident, $a:ty, $b:ty) => {
    // expr OP expr
    impl<'a, S, Q> ops::$op<Expr<Q, $b>> for Expr<S, $a>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $a>;

      fn $meth_name(self, rhs: Expr<Q, $b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a, S, Q> ops::$op<&'a Expr<Q, $b>> for Expr<S, $a>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $a>;

      fn $meth_name(self, rhs: &'a Expr<Q, $b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    impl<'a, S, Q> ops::$op<Expr<Q, $b>> for &'a Expr<S, $a>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $a>;

      fn $meth_name(self, rhs: Expr<Q, $b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    impl<'a, S, Q> ops::$op<&'a Expr<Q, $b>> for &'a Expr<S, $a>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $a>;

      fn $meth_name(self, rhs: &'a Expr<Q, $b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // expr OP t, where t is automatically lifted
    impl<'a, S> ops::$op<$b> for Expr<S, $a> {
      type Output = Expr<S, $a>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs: Expr<_, _> = rhs.into();
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a, S> ops::$op<$b> for &'a Expr<S, $a> {
      type Output = Expr<S, $a>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs: Expr<L, $b> = rhs.into();
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
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

macro_rules! impl_binshift_Expr {
  ($op:ident, $meth_name:ident, $ty:ty) => {
    // expr OP expr
    impl<S, Q> ops::$op<Expr<Q, u32>> for Expr<S, $ty>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $ty>;

      fn $meth_name(self, rhs: Expr<Q, u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a, S, Q> ops::$op<Expr<Q, u32>> for &'a Expr<S, $ty>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $ty>;

      fn $meth_name(self, rhs: Expr<Q, u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    impl<'a, S, Q> ops::$op<&'a Expr<Q, u32>> for Expr<S, $ty>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $ty>;

      fn $meth_name(self, rhs: &'a Expr<Q, u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    impl<'a, S, Q> ops::$op<&'a Expr<Q, u32>> for &'a Expr<S, $ty>
    where
      S: CompatibleStage<Q>,
    {
      type Output = Expr<S::Intersect, $ty>;

      fn $meth_name(self, rhs: &'a Expr<Q, u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // expr OP bits
    impl<S> ops::$op<u32> for Expr<S, $ty> {
      type Output = Self;

      fn $meth_name(self, rhs: u32) -> Self::Output {
        let rhs: Expr<_, _> = rhs.into();
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a, S> ops::$op<u32> for &'a Expr<S, $ty> {
      type Output = Expr<S, $ty>;

      fn $meth_name(self, rhs: u32) -> Self::Output {
        let rhs: Expr<_, _> = rhs.into();
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
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
    impl From<$t> for Expr<L, $t> {
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
    impl From<$t> for Expr<L, $t> {
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

/// Easily create literal expressions.
///
/// TODO
#[macro_export]
macro_rules! lit {
  ($e:expr) => {
    Expr::<L, _>::from($e)
  };

  ($a:expr, $b:expr) => {
    Expr::<L, _>::from(V2::from([$a, $b]))
  };

  ($a:expr, $b:expr, $c:expr) => {
    Expr::<L, _>::from(V3::from([$a, $b, $c]))
  };

  ($a:expr, $b:expr, $c:expr, $d:expr) => {
    Expr::<L, _>::from(V4::from([$a, $b, $c, $d]))
  };
}

#[derive(Clone, Debug, PartialEq)]
pub struct Return<S> {
  erased: ErasedReturn,
  _phantom: PhantomData<S>,
}

impl<S> Return<S> {
  fn new(erased: ErasedReturn) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
enum ErasedReturn {
  Void,
  Expr(ErasedExpr),
}

impl<S> From<()> for Return<S> {
  fn from(_: ()) -> Self {
    Self::new(ErasedReturn::Void)
  }
}

impl<S, Q, T> From<Expr<Q, T>> for Return<S>
where
  S: CompatibleStage<Q, Intersect = S>,
{
  fn from(expr: Expr<Q, T>) -> Self {
    Self::new(ErasedReturn::Expr(expr.erased))
  }
}

pub trait ToFun<S, R, A> {
  fn build_fn(self) -> FunDef<S, R, A>;
}

impl<S, F, R> ToFun<S, R, ()> for F
where
  Self: Fn(&mut Scope<S, R>) -> R,
  Return<S>: From<R>,
{
  fn build_fn(self) -> FunDef<S, R, ()> {
    let mut scope = Scope::new(0);
    let ret = self(&mut scope);

    let erased = ErasedFun::new(Vec::new(), scope.erased, Return::<S>::from(ret).erased);

    FunDef::new(erased)
  }
}

macro_rules! impl_ToFun_args {
  ($($arg:ident , $arg_ident:ident , $arg_rank:expr),*) => {
    impl<S, F, R, $($arg),*> ToFun<S, R, ($(Expr<S, $arg>),*)> for F
      where
          Self: Fn(&mut Scope<S, R>, $(Expr<S, $arg>),*) -> R,
          Return<S>: From<R>,
          $($arg: ToType),*
          {
            fn build_fn(self) -> FunDef<S, R, ($(Expr<S, $arg>),*)> {
              $( let $arg_ident = Expr::new(ErasedExpr::Var(ScopedHandle::fun_arg($arg_rank))); )*
              let args = vec![$( $arg::TYPE ),*];

              let mut scope = Scope::new(0);
              let ret = self(&mut scope, $($arg_ident),*);

              let erased = ErasedFun::new(args, scope.erased, Return::<S>::from(ret).erased);

              FunDef::new(erased)
            }
          }
  }
}

impl<S, F, R, A> ToFun<S, R, Expr<S, A>> for F
where
  Self: Fn(&mut Scope<S, R>, Expr<S, A>) -> R,
  Return<S>: From<R>,
  A: ToType,
{
  fn build_fn(self) -> FunDef<S, R, Expr<S, A>> {
    let arg = Expr::new(ErasedExpr::Var(ScopedHandle::fun_arg(0)));

    let mut scope = Scope::new(0);
    let ret = self(&mut scope, arg);

    let erased = ErasedFun::new(vec![A::TYPE], scope.erased, Return::<S>::from(ret).erased);

    FunDef::new(erased)
  }
}

impl_ToFun_args!(A0, a0, 0, A1, a1, 1);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8, 8
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13, A14, a14, 14
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13, A14, a14, 14, A15, a15, 15
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13, A14, a14, 14, A15, a15, 15,
  A16, a16, 16
);

#[derive(Clone, Debug, PartialEq)]
pub struct FunHandle<S, R, A> {
  erased: ErasedFunHandle,
  _phantom: PhantomData<(S, R, A)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ErasedFunHandle {
  Min,
  Max,
  UserDefined(u16),
}

#[derive(Clone, Debug)]
pub struct FunDef<S, R, A> {
  erased: ErasedFun,
  _phantom: PhantomData<(S, R, A)>,
}

impl<S, R, A> FunDef<S, R, A> {
  fn new(erased: ErasedFun) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

#[derive(Clone, Debug)]
pub struct ErasedFun {
  args: Vec<Type>,
  scope: ErasedScope,
  ret: ErasedReturn,
}

impl ErasedFun {
  fn new(args: Vec<Type>, scope: ErasedScope, ret: ErasedReturn) -> Self {
    Self { args, scope, ret }
  }
}

#[derive(Clone, Debug)]
pub struct Scope<S, R> {
  erased: ErasedScope,
  _phantom: PhantomData<(S, R)>,
}

impl<S, R> Scope<S, R>
where
  Return<S>: From<R>,
{
  fn new(id: u16) -> Self {
    Self {
      erased: ErasedScope::new(id),
      _phantom: PhantomData,
    }
  }

  fn deeper<Q>(&self) -> Scope<S, Q>
  where
    Return<S>: From<Q>,
  {
    Scope::new(self.erased.id + 1)
  }

  pub fn var<Q, T>(&mut self, init_value: impl Into<Expr<Q, T>>) -> Var<S, T>
  where
    S: CompatibleStage<Q>,
    T: ToType,
  {
    let n = self.erased.next_var;
    let handle = ScopedHandle::fun_var(self.erased.id, n);

    self.erased.next_var += 1;

    self.erased.instructions.push(ScopeInstr::VarDecl {
      ty: T::TYPE,
      handle,
      init_value: init_value.into().erased,
    });

    Var(Expr::new(ErasedExpr::Var(handle)))
  }

  pub fn leave(&mut self, ret: impl Into<R>) {
    self
      .erased
      .instructions
      .push(ScopeInstr::Return(Return::<S>::from(ret.into()).erased));
  }

  pub fn abort(&mut self) {
    self
      .erased
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Void));
  }

  pub fn when<'a, Q>(
    &'a mut self,
    condition: impl Into<Expr<Q, bool>>,
    body: impl Fn(&mut Scope<S, R>),
  ) -> When<'a, S, R>
  where
    S: CompatibleStage<Q>,
  {
    let mut scope = self.deeper();
    body(&mut scope);

    self.erased.instructions.push(ScopeInstr::If {
      condition: condition.into().erased,
      scope: scope.erased,
    });

    When { parent_scope: self }
  }

  pub fn unless<'a, Q>(
    &'a mut self,
    condition: impl Into<Expr<Q, bool>>,
    body: impl Fn(&mut Scope<S, R>),
  ) -> When<'a, S, R>
  where
    S: CompatibleStage<Q>,
  {
    self.when(!condition.into(), body)
  }

  pub fn loop_for<Q, T>(
    &mut self,
    init_value: impl Into<Expr<Q, T>>,
    condition: impl Fn(&Expr<S, T>) -> Expr<S, bool>,
    iter_fold: impl Fn(&Expr<S, T>) -> Expr<S, T>,
    body: impl Fn(&mut Scope<S, R>, &Expr<S, T>),
  ) where
    S: CompatibleStage<Q>,
    T: ToType,
  {
    let mut scope = self.deeper();

    // bind the init value so that it’s available in all closures
    let Var(init_expr) = scope.var(init_value);

    let condition = condition(&init_expr);

    // generate the “post expr”, which is basically the free from of the third part of the for loop; people usually
    // set this to ++i, i++, etc., but in our case, the expression is to treat as a fold’s accumulator
    let post_expr = iter_fold(&init_expr);

    body(&mut scope, &init_expr);

    self.erased.instructions.push(ScopeInstr::For {
      init_expr: init_expr.erased,
      condition: condition.erased,
      post_expr: post_expr.erased,
      scope: scope.erased,
    });
  }

  pub fn loop_while<Q>(
    &mut self,
    condition: impl Into<Expr<Q, bool>>,
    body: impl Fn(&mut Scope<S, R>),
  ) where
    S: CompatibleStage<Q>,
  {
    let mut scope = self.deeper();
    body(&mut scope);

    self.erased.instructions.push(ScopeInstr::While {
      condition: condition.into().erased,
      scope: scope.erased,
    });
  }

  pub fn loop_continue(&mut self) {
    self.erased.instructions.push(ScopeInstr::Continue);
  }

  pub fn loop_break(&mut self) {
    self.erased.instructions.push(ScopeInstr::Break);
  }
}

#[derive(Clone, Debug, PartialEq)]
struct ErasedScope {
  id: u16,
  instructions: Vec<ScopeInstr>,
  next_var: u16,
}

impl ErasedScope {
  fn new(id: u16) -> Self {
    Self {
      id,
      instructions: Vec::new(),
      next_var: 0,
    }
  }
}

pub struct When<'a, S, R> {
  /// The scope from which this [`When`] expression comes from.
  ///
  /// This will be handy if we want to chain this when with others (corresponding to `else if` and `else`, for
  /// instance).
  parent_scope: &'a mut Scope<S, R>,
}

impl<S, R> When<'_, S, R>
where
  Return<S>: From<R>,
{
  pub fn or_else<Q>(
    self,
    condition: impl Into<Expr<Q, bool>>,
    body: impl Fn(&mut Scope<S, R>),
  ) -> Self
  where
    S: CompatibleStage<Q>,
  {
    let mut scope = self.parent_scope.deeper();
    body(&mut scope);

    self
      .parent_scope
      .erased
      .instructions
      .push(ScopeInstr::ElseIf {
        condition: condition.into().erased,
        scope: scope.erased,
      });

    self
  }

  pub fn or(self, body: impl Fn(&mut Scope<S, R>)) {
    let mut scope = self.parent_scope.deeper();
    body(&mut scope);

    self
      .parent_scope
      .erased
      .instructions
      .push(ScopeInstr::Else {
        scope: scope.erased,
      });
  }
}

#[derive(Debug)]
pub struct Var<S, T>(pub Expr<S, T>);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ScopedHandle {
  BuiltIn(BuiltIn),
  Global(u16),
  FunArg(u16),
  FunVar { subscope: u16, handle: u16 },
}

impl ScopedHandle {
  const fn built_in(b: BuiltIn) -> Self {
    Self::BuiltIn(b)
  }

  const fn global(handle: u16) -> Self {
    Self::Global(handle)
  }

  const fn fun_arg(handle: u16) -> Self {
    Self::FunArg(handle)
  }

  const fn fun_var(subscope: u16, handle: u16) -> Self {
    Self::FunVar { subscope, handle }
  }
}

#[derive(Clone, Debug, PartialEq)]
enum ScopeInstr {
  VarDecl {
    ty: Type,
    handle: ScopedHandle,
    init_value: ErasedExpr,
  },

  Return(ErasedReturn),

  Continue,

  Break,

  If {
    condition: ErasedExpr,
    scope: ErasedScope,
  },

  ElseIf {
    condition: ErasedExpr,
    scope: ErasedScope,
  },

  Else {
    scope: ErasedScope,
  },

  For {
    init_expr: ErasedExpr,
    condition: ErasedExpr,
    post_expr: ErasedExpr,
    scope: ErasedScope,
  },

  While {
    condition: ErasedExpr,
    scope: ErasedScope,
  },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ArraySpec {
  SizedArray(u16),
  UnsizedArray,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Dim {
  Scalar,
  D2,
  D3,
  D4,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Type {
  prim_ty: PrimType,
  array_spec: Option<ArraySpec>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PrimType {
  Int(Dim),
  UInt(Dim),
  Float(Dim),
  Bool(Dim),
}

pub trait ToType {
  const TYPE: Type;
}

macro_rules! impl_ToType {
  ($t:ty, $q:ident, $d:ident) => {
    impl ToType for $t {
      const TYPE: Type = Type {
        prim_ty: PrimType::$q(Dim::$d),
        array_spec: None,
      };
    }
  };
}

impl_ToType!(i32, Int, Scalar);
impl_ToType!(u32, UInt, Scalar);
impl_ToType!(f32, Float, Scalar);
impl_ToType!(bool, Bool, Scalar);
impl_ToType!(V2<i32>, Int, D2);
impl_ToType!(V2<u32>, UInt, D2);
impl_ToType!(V2<f32>, Float, D2);
impl_ToType!(V2<bool>, Bool, D2);
impl_ToType!(V3<i32>, Int, D3);
impl_ToType!(V3<u32>, UInt, D3);
impl_ToType!(V3<f32>, Float, D3);
impl_ToType!(V3<bool>, Bool, D3);
impl_ToType!(V4<i32>, Int, D4);
impl_ToType!(V4<u32>, UInt, D4);
impl_ToType!(V4<f32>, Float, D4);
impl_ToType!(V4<bool>, Bool, D4);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwizzleSelector {
  X,
  Y,
  Z,
  W,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Swizzle {
  D1(SwizzleSelector),
  D2(SwizzleSelector, SwizzleSelector),
  D3(SwizzleSelector, SwizzleSelector, SwizzleSelector),
  D4(
    SwizzleSelector,
    SwizzleSelector,
    SwizzleSelector,
    SwizzleSelector,
  ),
}

pub trait Swizzlable<S> {
  fn swizzle(&self, sw: S) -> Self;
}

// 2D
impl<S, T> Swizzlable<SwizzleSelector> for Expr<S, V2<T>> {
  fn swizzle(&self, x: SwizzleSelector) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<S, T> Swizzlable<[SwizzleSelector; 2]> for Expr<S, V2<T>> {
  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

// 3D
impl<S, T> Swizzlable<SwizzleSelector> for Expr<S, V3<T>> {
  fn swizzle(&self, x: SwizzleSelector) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<S, T> Swizzlable<[SwizzleSelector; 2]> for Expr<S, V3<T>> {
  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

impl<S, T> Swizzlable<[SwizzleSelector; 3]> for Expr<S, V3<T>> {
  fn swizzle(&self, [x, y, z]: [SwizzleSelector; 3]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D3(x, y, z),
    ))
  }
}

// 4D
impl<S, T> Swizzlable<SwizzleSelector> for Expr<S, V4<T>> {
  fn swizzle(&self, x: SwizzleSelector) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<S, T> Swizzlable<[SwizzleSelector; 2]> for Expr<S, V4<T>> {
  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

impl<S, T> Swizzlable<[SwizzleSelector; 3]> for Expr<S, V4<T>> {
  fn swizzle(&self, [x, y, z]: [SwizzleSelector; 3]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D3(x, y, z),
    ))
  }
}

impl<S, T> Swizzlable<[SwizzleSelector; 4]> for Expr<S, V4<T>> {
  fn swizzle(&self, [x, y, z, w]: [SwizzleSelector; 4]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D4(x, y, z, w),
    ))
  }
}

#[macro_export]
macro_rules! sw {
  ($e:expr, . $a:tt) => {
    $e.swizzle(sw_extract!($a))
  };

  ($e:expr, . $a:tt . $b:tt) => {
    $e.swizzle([sw_extract!($a), sw_extract!($b)])
  };

  ($e:expr, . $a:tt . $b:tt . $c:tt) => {
    $e.swizzle([sw_extract!($a), sw_extract!($b), sw_extract!($c)])
  };

  ($e:expr, . $a:tt . $b:tt . $c:tt . $d:tt) => {
    $e.swizzle([
      sw_extract!($a),
      sw_extract!($b),
      sw_extract!($c),
      sw_extract!($d),
    ])
  };
}

#[macro_export]
macro_rules! sw_extract {
  (x) => {
    SwizzleSelector::X
  };

  (r) => {
    SwizzleSelector::X
  };

  (y) => {
    SwizzleSelector::Y
  };

  (g) => {
    SwizzleSelector::Y
  };

  (z) => {
    SwizzleSelector::Z
  };

  (b) => {
    SwizzleSelector::Z
  };

  (w) => {
    SwizzleSelector::Z
  };

  (a) => {
    SwizzleSelector::Z
  };
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BuiltIn {
  VertexID,
}

// vertex shader built-ins
pub const VERTEX_ID: Expr<V, i32> =
  Expr::new(ErasedExpr::Var(ScopedHandle::built_in(BuiltIn::VertexID)));

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn expr_lit() {
    assert_eq!(lit!(true).erased, ErasedExpr::LitBool(true));
    assert_eq!(lit![1, 2].erased, ErasedExpr::LitInt2([1, 2]));
  }

  #[test]
  fn expr_unary() {
    let mut scope = Scope::<(), ()>::new(0);

    let a = !lit!(true);
    let b = -lit!(3i32);
    let Var(c) = scope.var(17);

    assert_eq!(
      a.erased,
      ErasedExpr::Not(Box::new(ErasedExpr::LitBool(true)))
    );
    assert_eq!(b.erased, ErasedExpr::Neg(Box::new(ErasedExpr::LitInt(3))));
    assert_eq!(c.erased, ErasedExpr::Var(ScopedHandle::fun_var(0, 0)));
  }

  #[test]
  fn expr_binary() {
    let a = lit!(1i32) + lit!(2);
    let b = lit!(1i32) + 2;

    assert_eq!(a.erased, b.erased);
    assert_eq!(
      a.erased,
      ErasedExpr::Add(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );
    assert_eq!(
      b.erased,
      ErasedExpr::Add(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );

    let a = lit!(1i32) - lit!(2);
    let b = lit!(1i32) - 2;

    assert_eq!(a.erased, b.erased);
    assert_eq!(
      a.erased,
      ErasedExpr::Sub(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );
    assert_eq!(
      b.erased,
      ErasedExpr::Sub(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );

    let a = lit!(1i32) * lit!(2);
    let b = lit!(1i32) * 2;

    assert_eq!(a.erased, b.erased);
    assert_eq!(
      a.erased,
      ErasedExpr::Mul(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );
    assert_eq!(
      b.erased,
      ErasedExpr::Mul(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );

    let a = lit!(1i32) / lit!(2);
    let b = lit!(1i32) / 2;

    assert_eq!(a.erased, b.erased);
    assert_eq!(
      a.erased,
      ErasedExpr::Div(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );
    assert_eq!(
      b.erased,
      ErasedExpr::Div(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      )
    );
  }

  #[test]
  fn expr_ref_inference() {
    let a = lit!(1i32);
    let b = a.clone() + 1;
    let c = a + 1;

    assert_eq!(b.erased, c.erased);
  }

  #[test]
  fn expr_var() {
    let mut scope = Scope::<(), ()>::new(0);

    let Var(x) = scope.var(0);
    let Var(y) = scope.var(1u32);
    let Var(z) = scope.var(lit![false, true, false]);

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
          array_spec: None
        },
        handle: ScopedHandle::fun_var(0, 0),
        init_value: ErasedExpr::LitInt(0)
      }
    );
    assert_eq!(
      scope.erased.instructions[1],
      ScopeInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::UInt(Dim::Scalar),
          array_spec: None
        },
        handle: ScopedHandle::fun_var(0, 1),
        init_value: ErasedExpr::LitUInt(1)
      }
    );
    assert_eq!(
      scope.erased.instructions[2],
      ScopeInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Bool(Dim::D3),
          array_spec: None
        },
        handle: ScopedHandle::fun_var(0, 2),
        init_value: ErasedExpr::LitBool3([false, true, false])
      }
    );
  }

  #[test]
  fn min_max_clamp() {
    let a = lit!(1i32);
    let b = lit!(2);
    let c = lit!(3);

    assert_eq!(
      a.min(&b).erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Min,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)]
      )
    );

    assert_eq!(
      a.max(&b).erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Max,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)]
      )
    );

    assert_eq!(
      a.clamp(b, c).erased,
      ErasedExpr::FunCall(
        ErasedFunHandle::Max,
        vec![
          ErasedExpr::FunCall(
            ErasedFunHandle::Min,
            vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(3)]
          ),
          ErasedExpr::LitInt(2),
        ]
      )
    );
  }

  #[test]
  fn fun0() {
    let mut shader = Shader::new();
    let fun = shader.fun(|s: &mut Scope<(), ()>| {
      let _x = s.var(3);
    });

    assert_eq!(fun.erased, ErasedFunHandle::UserDefined(0));

    match shader.decls[0] {
      ShaderDecl::FunDef(ref fun) => {
        assert_eq!(fun.ret, ErasedReturn::Void);
        assert_eq!(fun.args, vec![]);
        assert_eq!(fun.scope.instructions.len(), 1);
        assert_eq!(
          fun.scope.instructions[0],
          ScopeInstr::VarDecl {
            ty: Type {
              prim_ty: PrimType::Int(Dim::Scalar),
              array_spec: None
            },
            handle: ScopedHandle::fun_var(0, 0),
            init_value: ErasedExpr::LitInt(3)
          }
        )
      }
      _ => panic!("wrong type"),
    }
  }

  #[test]
  fn fun1() {
    let mut shader = Shader::new();
    let fun = shader.fun(|f: &mut Scope<V, Expr<V, i32>>, _arg: Expr<V, i32>| {
      let Var(x) = f.var(lit!(3i32));
      x
    });

    assert_eq!(fun.erased, ErasedFunHandle::UserDefined(0));

    match shader.decls[0] {
      ShaderDecl::FunDef(ref fun) => {
        assert_eq!(
          fun.ret,
          ErasedReturn::Expr(ErasedExpr::Var(ScopedHandle::fun_var(0, 0)))
        );
        assert_eq!(
          fun.args,
          vec![Type {
            prim_ty: PrimType::Int(Dim::Scalar),
            array_spec: None
          }]
        );
        assert_eq!(fun.scope.instructions.len(), 1);
        assert_eq!(
          fun.scope.instructions[0],
          ScopeInstr::VarDecl {
            ty: Type {
              prim_ty: PrimType::Int(Dim::Scalar),
              array_spec: None
            },
            handle: ScopedHandle::fun_var(0, 0),
            init_value: ErasedExpr::LitInt(3)
          }
        )
      }
      _ => panic!("wrong type"),
    }
  }

  #[test]
  fn swizzling() {
    let mut scope = Scope::<(), ()>::new(0);
    let Var(foo) = scope.var(lit![1, 2]);
    let foo_xy = sw!(foo, .x.y);
    let foo_xx = sw!(foo, .x.x);

    assert_eq!(
      foo_xy.erased,
      ErasedExpr::Swizzle(
        Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
        Swizzle::D2(SwizzleSelector::X, SwizzleSelector::Y)
      )
    );

    assert_eq!(
      foo_xx.erased,
      ErasedExpr::Swizzle(
        Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
        Swizzle::D2(SwizzleSelector::X, SwizzleSelector::X)
      )
    );
  }

  #[test]
  fn when() {
    let mut s = Scope::<L, Expr<L, V4<f32>>>::new(0);

    let Var(x) = s.var(1);
    s.when(x.eq(lit!(2)), |s| {
      let Var(y) = s.var(lit![1., 2., 3., 4.]);
      s.leave(y);
    })
    .or_else(x.eq(lit!(0)), |s| s.leave(lit![0., 0., 0., 0.]))
    .or(|_| ());

    assert_eq!(s.erased.instructions.len(), 4);

    assert_eq!(
      s.erased.instructions[0],
      ScopeInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Int(Dim::Scalar),
          array_spec: None,
        },
        handle: ScopedHandle::fun_var(0, 0),
        init_value: ErasedExpr::LitInt(1)
      }
    );

    // if
    let mut scope = ErasedScope::new(1);
    scope.next_var = 1;
    scope.instructions.push(ScopeInstr::VarDecl {
      ty: Type {
        prim_ty: PrimType::Float(Dim::D4),
        array_spec: None,
      },
      handle: ScopedHandle::fun_var(1, 0),
      init_value: ErasedExpr::LitFloat4([1., 2., 3., 4.]),
    });
    scope
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Expr(ErasedExpr::Var(
        ScopedHandle::fun_var(1, 0),
      ))));

    assert_eq!(
      s.erased.instructions[1],
      ScopeInstr::If {
        condition: ErasedExpr::Eq(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
          Box::new(ErasedExpr::LitInt(2))
        ),
        scope,
      }
    );

    // else if
    let mut scope = ErasedScope::new(1);
    scope
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Expr(
        ErasedExpr::LitFloat4([0., 0., 0., 0.]),
      )));

    assert_eq!(
      s.erased.instructions[2],
      ScopeInstr::ElseIf {
        condition: ErasedExpr::Eq(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
          Box::new(ErasedExpr::LitInt(0))
        ),
        scope,
      }
    );

    // else
    assert_eq!(
      s.erased.instructions[3],
      ScopeInstr::Else {
        scope: ErasedScope::new(1)
      }
    );
  }

  #[test]
  fn for_loop() {
    let mut scope: Scope<L, Expr<L, i32>> = Scope::new(0);

    scope.loop_for(
      0,
      |a| a.lt(lit!(10)),
      |a| a + 1,
      |s, a| {
        s.leave(a);
      },
    );

    assert_eq!(scope.erased.instructions.len(), 1);

    let mut loop_scope = ErasedScope::new(1);
    loop_scope.next_var = 1;
    loop_scope.instructions.push(ScopeInstr::VarDecl {
      ty: Type {
        prim_ty: PrimType::Int(Dim::Scalar),
        array_spec: None,
      },
      handle: ScopedHandle::fun_var(1, 0),
      init_value: ErasedExpr::LitInt(0),
    });
    loop_scope
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Expr(ErasedExpr::Var(
        ScopedHandle::fun_var(1, 0),
      ))));

    assert_eq!(
      scope.erased.instructions[0],
      ScopeInstr::For {
        init_expr: ErasedExpr::Var(ScopedHandle::fun_var(1, 0)),
        condition: ErasedExpr::Lt(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(1, 0))),
          Box::new(ErasedExpr::LitInt(10))
        ),
        post_expr: ErasedExpr::Add(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(1, 0))),
          Box::new(ErasedExpr::LitInt(1))
        ),
        scope: loop_scope
      }
    );
  }

  #[test]
  fn while_loop() {
    let mut scope: Scope<L, Expr<L, i32>> = Scope::new(0);

    scope.loop_while(lit!(1).lt(lit!(2)), Scope::loop_continue);

    let mut loop_scope = ErasedScope::new(1);
    loop_scope.instructions.push(ScopeInstr::Continue);

    assert_eq!(scope.erased.instructions.len(), 1);
    assert_eq!(
      scope.erased.instructions[0],
      ScopeInstr::While {
        condition: ErasedExpr::Lt(
          Box::new(ErasedExpr::LitInt(1)),
          Box::new(ErasedExpr::LitInt(2))
        ),
        scope: loop_scope
      }
    );
  }

  #[test]
  fn vertex_id() {
    let x = lit!(1);
    let _ = VERTEX_ID + x;
  }
}
