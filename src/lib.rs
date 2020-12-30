use std::{marker::PhantomData, ops};

#[derive(Debug)]
pub struct Shader {
  decls: Vec<ShaderDecl>,
}

impl Shader {
  pub fn new() -> Self {
    Self { decls: Vec::new() }
  }

  pub fn fun<F, R, A>(&mut self, f: F) -> FunHandle<R, A>
  where
    F: ToFun<R, A>,
  {
    let fundef = f.build_fn();
    let handle = self.decls.len();

    self.decls.push(ShaderDecl::FunDef(fundef.erased));

    FunHandle {
      erased: ErasedFunHandle::UserDefined(handle as _),
      _phantom: PhantomData,
    }
  }

  pub fn constant<T>(&mut self, expr: Expr<T>) -> Var<T>
  where
    T: ToType,
  {
    let n = self.decls.len() as u16;
    self.decls.push(ShaderDecl::Const(expr.erased));

    Var(Expr::new(ErasedExpr::Var(ScopedHandle::global(n))))
  }

  pub fn input<T>(&mut self) -> Var<T>
  where
    T: ToType,
  {
    let n = self.decls.len() as u16;
    self.decls.push(ShaderDecl::In(T::TYPE, n));

    Var(Expr::new(ErasedExpr::Var(ScopedHandle::global(n))))
  }

  pub fn output<T>(&mut self) -> Var<T>
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
pub struct Expr<T> {
  erased: ErasedExpr,
  _phantom: PhantomData<T>,
}

impl<T> Clone for Expr<T> {
  fn clone(&self) -> Self {
    Self::new(self.erased.clone())
  }
}

impl<T> From<&'_ Self> for Expr<T> {
  fn from(e: &Self) -> Self {
    e.clone()
  }
}

impl<T> Expr<T> {
  fn new(erased: ErasedExpr) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

pub trait Eq<RHS> {
  fn eq(&self, rhs: RHS) -> Expr<bool>;

  fn neq(&self, rhs: RHS) -> Expr<bool>;
}

impl<T> Eq<Expr<T>> for Expr<T> {
  fn eq(&self, rhs: Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Eq(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }

  fn neq(&self, rhs: Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Neq(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }
}

impl<'a, T> Eq<&'a Expr<T>> for Expr<T> {
  fn eq(&self, rhs: &'a Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Eq(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  fn neq(&self, rhs: &'a Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Neq(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }
}

pub trait Ord<RHS>: Eq<RHS> {
  fn lt(&self, rhs: RHS) -> Expr<bool>;

  fn lte(&self, rhs: RHS) -> Expr<bool>;

  fn gt(&self, rhs: RHS) -> Expr<bool>;

  fn gte(&self, rhs: RHS) -> Expr<bool>;
}

impl<T> Ord<Expr<T>> for Expr<T>
where
  T: PartialOrd,
{
  fn lt(&self, rhs: Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lt(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }

  fn lte(&self, rhs: Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lte(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }

  fn gt(&self, rhs: Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gt(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }

  fn gte(&self, rhs: Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gte(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }
}

impl<'a, T> Ord<&'a Expr<T>> for Expr<T>
where
  T: PartialOrd,
{
  fn lt(&self, rhs: &'a Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lt(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  fn lte(&self, rhs: &'a Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lte(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  fn gt(&self, rhs: &'a Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gt(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  fn gte(&self, rhs: &'a Expr<T>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gte(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }
}

pub trait Boolean<RHS> {
  fn and(&self, rhs: RHS) -> Self;

  fn or(&self, rhs: RHS) -> Self;

  fn xor(&self, rhs: RHS) -> Self;
}

impl Boolean<Expr<bool>> for Expr<bool> {
  fn and(&self, rhs: Self) -> Self {
    Self::new(ErasedExpr::And(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }

  fn or(&self, rhs: Self) -> Self {
    Self::new(ErasedExpr::Or(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }

  fn xor(&self, rhs: Self) -> Self {
    Self::new(ErasedExpr::Xor(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased),
    ))
  }
}

impl<'a> Boolean<&'a Self> for Expr<bool> {
  fn and(&self, rhs: &'a Self) -> Self {
    Self::new(ErasedExpr::And(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  fn or(&self, rhs: &'a Self) -> Self {
    Self::new(ErasedExpr::Or(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  fn xor(&self, rhs: &'a Self) -> Self {
    Self::new(ErasedExpr::Xor(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }
}

trait Bounded<RHS = Self>: Sized {
  type Target;

  fn min(&self, rhs: RHS) -> Self::Target;
  fn max(&self, rhs: RHS) -> Self::Target;

  fn clamp(&self, min_value: RHS, max_value: RHS) -> Self::Target;
}

macro_rules! impl_Bounded {
  ($t:ty) => {
    impl Bounded for Expr<$t> {
      type Target = Self;

      fn min(&self, rhs: Self) -> Self::Target {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Min,
          vec![self.erased.clone(), rhs.erased],
        ))
      }

      fn max(&self, rhs: Self) -> Self::Target {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Max,
          vec![self.erased.clone(), rhs.erased],
        ))
      }

      fn clamp(&self, min_value: Self, max_value: Self) -> Self::Target {
        self.min(max_value).max(min_value)
      }
    }

    impl<'a> Bounded<&'a Self> for Expr<$t> {
      type Target = Self;

      fn min(&self, rhs: &'a Self) -> Self::Target {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Min,
          vec![self.erased.clone(), rhs.erased.clone()],
        ))
      }

      fn max(&self, rhs: &'a Self) -> Self::Target {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Max,
          vec![self.erased.clone(), rhs.erased.clone()],
        ))
      }

      fn clamp(&self, min_value: &'a Self, max_value: &'a Self) -> Self::Target {
        self.min(max_value).max(min_value)
      }
    }
  };
}

impl_Bounded!(i32);
impl_Bounded!([i32; 2]);
impl_Bounded!([i32; 3]);
impl_Bounded!([i32; 4]);

impl_Bounded!(u32);
impl_Bounded!([u32; 2]);
impl_Bounded!([u32; 3]);
impl_Bounded!([u32; 4]);

impl_Bounded!(f32);
impl_Bounded!([f32; 2]);
impl_Bounded!([f32; 3]);
impl_Bounded!([f32; 4]);

impl_Bounded!(bool);
impl_Bounded!([bool; 2]);
impl_Bounded!([bool; 3]);
impl_Bounded!([bool; 4]);

// not
macro_rules! impl_Not_Expr {
  ($t:ty) => {
    impl ops::Not for Expr<$t> {
      type Output = Self;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.erased)))
      }
    }

    impl<'a> ops::Not for &'a Expr<$t> {
      type Output = Expr<$t>;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.erased.clone())))
      }
    }
  };
}

impl_Not_Expr!(bool);
impl_Not_Expr!([bool; 2]);
impl_Not_Expr!([bool; 3]);
impl_Not_Expr!([bool; 4]);

// neg
macro_rules! impl_Neg_Expr {
  ($t:ty) => {
    impl ops::Neg for Expr<$t> {
      type Output = Self;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.erased)))
      }
    }

    impl<'a> ops::Neg for &'a Expr<$t> {
      type Output = Expr<$t>;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.erased.clone())))
      }
    }
  };
}

impl_Neg_Expr!(i32);
impl_Neg_Expr!([i32; 2]);
impl_Neg_Expr!([i32; 3]);
impl_Neg_Expr!([i32; 4]);

impl_Neg_Expr!(u32);
impl_Neg_Expr!([u32; 2]);
impl_Neg_Expr!([u32; 3]);
impl_Neg_Expr!([u32; 4]);

impl_Neg_Expr!(f32);
impl_Neg_Expr!([f32; 2]);
impl_Neg_Expr!([f32; 3]);
impl_Neg_Expr!([f32; 4]);

// binary arithmetic and logical (+, -, *, /)
// binop
macro_rules! impl_binop_Expr {
  ($op:ident, $meth_name:ident, $a:ty, $b:ty) => {
    // expr OP expr
    impl<'a> ops::$op<Expr<$b>> for Expr<$a> {
      type Output = Expr<$a>;

      fn $meth_name(self, rhs: Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a> ops::$op<&'a Expr<$b>> for Expr<$a> {
      type Output = Expr<$a>;

      fn $meth_name(self, rhs: &'a Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    impl<'a> ops::$op<Expr<$b>> for &'a Expr<$a> {
      type Output = Expr<$a>;

      fn $meth_name(self, rhs: Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    impl<'a> ops::$op<&'a Expr<$b>> for &'a Expr<$a> {
      type Output = Expr<$a>;

      fn $meth_name(self, rhs: &'a Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // expr OP t, where t is automatically lifted
    impl<'a> ops::$op<$b> for Expr<$a> {
      type Output = Expr<$a>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs = Expr::from(rhs);
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a> ops::$op<$b> for &'a Expr<$a> {
      type Output = Expr<$a>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs = Expr::from(rhs);
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
impl_binop_Expr!(BitOr, bitor, [bool; 2], [bool; 2]);
impl_binop_Expr!(BitOr, bitor, [bool; 2], bool);
impl_binop_Expr!(BitOr, bitor, [bool; 3], [bool; 3]);
impl_binop_Expr!(BitOr, bitor, [bool; 3], bool);
impl_binop_Expr!(BitOr, bitor, [bool; 4], [bool; 4]);
impl_binop_Expr!(BitOr, bitor, [bool; 4], bool);

// and
impl_binop_Expr!(BitAnd, bitand, bool, bool);
impl_binop_Expr!(BitAnd, bitand, [bool; 2], [bool; 2]);
impl_binop_Expr!(BitAnd, bitand, [bool; 2], bool);
impl_binop_Expr!(BitAnd, bitand, [bool; 3], [bool; 3]);
impl_binop_Expr!(BitAnd, bitand, [bool; 3], bool);
impl_binop_Expr!(BitAnd, bitand, [bool; 4], [bool; 4]);
impl_binop_Expr!(BitAnd, bitand, [bool; 4], bool);

// xor
impl_binop_Expr!(BitXor, bitxor, bool, bool);
impl_binop_Expr!(BitXor, bitxor, [bool; 2], [bool; 2]);
impl_binop_Expr!(BitXor, bitxor, [bool; 2], bool);
impl_binop_Expr!(BitXor, bitxor, [bool; 3], [bool; 3]);
impl_binop_Expr!(BitXor, bitxor, [bool; 3], bool);
impl_binop_Expr!(BitXor, bitxor, [bool; 4], [bool; 4]);
impl_binop_Expr!(BitXor, bitxor, [bool; 4], bool);

/// Run a macro on all supported types to generate the impl for them
///
/// The macro has to have to take two `ty` as argument and yield a `std::ops` trait implementor.
macro_rules! impl_binarith_Expr {
  ($op:ident, $meth_name:ident) => {
    impl_binop_Expr!($op, $meth_name, i32, i32);
    impl_binop_Expr!($op, $meth_name, [i32; 2], [i32; 2]);
    impl_binop_Expr!($op, $meth_name, [i32; 2], i32);
    impl_binop_Expr!($op, $meth_name, [i32; 3], [i32; 3]);
    impl_binop_Expr!($op, $meth_name, [i32; 3], i32);
    impl_binop_Expr!($op, $meth_name, [i32; 4], [i32; 4]);
    impl_binop_Expr!($op, $meth_name, [i32; 4], i32);

    impl_binop_Expr!($op, $meth_name, u32, u32);
    impl_binop_Expr!($op, $meth_name, [u32; 2], [u32; 2]);
    impl_binop_Expr!($op, $meth_name, [u32; 2], u32);
    impl_binop_Expr!($op, $meth_name, [u32; 3], [u32; 3]);
    impl_binop_Expr!($op, $meth_name, [u32; 3], u32);
    impl_binop_Expr!($op, $meth_name, [u32; 4], [u32; 4]);
    impl_binop_Expr!($op, $meth_name, [u32; 4], u32);

    impl_binop_Expr!($op, $meth_name, f32, f32);
    impl_binop_Expr!($op, $meth_name, [f32; 2], [f32; 2]);
    impl_binop_Expr!($op, $meth_name, [f32; 2], f32);
    impl_binop_Expr!($op, $meth_name, [f32; 3], [f32; 3]);
    impl_binop_Expr!($op, $meth_name, [f32; 3], f32);
    impl_binop_Expr!($op, $meth_name, [f32; 4], [f32; 4]);
    impl_binop_Expr!($op, $meth_name, [f32; 4], f32);
  };
}

impl_binarith_Expr!(Add, add);
impl_binarith_Expr!(Sub, sub);
impl_binarith_Expr!(Mul, mul);
impl_binarith_Expr!(Div, div);

macro_rules! impl_binshift_Expr {
  ($op:ident, $meth_name:ident, $ty:ty) => {
    // expr OP expr
    impl ops::$op<Expr<u32>> for Expr<$ty> {
      type Output = Self;

      fn $meth_name(self, rhs: Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a> ops::$op<Expr<u32>> for &'a Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    impl<'a> ops::$op<&'a Expr<u32>> for Expr<$ty> {
      type Output = Self;

      fn $meth_name(self, rhs: &'a Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    impl<'a> ops::$op<&'a Expr<u32>> for &'a Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: &'a Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // expr OP bits
    impl ops::$op<u32> for Expr<$ty> {
      type Output = Self;

      fn $meth_name(self, rhs: u32) -> Self::Output {
        let rhs = Expr::from(rhs);
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a> ops::$op<u32> for &'a Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: u32) -> Self::Output {
        let rhs = Expr::from(rhs);
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
    impl_binshift_Expr!($op, $meth_name, [i32; 2]);
    impl_binshift_Expr!($op, $meth_name, [i32; 3]);
    impl_binshift_Expr!($op, $meth_name, [i32; 4]);

    impl_binshift_Expr!($op, $meth_name, u32);
    impl_binshift_Expr!($op, $meth_name, [u32; 2]);
    impl_binshift_Expr!($op, $meth_name, [u32; 3]);
    impl_binshift_Expr!($op, $meth_name, [u32; 4]);

    impl_binshift_Expr!($op, $meth_name, f32);
    impl_binshift_Expr!($op, $meth_name, [f32; 2]);
    impl_binshift_Expr!($op, $meth_name, [f32; 3]);
    impl_binshift_Expr!($op, $meth_name, [f32; 4]);
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

macro_rules! impl_From_Expr_array {
  ([$t:ty; $dim:expr], $q:ident) => {
    impl From<[$t; $dim]> for Expr<[$t; $dim]> {
      fn from(a: [$t; $dim]) -> Self {
        Self::new(ErasedExpr::$q(a))
      }
    }
  };
}

impl_From_Expr_array!([i32; 2], LitInt2);
impl_From_Expr_array!([u32; 2], LitUInt2);
impl_From_Expr_array!([f32; 2], LitFloat2);
impl_From_Expr_array!([bool; 2], LitBool2);
impl_From_Expr_array!([i32; 3], LitInt3);
impl_From_Expr_array!([u32; 3], LitUInt3);
impl_From_Expr_array!([f32; 3], LitFloat3);
impl_From_Expr_array!([bool; 3], LitBool3);
impl_From_Expr_array!([i32; 4], LitInt4);
impl_From_Expr_array!([u32; 4], LitUInt4);
impl_From_Expr_array!([f32; 4], LitFloat4);
impl_From_Expr_array!([bool; 4], LitBool4);

/// Easily create literal expressions.
///
/// TODO
#[macro_export]
macro_rules! lit {
  ($e:expr) => {
    Expr::from($e)
  };

  ($a:expr, $b:expr) => {
    Expr::from([$a, $b])
  };

  ($a:expr, $b:expr, $c:expr) => {
    Expr::from([$a, $b, $c])
  };

  ($a:expr, $b:expr, $c:expr, $d:expr) => {
    Expr::from([$a, $b, $c, $d])
  };
}

#[derive(Clone, Debug, PartialEq)]
pub enum Return {
  Void,
  Expr(ErasedExpr),
}

impl From<()> for Return {
  fn from(_: ()) -> Self {
    Return::Void
  }
}

impl<T> From<Expr<T>> for Return {
  fn from(expr: Expr<T>) -> Self {
    Self::Expr(expr.erased)
  }
}

pub trait ToFun<R, A> {
  fn build_fn(self) -> FunDef<R, A>;
}

impl<F, R> ToFun<R, ()> for F
where
  Self: Fn(&mut Scope<R>) -> R,
  Return: From<R>,
{
  fn build_fn(self) -> FunDef<R, ()> {
    let mut scope = Scope::new(0);
    let ret = self(&mut scope);

    let erased = ErasedFun::new(Vec::new(), scope.erased, ret.into());

    FunDef::new(erased)
  }
}

macro_rules! impl_ToFun_args {
  ($($arg:ident , $arg_ident:ident , $arg_rank:expr),*) => {
    impl<F, R, $($arg),*> ToFun<R, ($(Expr<$arg>),*)> for F
      where
          Self: Fn(&mut Scope<R>, $(Expr<$arg>),*) -> R,
          Return: From<R>,
          $($arg: ToType),*
          {
            fn build_fn(self) -> FunDef<R, ($(Expr<$arg>),*)> {
              $( let $arg_ident = Expr::new(ErasedExpr::Var(ScopedHandle::fun_arg($arg_rank))); )*
              let args = vec![$( $arg::TYPE ),*];

              let mut scope = Scope::new(0);
              let ret = self(&mut scope, $($arg_ident),*);

              let erased = ErasedFun::new(args, scope.erased, ret.into());

              FunDef::new(erased)
            }
          }
  }
}

impl<F, R, A> ToFun<R, Expr<A>> for F
where
  Self: Fn(&mut Scope<R>, Expr<A>) -> R,
  Return: From<R>,
  A: ToType,
{
  fn build_fn(self) -> FunDef<R, Expr<A>> {
    let arg = Expr::new(ErasedExpr::Var(ScopedHandle::fun_arg(0)));

    let mut scope = Scope::new(0);
    let ret = self(&mut scope, arg);

    let erased = ErasedFun::new(vec![A::TYPE], scope.erased, ret.into());

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
pub struct FunHandle<R, A> {
  erased: ErasedFunHandle,
  _phantom: PhantomData<(R, A)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ErasedFunHandle {
  Min,
  Max,
  UserDefined(u16),
}

#[derive(Debug)]
pub struct FunExpr<R, A> {
  handle: ErasedFunHandle,
  _phantom: PhantomData<(R, A)>,
}

#[derive(Clone, Debug)]
pub struct FunDef<R, A> {
  erased: ErasedFun,
  _phantom: PhantomData<(R, A)>,
}

impl<R, A> FunDef<R, A> {
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
  ret: Return,
}

impl ErasedFun {
  fn new(args: Vec<Type>, scope: ErasedScope, ret: Return) -> Self {
    Self { args, scope, ret }
  }
}

#[derive(Clone, Debug)]
pub struct Scope<R> {
  erased: ErasedScope,
  _phantom: PhantomData<R>,
}

impl<R> Scope<R>
where
  Return: From<R>,
{
  fn new(id: u16) -> Self {
    Self {
      erased: ErasedScope::new(id),
      _phantom: PhantomData,
    }
  }

  fn deeper<Q>(&self) -> Scope<Q>
  where
    Return: From<Q>,
  {
    Scope::new(self.erased.id + 1)
  }

  pub fn var<T>(&mut self, init_value: impl Into<Expr<T>>) -> Var<T>
  where
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
      .push(ScopeInstr::Return(ret.into().into()));
  }

  pub fn abort(&mut self) {
    self
      .erased
      .instructions
      .push(ScopeInstr::Return(Return::Void));
  }

  pub fn when<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Scope<R>),
  ) -> When<'a, R> {
    let mut scope = self.deeper();
    body(&mut scope);

    self.erased.instructions.push(ScopeInstr::If {
      condition: condition.into().erased,
      scope: scope.erased,
    });

    When { parent_scope: self }
  }

  pub fn unless<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Scope<R>),
  ) -> When<'a, R> {
    self.when(!condition.into(), body)
  }

  pub fn loop_for<T>(
    &mut self,
    init_value: impl Into<Expr<T>>,
    condition: impl FnOnce(&Expr<T>) -> Expr<bool>,
    iter_fold: impl FnOnce(&Expr<T>) -> Expr<T>,
    body: impl FnOnce(&mut Scope<R>, &Expr<T>),
  ) where
    T: ToType,
  {
    let mut scope = self.deeper();

    // bind the init value so that it’s available in all closures
    let Var(init_expr) = scope.var(init_value);

    let condition = condition(&init_expr);

    // generate the “post expr”, which is basically the free from of the third part of the for loop; people usually
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

  pub fn loop_while(&mut self, condition: impl Into<Expr<bool>>, body: impl FnOnce(&mut Scope<R>)) {
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

pub struct When<'a, R> {
  /// The scope from which this [`When`] expression comes from.
  ///
  /// This will be handy if we want to chain this when with others (corresponding to `else if` and `else`, for
  /// instance).
  parent_scope: &'a mut Scope<R>,
}

impl<R> When<'_, R>
where
  Return: From<R>,
{
  pub fn or_else(self, condition: impl Into<Expr<bool>>, body: impl FnOnce(&mut Scope<R>)) -> Self {
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

  pub fn or(self, body: impl FnOnce(&mut Scope<R>)) {
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
pub struct Var<T>(pub Expr<T>);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ScopedHandle {
  Global(u16),
  FunArg(u16),
  FunVar { subscope: u16, handle: u16 },
}

impl ScopedHandle {
  fn global(handle: u16) -> Self {
    Self::Global(handle)
  }

  fn fun_arg(handle: u16) -> Self {
    Self::FunArg(handle)
  }

  fn fun_var(subscope: u16, handle: u16) -> Self {
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

  Return(Return),

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
impl_ToType!([i32; 2], Int, D2);
impl_ToType!([u32; 2], UInt, D2);
impl_ToType!([f32; 2], Float, D2);
impl_ToType!([bool; 2], Bool, D2);
impl_ToType!([i32; 3], Int, D3);
impl_ToType!([u32; 3], UInt, D3);
impl_ToType!([f32; 3], Float, D3);
impl_ToType!([bool; 3], Bool, D3);
impl_ToType!([i32; 4], Int, D4);
impl_ToType!([u32; 4], UInt, D4);
impl_ToType!([f32; 4], Float, D4);
impl_ToType!([bool; 4], Bool, D4);

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
impl<T> Swizzlable<SwizzleSelector> for Expr<[T; 2]> {
  fn swizzle(&self, x: SwizzleSelector) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 2]> for Expr<[T; 2]> {
  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

// 3D
impl<T> Swizzlable<SwizzleSelector> for Expr<[T; 3]> {
  fn swizzle(&self, x: SwizzleSelector) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 2]> for Expr<[T; 3]> {
  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 3]> for Expr<[T; 3]> {
  fn swizzle(&self, [x, y, z]: [SwizzleSelector; 3]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D3(x, y, z),
    ))
  }
}

// 4D
impl<T> Swizzlable<SwizzleSelector> for Expr<[T; 4]> {
  fn swizzle(&self, x: SwizzleSelector) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D1(x),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 2]> for Expr<[T; 4]> {
  fn swizzle(&self, [x, y]: [SwizzleSelector; 2]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D2(x, y),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 3]> for Expr<[T; 4]> {
  fn swizzle(&self, [x, y, z]: [SwizzleSelector; 3]) -> Self {
    Expr::new(ErasedExpr::Swizzle(
      Box::new(self.erased.clone()),
      Swizzle::D3(x, y, z),
    ))
  }
}

impl<T> Swizzlable<[SwizzleSelector; 4]> for Expr<[T; 4]> {
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
    let mut scope = Scope::<()>::new(0);

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
    let mut scope = Scope::<()>::new(0);

    let Var(x) = scope.var(0);
    let Var(y) = scope.var(1u32);
    let Var(z) = scope.var([false, true, false]);

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
    let fun = shader.fun(|s: &mut Scope<()>| {
      let _x = s.var(3);
    });

    assert_eq!(fun.erased, ErasedFunHandle::UserDefined(0));

    match shader.decls[0] {
      ShaderDecl::FunDef(ref fun) => {
        assert_eq!(fun.ret, Return::Void);
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
    let fun = shader.fun(|f: &mut Scope<Expr<i32>>, _arg: Expr<i32>| {
      let Var(x) = f.var(3i32);
      x
    });

    assert_eq!(fun.erased, ErasedFunHandle::UserDefined(0));

    match shader.decls[0] {
      ShaderDecl::FunDef(ref fun) => {
        assert_eq!(fun.ret, Return::Void);
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
    let mut scope = Scope::<()>::new(0);
    let Var(foo) = scope.var([1, 2]);
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
    let mut s = Scope::<Expr<[f32; 4]>>::new(0);

    let Var(x) = s.var(1);
    s.when(x.eq(lit!(2)), |s| {
      let Var(y) = s.var(lit![1., 2., 3., 4.]);
      s.leave(y);
    })
    .or_else(x.eq(lit!(0)), |s| s.leave(lit!([0., 0., 0., 0.])))
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
      .push(ScopeInstr::Return(Return::Expr(ErasedExpr::Var(
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
      .push(ScopeInstr::Return(Return::Expr(ErasedExpr::LitFloat4([
        0., 0., 0., 0.,
      ]))));

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
    let mut scope: Scope<Expr<i32>> = Scope::new(0);

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
      .push(ScopeInstr::Return(Return::Expr(ErasedExpr::Var(
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
    let mut scope: Scope<Expr<i32>> = Scope::new(0);

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
}
