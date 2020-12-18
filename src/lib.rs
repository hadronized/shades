use std::{collections::HashSet, marker::PhantomData, ops};

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
enum ErasedExpr {
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr<T> {
  erased: ErasedExpr,
  _phantom: PhantomData<T>,
}

impl<T> Expr<T> {
  fn new(erased: ErasedExpr) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

impl<T> Expr<T>
where
  T: PartialEq,
{
  pub fn eq(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Eq(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  pub fn neq(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Neq(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }
}

impl<T> Expr<T>
where
  T: PartialOrd,
{
  pub fn lt(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Lt(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  pub fn lte(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Lte(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  pub fn gt(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Gt(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  pub fn gte(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Gte(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }
}

impl Expr<bool> {
  pub fn and(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::And(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  pub fn or(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Or(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }

  pub fn xor(&self, rhs: &Self) -> Self {
    Self::new(ErasedExpr::Xor(
      Box::new(self.erased.clone()),
      Box::new(rhs.erased.clone()),
    ))
  }
}

trait Bounded: Sized {
  fn min(&self, rhs: &Self) -> Self;
  fn max(&self, rhs: &Self) -> Self;

  fn clamp(&self, min_value: &Self, max_value: &Self) -> Self {
    self.min(max_value).max(min_value)
  }
}

macro_rules! impl_Bounded {
  ($t:ty) => {
    impl Bounded for Expr<$t> {
      fn min(&self, rhs: &Self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Min,
          vec![self.erased.clone(), rhs.erased.clone()],
        ))
      }

      fn max(&self, rhs: &Self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          ErasedFunHandle::Max,
          vec![self.erased.clone(), rhs.erased.clone()],
        ))
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
pub enum RetType {
  Void,
  Type(Type),
}

pub trait ToReturn {
  const RET_TYPE: RetType;
}

impl ToReturn for () {
  const RET_TYPE: RetType = RetType::Void;
}

impl<T> ToReturn for Expr<T>
where
  T: ToType,
{
  const RET_TYPE: RetType = RetType::Type(T::TYPE);
}

pub trait ToFun<R, A> {
  fn build_fn(self) -> FunDef<R, A>;
}

impl<F, R> ToFun<R, ()> for F
where
  Self: Fn(&mut ErasedFun) -> R,
  R: ToReturn,
{
  fn build_fn(self) -> FunDef<R, ()> {
    let ret_ty = R::RET_TYPE;
    let mut erased = ErasedFun::new(ret_ty, vec![]);

    self(&mut erased);

    FunDef::new(erased)
  }
}

macro_rules! impl_ToFun_args {
  ($($arg:ident , $arg_ident:ident , $arg_rank:expr),*) => {
    impl<F, R, $($arg),*> ToFun<R, ($($arg),*)> for F
    where
      Self: Fn(&mut ErasedFun, $(Expr<$arg>),*) -> R,
      R: ToReturn,
      $($arg: ToType),*
    {
      fn build_fn(self) -> FunDef<R, ($($arg),*)> {
        $( let $arg_ident = Expr::new(ErasedExpr::Var(ScopedHandle::fun_arg($arg_rank))); )*
        let args = vec![$( $arg::TYPE ),*];
        let ret_ty = R::RET_TYPE;
        let mut erased = ErasedFun::new(ret_ty, args);

        self(&mut erased, $($arg_ident),*);

        FunDef::new(erased)
      }
    }
  }
}

impl<F, R, A> ToFun<R, A> for F
where
  Self: Fn(&mut ErasedFun, Expr<A>) -> R,
  R: ToReturn,
  A: ToType,
{
  fn build_fn(self) -> FunDef<R, A> {
    let arg = Expr::new(ErasedExpr::Var(ScopedHandle::fun_arg(0)));
    let ret_ty = R::RET_TYPE;
    let mut erased = ErasedFun::new(ret_ty, vec![A::TYPE]);

    self(&mut erased, arg);

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

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct ErasedFun {
  ret_ty: RetType,
  args: Vec<Type>,
  instructions: Vec<FunInstr>,
  next_var: u16,
}

impl ErasedFun {
  fn new(ret_ty: RetType, args: Vec<Type>) -> Self {
    Self {
      ret_ty,
      args,
      instructions: Vec::new(),
      next_var: 0,
    }
  }
}

impl ErasedFun {
  pub fn var<T>(&mut self, init_value: impl Into<Expr<T>>) -> Var<T>
  where
    T: ToType,
  {
    let n = self.next_var;
    let handle = ScopedHandle::fun_var(0, n);

    self.next_var += 1;

    self.instructions.push(FunInstr::VarDecl {
      ty: T::TYPE,
      handle,
      init_value: init_value.into().erased,
    });

    Var(Expr::new(ErasedExpr::Var(handle)))
  }
}

#[derive(Debug)]
pub struct Var<T>(pub Expr<T>);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum ScopedHandle {
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
enum FunInstr {
  VarDecl {
    ty: Type,
    handle: ScopedHandle,
    init_value: ErasedExpr,
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
    let mut fun = ErasedFun::new(RetType::Void, Vec::new());

    let a = !lit!(true);
    let b = -lit!(3);
    let Var(c) = fun.var(17);

    assert_eq!(
      a,
      Expr::new(ErasedExpr::Not(Box::new(ErasedExpr::LitBool(true))))
    );
    assert_eq!(
      b,
      Expr::new(ErasedExpr::Neg(Box::new(ErasedExpr::LitInt(3))))
    );
    assert_eq!(
      c,
      Expr::new(ErasedExpr::Neg(Box::new(ErasedExpr::Var(
        ScopedHandle::fun_var(0, 0)
      ))))
    );
  }

  #[test]
  fn expr_binary() {
    let a = lit!(1) + lit!(2);
    let b = lit!(1) + 2;

    assert_eq!(a, b);
    assert_eq!(
      a,
      Expr::new(ErasedExpr::Add(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );
    assert_eq!(
      b,
      Expr::new(ErasedExpr::Add(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );

    let a = lit!(1) - lit!(2);
    let b = lit!(1) - 2;

    assert_eq!(a, b);
    assert_eq!(
      a,
      Expr::new(ErasedExpr::Sub(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );
    assert_eq!(
      b,
      Expr::new(ErasedExpr::Sub(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );

    let a = lit!(1) * lit!(2);
    let b = lit!(1) * 2;

    assert_eq!(a, b);
    assert_eq!(
      a,
      Expr::new(ErasedExpr::Mul(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );
    assert_eq!(
      b,
      Expr::new(ErasedExpr::Mul(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );

    let a = lit!(1) / lit!(2);
    let b = lit!(1) / 2;

    assert_eq!(a, b);
    assert_eq!(
      a,
      Expr::new(ErasedExpr::Div(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );
    assert_eq!(
      b,
      Expr::new(ErasedExpr::Div(
        Box::new(ErasedExpr::LitInt(1)),
        Box::new(ErasedExpr::LitInt(2))
      ))
    );
  }

  #[test]
  fn expr_ref_inference() {
    let a = lit!(1);
    let b = a.clone() + 1;
    let c = a + 1;

    assert_eq!(b, c);
  }

  #[test]
  fn expr_var() {
    let mut fun = ErasedFun::new(RetType::Void, Vec::new());

    let Var(x) = fun.var(0);
    let Var(y) = fun.var(1u32);
    let Var(z) = fun.var([false, true, false]);

    assert_eq!(x, Expr::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))));
    assert_eq!(y, Expr::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 1))));
    assert_eq!(
      z,
      Expr::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 2).into()))
    );
    assert_eq!(fun.instructions.len(), 3);
    assert_eq!(
      fun.instructions[0],
      FunInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Int(Dim::Scalar),
          array_spec: None
        },
        handle: ScopedHandle::fun_var(0, 1),
        init_value: ErasedExpr::LitInt(0)
      }
    );
    assert_eq!(
      fun.instructions[1],
      FunInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::UInt(Dim::Scalar),
          array_spec: None
        },
        handle: ScopedHandle::fun_var(0, 1),
        init_value: ErasedExpr::LitUInt(1)
      }
    );
    assert_eq!(
      fun.instructions[2],
      FunInstr::VarDecl {
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
    let a = lit!(1);
    let b = lit!(2);
    let c = lit!(3);

    assert_eq!(
      a.min(&b),
      Expr::new(ErasedExpr::FunCall(
        ErasedFunHandle::Min,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)]
      ))
    );

    assert_eq!(
      a.max(&b),
      Expr::new(ErasedExpr::FunCall(
        ErasedFunHandle::Max,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)]
      ))
    );

    assert_eq!(
      a.clamp(&b, &c),
      Expr::new(ErasedExpr::FunCall(
        ErasedFunHandle::Max,
        vec![
          ErasedExpr::FunCall(
            ErasedFunHandle::Min,
            vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(3)]
          ),
          ErasedExpr::LitInt(2),
        ]
      ))
    );
  }

  #[test]
  fn erased_fn0() {
    let mut shader = Shader::new();
    let fun = shader.fun(|f: &mut ErasedFun| {
      let _x = f.var(3);
    });

    assert_eq!(fun.erased, ErasedFunHandle::UserDefined(0));

    match shader.decls[0] {
      ShaderDecl::FunDef(ref fun) => {
        assert_eq!(fun.ret_ty, RetType::Void);
        assert_eq!(fun.args, vec![]);
        assert_eq!(fun.instructions.len(), 1);
        assert_eq!(
          fun.instructions[0],
          FunInstr::VarDecl {
            ty: Type {
              prim_ty: PrimType::Int(Dim::Scalar),
              array_spec: None
            },
            handle: ScopedHandle::fun_var(0, 1),
            init_value: ErasedExpr::LitInt(3)
          }
        )
      }
      _ => panic!("wrong type"),
    }
  }

  #[test]
  fn erased_fn1() {
    let mut shader = Shader::new();
    let fun: FunHandle<(), i32> = shader.fun(|f: &mut ErasedFun, _arg| {
      let _x = f.var(3);
    });

    assert_eq!(fun.erased, ErasedFunHandle::UserDefined(0));

    match shader.decls[0] {
      ShaderDecl::FunDef(ref fun) => {
        assert_eq!(fun.ret_ty, RetType::Void);
        assert_eq!(
          fun.args,
          vec![Type {
            prim_ty: PrimType::Int(Dim::Scalar),
            array_spec: None
          }]
        );
        assert_eq!(fun.instructions.len(), 1);
        assert_eq!(
          fun.instructions[0],
          FunInstr::VarDecl {
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
  fn simple_shader() {
    let mut shader = Shader::new();
    let fun_a = shader.fun(|f: &mut ErasedFun, a, b| a + b);
  }
}
