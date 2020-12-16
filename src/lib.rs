use std::{collections::HashSet, marker::PhantomData, ops};

#[derive(Debug)]
pub struct Shader;

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
  Var(String),
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
  FunCall(FunHandle, Vec<Self>),
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
          FunHandle::Min,
          vec![self.erased.clone(), rhs.erased.clone()],
        ))
      }

      fn max(&self, rhs: &Self) -> Self {
        Expr::new(ErasedExpr::FunCall(
          FunHandle::Max,
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

#[derive(Clone, Debug, PartialEq)]
pub struct ErasedArg {
  name: String,
  ty: Type,
}

#[derive(Debug, PartialEq)]
pub struct Arg<T> {
  erased: ErasedArg,
  _phantom: PhantomData<T>,
}

impl<T> Clone for Arg<T> {
  fn clone(&self) -> Self {
    Self {
      erased: self.erased.clone(),
      _phantom: PhantomData,
    }
  }
}

impl<T> Arg<T>
where
  T: ToType,
{
  fn new(name: impl Into<String>) -> Self {
    let erased = ErasedArg {
      name: name.into(),
      ty: T::TYPE,
    };

    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

pub trait ToFn<F, R, A> {
  fn build_fn(self, f: F) -> FunDef<R, A>;
}

impl<F, R> ToFn<F, R, ()> for Shader
where
  F: Fn(&mut ErasedFn) -> R,
  R: ToReturn,
{
  fn build_fn(self, f: F) -> FunDef<R, ()> {
    let ret_ty = R::RET_TYPE;
    let mut erased = ErasedFn::new("f", ret_ty, vec![]);

    f(&mut erased);

    FunDef::new(erased)
  }
}

macro_rules! impl_ToFn_args {
  ($($arg:ident / $arg_ident:ident / $arg_name:expr),*) => {
    impl<F, R, $($arg),*> ToFn<F, R, ($($arg),*)> for Shader
    where
      F: Fn(&mut ErasedFn, $(Arg<$arg>),*) -> R,
      R: ToReturn,
      $($arg: ToType),*
    {
      fn build_fn(self, f: F) -> FunDef<R, ($($arg),*)> {
        $( let $arg_ident = Arg::new($arg_name); )*
        let args = vec![$( $arg_ident.clone().erased ),*];
        let ret_ty = R::RET_TYPE;
        let mut erased = ErasedFn::new("f", ret_ty, args);

        f(&mut erased, $($arg_ident),*);

        FunDef::new(erased)
      }
    }
  }
}

macro_rules! impl_ToFn_args_rec {
  ($arg:ident / $arg_ident:ident / $arg_name:expr) => {
    // hey this one is already implemented as a special case, thank you have a good day
  };

  ($arg:ident / $arg_ident:ident / $arg_name:expr, $($r:tt)*) => {
    impl_ToFn_args!($arg / $arg_ident / $arg_name, $($r)*);
    impl_ToFn_args_rec!($($r)*);
  };
}

impl_ToFn_args_rec!(
  A0 / a / "fn_a",
  A1 / b / "fn_b",
  A2 / c / "fn_c",
  A3 / d / "fn_d",
  A4 / e / "fn_e",
  A5 / f / "fn_f",
  A6 / g / "fn_g",
  A7 / h / "fn_h",
  A8 / i / "fn_i",
  A9 / j / "fn_j",
  A10 / k / "fn_k",
  A11 / l / "fn_l",
  A12 / m / "fn_m",
  A13 / n / "fn_n",
  A14 / o / "fn_o",
  A15 / p / "fn_p"
);

impl<F, R, A> ToFn<F, R, A> for Shader
where
  F: Fn(&mut ErasedFn, Arg<A>) -> R,
  R: ToReturn,
  A: ToType,
{
  fn build_fn(self, f: F) -> FunDef<R, A> {
    let arg = Arg::new("fn_a");
    let ret_ty = R::RET_TYPE;
    let mut erased = ErasedFn::new("f", ret_ty, vec![arg.clone().erased]);

    f(&mut erased, arg);

    FunDef::new(erased)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FunHandle {
  Min,
  Max,
  UserDefined(u16),
}

#[derive(Debug)]
pub struct FunExpr<R, A> {
  handle: FunHandle,
  _phantom: PhantomData<(R, A)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunDef<R, A> {
  erased: ErasedFn,
  _phantom: PhantomData<(R, A)>,
}

impl<R, A> FunDef<R, A> {
  fn new(erased: ErasedFn) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ErasedFn {
  scope: String,
  ret_ty: RetType,
  args: Vec<ErasedArg>,
  instructions: Vec<FnInstr>,
  vars: HashSet<String>,
}

impl ErasedFn {
  pub fn new(scope: impl Into<String>, ret_ty: RetType, args: Vec<ErasedArg>) -> Self {
    Self {
      scope: scope.into(),
      ret_ty,
      args,
      instructions: Vec::new(),
      vars: HashSet::new(),
    }
  }
}

impl ErasedFn {
  pub fn var<T>(&mut self, init_value: impl Into<Expr<T>>) -> Expr<T>
  where
    T: ToType,
  {
    let name = format!("{}_v{}", self.scope, self.vars.len());
    self.vars.insert(name.clone());

    self.instructions.push(FnInstr::VarDecl {
      ty: T::TYPE,
      name: name.clone(),
      init_value: init_value.into().erased,
    });

    Expr::new(ErasedExpr::Var(name))
  }
}

#[derive(Clone, Debug, PartialEq)]
enum FnInstr {
  VarDecl {
    ty: Type,
    name: String,
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
    let mut fun = ErasedFn::new("test", RetType::Void, Vec::new());

    let a = !lit!(true);
    let b = -lit!(3);
    let c = -fun.var(17);

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
        "test_v0".to_owned()
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
    let mut fun = ErasedFn::new("test", RetType::Void, Vec::new());

    let x = fun.var(0);
    let y = fun.var(1u32);
    let z = fun.var([false, true, false]);

    assert_eq!(x, Expr::new(ErasedExpr::Var("test_v0".into())));
    assert_eq!(y, Expr::new(ErasedExpr::Var("test_v1".into())));
    assert_eq!(z, Expr::new(ErasedExpr::Var("test_v2".into())));
    assert_eq!(fun.instructions.len(), 3);
    assert_eq!(
      fun.instructions[0],
      FnInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Int(Dim::Scalar),
          array_spec: None
        },
        name: "test_v0".into(),
        init_value: ErasedExpr::LitInt(0)
      }
    );
    assert_eq!(
      fun.instructions[1],
      FnInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::UInt(Dim::Scalar),
          array_spec: None
        },
        name: "test_v1".into(),
        init_value: ErasedExpr::LitUInt(1)
      }
    );
    assert_eq!(
      fun.instructions[2],
      FnInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Bool(Dim::D3),
          array_spec: None
        },
        name: "test_v2".into(),
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
        FunHandle::Min,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)]
      ))
    );

    assert_eq!(
      a.max(&b),
      Expr::new(ErasedExpr::FunCall(
        FunHandle::Max,
        vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(2)]
      ))
    );

    assert_eq!(
      a.clamp(&b, &c),
      Expr::new(ErasedExpr::FunCall(
        FunHandle::Max,
        vec![
          ErasedExpr::FunCall(
            FunHandle::Min,
            vec![ErasedExpr::LitInt(1), ErasedExpr::LitInt(3)]
          ),
          ErasedExpr::LitInt(2),
        ]
      ))
    );
  }

  #[test]
  fn erased_fn0() {
    let shader = Shader;
    let fun = shader.build_fn(|f: &mut ErasedFn| {
      let _x = f.var(3);
    });

    let fun = fun.erased;
    assert_eq!(fun.ret_ty, RetType::Void);
    assert_eq!(fun.args, vec![]);
    assert_eq!(fun.instructions.len(), 1);
    assert_eq!(
      fun.instructions[0],
      FnInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Int(Dim::Scalar),
          array_spec: None
        },
        name: "f_v0".into(),
        init_value: ErasedExpr::LitInt(3)
      }
    )
  }

  #[test]
  fn erased_fn1() {
    let shader = Shader;
    let fun: FunDef<(), i32> = shader.build_fn(|f: &mut ErasedFn, _arg| {
      let _x = f.var(3);
    });

    let fun = fun.erased;
    assert_eq!(fun.ret_ty, RetType::Void);
    assert_eq!(
      fun.args,
      vec![ErasedArg {
        name: "fn_a".into(),
        ty: Type {
          prim_ty: PrimType::Int(Dim::Scalar),
          array_spec: None
        }
      }]
    );
    assert_eq!(fun.instructions.len(), 1);
    assert_eq!(
      fun.instructions[0],
      FnInstr::VarDecl {
        ty: Type {
          prim_ty: PrimType::Int(Dim::Scalar),
          array_spec: None
        },
        name: "f_v0".into(),
        init_value: ErasedExpr::LitInt(3)
      }
    )
  }
}
