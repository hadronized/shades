use crate::{
  builtin::BuiltIn,
  erased::Erased,
  expr::{ErasedExpr, Expr},
  fun::{ErasedReturn, Return},
  types::{ToType, Type},
  var::Var,
};
use std::{
  marker::PhantomData,
  ops::{Deref, DerefMut},
};

/// Lexical scope that must output an `R`.
///
/// Scopes are the only way to add control flow expressions to shaders. [`Scope<R>`] is the most general one, parent
/// of all scopes. Depending on the kind of control flow, several kind of scopes are possible:
///
/// - [`Scope<R>`] is the most general one and every scopes share its features.
/// - [`EscapeScope<R>`] is a special kind of [`Scope<R>`] that allows escaping from anywhere in the scope.
/// - [`LoopScope<R>`] is a special kind of [`EscapeScope<R>`] that also allows to escape local looping expressions,
///   such as `for` and `while` loops.
///
/// A [`Scope<R>`] allows to perform a bunch of actions:
///
/// - Creating variable via [`Scope::var`]. Expressions of type [`Expr<T>`] where [`T: ToType`](ToType) are bound in a
///   [`Scope<R>`] via [`Scope::var`] and a [`Var<T>`] is returned, representing the bound variable.
/// - Variable mutation via [`Scope::set`]. Any [`Var<T>`] declared previously and still reachable in the current [`Scope`]
///   can be mutated.
/// - Introducing conditional statements with [`Scope::when`] and [`Scope::unless`].
/// - Introducing looping statements with [`Scope::loop_for`] and [`Scope::loop_while`].
#[derive(Debug)]
pub struct Scope<R> {
  pub(crate) erased: ErasedScope,
  _phantom: PhantomData<R>,
}

impl<R> Scope<R>
where
  Return: From<R>,
{
  /// Create a new [`Scope<R>`] for which the ID is explicitly passed.
  ///
  /// The ID is unique in the scope hierarchy, but is not necessarily unique in the parent scope. What it means is that
  /// creating a scope `s` in a (parent) scope of ID `p` will give `s` the ID `p + 1`. So any scope created directly
  /// under the scope of ID `p` will get the `p + 1` ID. The reason for this is that variables go out of scope at the
  /// end of the scope they were created in, so it’s safe to reuse the same ID for sibling scopes, as they can’t share
  /// variables.
  pub fn new(id: u16) -> Self {
    Self {
      erased: ErasedScope::new(id),
      _phantom: PhantomData,
    }
  }

  /// Bind an expression to a variable in the current scope.
  ///
  /// `let v = s.var(e);` binds the `e` expression to `v` in the `s` [`Scope<T>`], and `e` must have type [`Expr<T>`]
  /// and `v` must be a [`Var<T>`], with [`T: ToType`](ToType).
  ///
  /// # Return
  ///
  /// The resulting [`Var<T>`] contains the representation of the binding in the EDSL and the actual binding is
  /// recorded in the current scope.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// let v = s.var(3.1415); // assign the literal 3.1415 to v
  /// let q = s.var(v * 2.); // assign v * 2. to q
  /// #   })
  /// # });
  /// ```
  pub fn var<T>(&mut self, init_value: impl Into<Expr<T>>) -> Var<T>
  where
    T: ToType,
  {
    let n = self.erased.next_var;
    let handle = ScopedHandle::fun_var(self.erased.id, n);

    self.erased.next_var += 1;

    self.erased.instructions.push(ScopeInstr::VarDecl {
      ty: T::ty(),
      handle: handle.clone(),
      init_value: init_value.into().erased,
    });

    Var::new(handle)
  }

  /// For looping statement — `for`.
  ///
  /// `s.loop_for(i, |i| /* cond */, |i| /* fold */, |i| /* body */ )` inserts a looping statement into the EDSL
  /// representing a typical “for” loop. `i` is an [`Expr<T>`] satisfying [`T: ToType`](ToType) and is used as
  /// _initial_ value.
  ///
  /// In all the following closures, `i` refers to the initial value.
  ///
  /// The first `cond` closure must return an [`Expr<bool>`], representing the condition that is held until the loop
  /// exits. The second `fold` closure is a pure computation that must return an [`Expr<T>`] and that will be evaluated
  /// at the end of each iteration before the next check on `cond`. The last and third `body` closure is the body of the
  /// loop.
  ///
  /// The behaviors of the first two closures is important to understand. Those are akin to _filtering_ and _folding_.
  /// The closure returning the [`Expr<bool>`] is given the [`Expr<T>`] at each iteration and the second closure creates
  /// the new [`Expr<T>`] for the next iteration. Normally, people are used to write this pattern as `i++`, for
  /// instance, but in the case of our EDSL, it is more akin go `i + 1`, and this value is affected to a local
  /// accumulator hidden from the user.
  ///
  /// The [`LoopScope<R>`] argument to the `body` closure is a specialization of [`Scope<R>`] that allows breaking out
  /// of loops.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{CanEscape as _, LoopScope, Scope, StageBuilder};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.loop_for(0, |i| i.lt(10), |i| i + 1, |s: &mut LoopScope<()>, i| {
  ///       s.when(i.eq(5), |s: &mut LoopScope<()>| {
  ///         // when i == 5, abort from the main function
  ///         s.abort();
  ///       });
  ///     });
  ///   })
  /// });
  /// ```
  pub fn loop_for<T>(
    &mut self,
    init_value: impl Into<Expr<T>>,
    condition: impl FnOnce(&Expr<T>) -> Expr<bool>,
    iter_fold: impl FnOnce(&Expr<T>) -> Expr<T>,
    body: impl FnOnce(&mut LoopScope<R>, &Expr<T>),
  ) where
    T: ToType,
  {
    let mut scope = LoopScope::new(self.deeper());

    // bind the init value so that it’s available in all closures
    let init_var = scope.var(init_value);

    let condition = condition(&init_var);

    // generate the “post expr”, which is basically the free from of the third part of the for loop; people usually
    // set this to ++i, i++, etc., but in our case, the expression is to treat as a fold’s accumulator
    let post_expr = iter_fold(&init_var);

    body(&mut scope, &init_var);

    let scope = Scope::from(scope);
    self.erased.instructions.push(ScopeInstr::For {
      init_ty: T::ty(),
      init_handle: ScopedHandle::fun_var(scope.erased.id, 0),
      init_expr: init_var.to_expr().erased,
      condition: condition.erased,
      post_expr: post_expr.erased,
      scope: scope.erased,
    });
  }

  /// While looping statement — `while`.
  ///
  /// `s.loop_while(cond, body)` inserts a looping statement into the EDSL representing a typical “while” loop.
  ///
  /// `cond` is an [`Expr<bool>`], representing the condition that is held until the loop exits. `body` is the content
  /// the loop will execute at each iteration.
  ///
  /// The [`LoopScope<R>`] argument to the `body` closure is a specialization of [`Scope<R>`] that allows breaking out
  /// of loops.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{CanEscape as _, LoopScope, Scope, StageBuilder};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     let i = s.var(10);
  ///
  ///     s.loop_while(i.lt(10), |s| {
  ///       s.set(&i, &i + 1);
  ///     });
  ///   })
  /// });
  /// ```
  pub fn loop_while(
    &mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut LoopScope<R>),
  ) {
    let mut scope = LoopScope::new(self.deeper());
    body(&mut scope);

    self.erased.instructions.push(ScopeInstr::While {
      condition: condition.into().erased,
      scope: Scope::from(scope).erased,
    });
  }

  /// Mutate a variable in the current scope.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// let v = s.var(1); // v = 1
  /// s.set(&v, 10); // v = 10
  /// #   })
  /// # });
  /// ```
  pub fn set<T>(
    &mut self,
    var: impl Into<Var<T>>,
    bin_op: impl Into<Option<MutateBinOp>>,
    value: impl Into<Expr<T>>,
  ) {
    self.erased.instructions.push(ScopeInstr::MutateVar {
      var: var.into().to_expr().erased,
      bin_op: bin_op.into(),
      expr: value.into().erased,
    });
  }

  /// Early-return the current function with an expression.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::StageBuilder;
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{Expr, Scope};
  ///
  /// let _fun = s.fun(|s: &mut Scope<Expr<i32>>, arg: Expr<i32>| {
  ///   // if arg is less than 10, early-return with 0
  ///   s.when(arg.lt(10), |s| {
  ///     s.leave(0);
  ///   });
  ///
  ///   arg
  /// });
  ///
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// #   })
  /// # });
  /// ```
  pub fn leave(&mut self, ret: impl Into<R>) {
    self
      .erased
      .instructions
      .push(ScopeInstr::Return(Return::from(ret.into()).erased));
  }
}

impl Scope<()> {
  /// Early-abort the current function.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::StageBuilder;
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{CanEscape as _, Expr, Scope};
  ///
  /// let _fun = s.fun(|s: &mut Scope<()>, arg: Expr<i32>| {
  ///   // if arg is less than 10, early-return with 0
  ///   s.when(arg.lt(10), |s| {
  ///     s.abort();
  ///   });
  ///
  ///   // do something else…
  /// });
  ///
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// #   })
  /// # });
  /// ```
  pub fn abort(&mut self) {
    self
      .erased
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Void));
  }
}

impl<R> Erased for Scope<R> {
  type Erased = ErasedScope;

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

/// A special kind of [`Scope`] that can also break loops.
#[derive(Debug)]
pub struct LoopScope<R>(Scope<R>);

impl<R> From<LoopScope<R>> for Scope<R> {
  fn from(s: LoopScope<R>) -> Self {
    s.0
  }
}

impl<R> Deref for LoopScope<R> {
  type Target = Scope<R>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl<R> DerefMut for LoopScope<R> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

impl<R> LoopScope<R>
where
  Return: From<R>,
{
  fn new(s: Scope<R>) -> Self {
    Self(s)
  }

  /// Break the current iteration of the nearest loop and continue to the next iteration.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::CanEscape as _;
  ///
  /// s.loop_while(true, |s| {
  ///   s.loop_continue();
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn loop_continue(&mut self) {
    self.erased.instructions.push(ScopeInstr::Continue);
  }

  /// Break the nearest loop.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::CanEscape as _;
  ///
  /// s.loop_while(true, |s| {
  ///   s.loop_break();
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn loop_break(&mut self) {
    self.erased.instructions.push(ScopeInstr::Break);
  }
}

impl<R> Erased for LoopScope<R> {
  type Erased = ErasedScope;

  fn to_erased(self) -> Self::Erased {
    self.0.erased
  }

  fn erased(&self) -> &Self::Erased {
    &self.erased
  }

  fn erased_mut(&mut self) -> &mut Self::Erased {
    &mut self.erased
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ErasedScope {
  id: u16,
  pub(crate) instructions: Vec<ScopeInstr>,
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

/// Go one level deeper in the scope.
pub trait DeepScope {
  /// Create a new fresh scope under the current scope.
  fn deeper(&self) -> Self;
}

impl<R> DeepScope for Scope<R>
where
  Return: From<R>,
{
  fn deeper(&self) -> Self {
    Scope::new(self.erased.id + 1)
  }
}

impl<R> DeepScope for LoopScope<R>
where
  Return: From<R>,
{
  fn deeper(&self) -> Self {
    LoopScope(self.0.deeper())
  }
}

/// Scopes allowing to enter conditional scopes.
///
/// Conditional scopes allow to break out of a function by early-return / aborting the function.
pub trait Conditional: Sized {
  /// Conditional statement — `if`.
  ///
  /// `s.when(cond, |s: &mut EscapeScope<R>| { /* body */ })` inserts a conditional branch in the EDSL using the `cond`
  /// expression as truth and the passed closure as body to run when the represented condition is `true`. The
  /// [`EscapeScope<R>`] provides you with the possibility to escape and leave the function earlier, either by returning
  /// an expression or by aborting the function, depending on the value of `R`: `Expr<_>` allows for early-returns and
  /// `()` allows for aborting.
  ///
  /// # Return
  ///
  /// A [`When<R>`], authorizing the same escape rules with `R`. This object allows you to chain other conditional
  /// statements, commonly referred to as `else if` and `else` in common languages.
  ///
  /// Have a look at the documentation of [`When`] for further information.
  ///
  /// # Examples
  ///
  /// Early-return:
  ///
  /// ```
  /// use shades::{CanEscape as _, EscapeScope, Expr, Scope, StageBuilder, lit};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   let f = s.fun(|s: &mut Scope<Expr<i32>>| {
  ///     s.when(lit!(1).lt(3), |s: &mut EscapeScope<Expr<i32>>| {
  ///       // do something in here
  ///
  ///       // early-return with 0; only possible if the function returns Expr<i32>
  ///       s.leave(0);
  ///     });
  ///
  ///     lit!(1)
  ///   });
  ///
  ///   s.main_fun(|s: &mut Scope<()>| {
  /// # #[cfg(feature = "fun-call")]
  ///     let x = s.var(f());
  ///   })
  /// });
  /// ```
  ///
  /// Aborting a function:
  ///
  /// ```
  /// use shades::{CanEscape as _, EscapeScope, Scope, StageBuilder, lit};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.when(lit!(1).lt(3), |s: &mut EscapeScope<()>| {
  ///       // do something in here
  ///
  ///       // break the parent function by aborting; this is possible because the return type is ()
  ///       s.abort();
  ///     });
  ///   })
  /// });
  /// ```
  fn when<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Self),
  ) -> When<'a, Self>;

  /// Complement form of [`Scope::when`].
  ///
  /// This method does the same thing as [`Scope::when`] but applies the [`Not::not`](std::ops::Not::not) operator on
  /// the condition first.
  fn unless<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Self),
  ) -> When<'a, Self> {
    self.when(!condition.into(), body)
  }
}

impl<S> Conditional for S
where
  S: DeepScope + Erased<Erased = ErasedScope>,
{
  fn when<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Self),
  ) -> When<'a, Self> {
    let mut scope = self.deeper();
    body(&mut scope);

    self.erased_mut().instructions.push(ScopeInstr::If {
      condition: condition.into().erased,
      scope: scope.erased().clone(),
    });

    When { parent_scope: self }
  }
}

/// Conditional combinator.
///
/// A [`When<S, R>`] is returned from functions such as [`CanEscape::when`] or [`CanEscape::unless`] and allows to
/// continue chaining conditional statements, encoding the concept of `else if` and `else` in more traditional languages.
#[derive(Debug)]
pub struct When<'a, S> {
  /// The scope from which this [`When`] expression comes from.
  ///
  /// This will be handy if we want to chain this when with others (corresponding to `else if` and `else`, for
  /// instance).
  parent_scope: &'a mut S,
}

impl<S> When<'_, S>
where
  S: DeepScope + Erased<Erased = ErasedScope>,
{
  /// Add a conditional branch — `else if`.
  ///
  /// This method is often found chained after [`CanEscape::when`] and allows to add a new conditional if the previous
  /// conditional fails (i.e. `else if`). The behavior is the same as with [`CanEscape::when`].
  ///
  /// # Return
  ///
  /// Another [`When<R>`], allowing to add more conditional branches.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::{CanEscape as _, lit};
  ///
  /// let x = lit!(1);
  ///
  /// // you will need CanEscape in order to use when
  /// s.when(x.lt(2), |s| {
  ///   // do something if x < 2
  /// }).or_else(x.lt(10), |s| {
  ///   // do something if x < 10
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn or_else(self, condition: impl Into<Expr<bool>>, body: impl FnOnce(&mut S)) -> Self {
    let mut scope = self.parent_scope.deeper();
    body(&mut scope);

    self
      .parent_scope
      .erased_mut()
      .instructions
      .push(ScopeInstr::ElseIf {
        condition: condition.into().erased,
        scope: scope.erased().clone(),
      });

    self
  }

  /// Add a final catch-all conditional branch — `else`.
  ///
  /// This method is often found chained after [`CanEscape::when`] and allows to finish the chain of conditional
  /// branches if the previous conditional fails (i.e. `else`). The behavior is the same as with [`CanEscape::when`].
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::{CanEscape as _, lit};
  ///
  /// let x = lit!(1);
  ///
  /// // you will need CanEscape in order to use when
  /// s.when(x.lt(2), |s| {
  ///   // do something if x < 2
  /// }).or(|s| {
  ///   // do something if x >= 2
  /// });
  /// #   })
  /// # });
  /// ```
  ///
  /// Can chain and mix conditional but [`When::or`] cannot be anywhere else but the end of the chain:
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::{CanEscape as _, lit};
  ///
  /// let x = lit!(1);
  ///
  /// // you will need CanEscape in order to use when
  /// s.when(x.lt(2), |s| {
  ///   // do something if x < 2
  /// }).or_else(x.lt(5), |s| {
  ///   // do something if x < 5
  /// }).or_else(x.lt(10), |s| {
  ///   // do something if x < 10
  /// }).or(|s| {
  ///   // else, do this
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn or(self, body: impl FnOnce(&mut S)) {
    let mut scope = self.parent_scope.deeper();
    body(&mut scope);

    self
      .parent_scope
      .erased_mut()
      .instructions
      .push(ScopeInstr::Else {
        scope: scope.erased().clone(),
      });
  }
}

/// Hierarchical and namespaced handle.
///
/// Handles live in different namespaces:
///
/// - The _built-in_ namespace gathers all built-ins.
/// - The _global_ namespace gathers everything that can be declared at top-level of a shader stage — i.e. mainly
///   constants for this namespace.
/// - The _input_ namespace gathers inputs.
/// - The _output_ namespace gathers outputs.
/// - The _function argument_ namespace gives handles to function arguments, which exist only in a function body.
/// - The _function variable_ namespace gives handles to variables defined in function bodies. This namespace is
/// hierarchical: for each scope, a new namespace is created. The depth at which a namespace is located is referred to
/// as its _subscope_.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ScopedHandle {
  BuiltIn(BuiltIn),
  Global(u16),
  FunArg(u16),
  FunVar { subscope: u16, handle: u16 },
  Input(String),
  Output(String),
  Uniform(String),

  // new type-sound representation
  Input2(u16),
}

impl ScopedHandle {
  pub(crate) const fn builtin(b: BuiltIn) -> Self {
    Self::BuiltIn(b)
  }

  pub(crate) const fn global(handle: u16) -> Self {
    Self::Global(handle)
  }

  pub(crate) const fn fun_var(subscope: u16, handle: u16) -> Self {
    Self::FunVar { subscope, handle }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScopeInstr {
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
    init_ty: Type,
    init_handle: ScopedHandle,
    init_expr: ErasedExpr,
    condition: ErasedExpr,
    post_expr: ErasedExpr,
    scope: ErasedScope,
  },

  While {
    condition: ErasedExpr,
    scope: ErasedScope,
  },

  MutateVar {
    var: ErasedExpr,
    bin_op: Option<MutateBinOp>,
    expr: ErasedExpr,
  },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MutateBinOp {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Xor,
  And,
  Or,
  Shl,
  Shr,
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    lit,
    types::{Dim, PrimType, V4},
  };

  #[test]
  fn when() {
    let mut s = Scope::<Expr<V4<f32>>>::new(0);

    let x = s.var(1);
    s.when(x.eq(lit!(2)), |s| {
      let y = s.var(lit![1., 2., 3., 4.]);
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
          array_dims: Vec::new(),
        },
        handle: ScopedHandle::fun_var(0, 0),
        init_value: ErasedExpr::LitInt(1),
      }
    );

    // if
    let mut scope = ErasedScope::new(1);
    scope.next_var = 1;
    scope.instructions.push(ScopeInstr::VarDecl {
      ty: Type {
        prim_ty: PrimType::Float(Dim::D4),
        array_dims: Vec::new(),
      },
      handle: ScopedHandle::fun_var(1, 0),
      init_value: ErasedExpr::LitFloat4([1., 2., 3., 4.]),
    });
    scope
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Expr(
        V4::<f32>::ty(),
        ErasedExpr::Var(ScopedHandle::fun_var(1, 0)),
      )));

    assert_eq!(
      s.erased.instructions[1],
      ScopeInstr::If {
        condition: ErasedExpr::Eq(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
          Box::new(ErasedExpr::LitInt(2)),
        ),
        scope,
      }
    );

    // else if
    let mut scope = ErasedScope::new(1);
    scope
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Expr(
        V4::<f32>::ty(),
        ErasedExpr::LitFloat4([0., 0., 0., 0.]),
      )));

    assert_eq!(
      s.erased.instructions[2],
      ScopeInstr::ElseIf {
        condition: ErasedExpr::Eq(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(0, 0))),
          Box::new(ErasedExpr::LitInt(0)),
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
        array_dims: Vec::new(),
      },
      handle: ScopedHandle::fun_var(1, 0),
      init_value: ErasedExpr::LitInt(0),
    });
    loop_scope
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Expr(
        i32::ty(),
        ErasedExpr::Var(ScopedHandle::fun_var(1, 0)),
      )));

    assert_eq!(
      scope.erased.instructions[0],
      ScopeInstr::For {
        init_ty: i32::ty(),
        init_handle: ScopedHandle::fun_var(1, 0),
        init_expr: ErasedExpr::Var(ScopedHandle::fun_var(1, 0)),
        condition: ErasedExpr::Lt(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(1, 0))),
          Box::new(ErasedExpr::LitInt(10)),
        ),
        post_expr: ErasedExpr::Add(
          Box::new(ErasedExpr::Var(ScopedHandle::fun_var(1, 0))),
          Box::new(ErasedExpr::LitInt(1)),
        ),
        scope: loop_scope,
      }
    );
  }

  #[test]
  fn while_loop() {
    let mut scope: Scope<Expr<i32>> = Scope::new(0);

    scope.loop_while(lit!(1).lt(lit!(2)), LoopScope::loop_continue);

    let mut loop_scope = ErasedScope::new(1);
    loop_scope.instructions.push(ScopeInstr::Continue);

    assert_eq!(scope.erased.instructions.len(), 1);
    assert_eq!(
      scope.erased.instructions[0],
      ScopeInstr::While {
        condition: ErasedExpr::Lt(
          Box::new(ErasedExpr::LitInt(1)),
          Box::new(ErasedExpr::LitInt(2)),
        ),
        scope: loop_scope,
      }
    );
  }

  #[test]
  fn while_loop_if() {
    let mut scope: Scope<Expr<i32>> = Scope::new(0);

    scope.loop_while(lit!(1).lt(lit!(2)), |scope| {
      scope
        .when(lit!(1).lt(lit!(2)), |scope| scope.loop_break())
        .or(|scope| scope.loop_break());
    });
  }
}
