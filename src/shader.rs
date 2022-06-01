use crate::{fun::ErasedFun, expr::ErasedExpr, types::Type};

/// Shader declaration.
///
/// This contains everything that can be declared at top-level of a shader.
#[derive(Debug)]
pub(crate) enum ShaderDecl {
  /// The `main` function declaration. The [`ErasedFun`] is a function that returns nothing and has no argument.
  Main(ErasedFun),

  /// A function definition.
  ///
  /// The [`u16`] represents the _handle_ of the function, and is unique for each shader stage. The [`ErasedFun`] is
  /// the representation of the function definition.
  FunDef(u16, ErasedFun),

  /// A constant definition.
  ///
  /// The [`u16`] represents the _handle_ of the constant, and is unique for each shader stage. The [`Type`] is the
  /// the type of the constant expression. [`ErasedExpr`] is the representation of the constant.
  Const(u16, Type, ErasedExpr),
}
