//! GLSL writers.

use crate::{
  BuiltIn, Dim, ErasedExpr, ErasedFun, ErasedFunHandle, ErasedReturn, ErasedScope, FragmentBuiltIn,
  GeometryBuiltIn, MatrixDim, PrimType, ScopeInstr, ScopedHandle, Stage, ShaderDecl, Swizzle,
  SwizzleSelector, TessCtrlBuiltIn, TessEvalBuiltIn, Type, VertexBuiltIn,
};
use std::fmt;

// Number of space an indent level represents.
const INDENT_SPACES: usize = 2;

/// Write a [`Shader`] to a [`String`].
pub fn write_shader_to_str(shader: impl AsRef<Stage>) -> Result<String, fmt::Error> {
  let mut output = String::new();
  write_shader(&mut output, shader)?;
  Ok(output)
}

/// Write a [`Shader`] to a [`fmt::Write`](std::fmt::Write).
pub fn write_shader(f: &mut impl fmt::Write, shader: impl AsRef<Stage>) -> Result<(), fmt::Error> {
  for decl in &shader.as_ref().builder.decls {
    match decl {
      ShaderDecl::Main(fun) => write_main_fun(f, fun)?,
      ShaderDecl::FunDef(handle, fun) => write_fun_def(f, *handle, fun)?,
      ShaderDecl::Const(handle, ty, ref constant) => write_constant(f, *handle, ty, constant)?,
      ShaderDecl::In(name, ty) => write_input(f, name, ty)?,
      ShaderDecl::Out(name, ty) => write_output(f, name, ty)?,
      ShaderDecl::Uniform(name, ty) => write_uniform(f, name, ty)?,
    }
  }

  Ok(())
}

fn write_main_fun(f: &mut impl fmt::Write, fun: &ErasedFun) -> Result<(), fmt::Error> {
  f.write_str("\nvoid main() {\n")?;
  write_scope(f, &fun.scope, 1)?;
  f.write_str("}")
}

fn write_fun_def(f: &mut impl fmt::Write, handle: u16, fun: &ErasedFun) -> Result<(), fmt::Error> {
  // just for aesthetics :')
  f.write_str("\n")?;

  let ret_expr = match &fun.ret {
    ErasedReturn::Void => {
      f.write_str("void")?;
      None
    }

    ErasedReturn::Expr(ref ty, expr) => {
      write_type(f, ty)?;
      Some(expr)
    }
  };

  f.write_str(" ")?;
  write_user_fun_handle(f, handle)?;

  f.write_str("(")?;
  if !fun.args.is_empty() {
    write_type(f, &fun.args[0])?;
    f.write_str("arg_0")?;

    for (i, arg) in fun.args.iter().enumerate().skip(1) {
      f.write_str(", ")?;

      write_type(f, arg)?;
      write!(f, " arg_{}", i)?;
    }
  }
  f.write_str(") {\n")?;

  write_scope(f, &fun.scope, 1)?;

  if let Some(ref expr) = ret_expr {
    write_indent(f, 1)?;
    f.write_str("return ")?;
    write_expr(f, expr)?;
    f.write_str(";")?;
  }

  f.write_str("\n}\n")
}

fn write_scope(
  f: &mut impl fmt::Write,
  scope: &ErasedScope,
  indent_lvl: usize,
) -> Result<(), fmt::Error> {
  for instr in &scope.instructions {
    write_indent(f, indent_lvl)?;

    match instr {
      ScopeInstr::VarDecl {
        ty,
        handle,
        init_value,
      } => {
        write_type(f, ty)?;
        f.write_str(" ")?;
        write_scoped_handle(f, handle)?;
        f.write_str(" = ")?;
        write_expr(f, init_value)?;
        f.write_str(";")?;
      }

      ScopeInstr::Return(ret) => match ret {
        ErasedReturn::Void => {
          f.write_str("return;")?;
        }

        ErasedReturn::Expr(_, expr) => {
          f.write_str("return ")?;
          write_expr(f, expr)?;
          f.write_str(";")?;
        }
      },

      ScopeInstr::Continue => {
        f.write_str("continue;")?;
      }

      ScopeInstr::Break => {
        f.write_str("break;")?;
      }

      ScopeInstr::If { condition, scope } => {
        f.write_str("if (")?;
        write_expr(f, condition)?;
        f.write_str(") {\n")?;
        write_scope(f, scope, indent_lvl + 1)?;
        write_indented(f, indent_lvl, "}")?;
      }

      ScopeInstr::ElseIf { condition, scope } => {
        f.write_str(" else if (")?;
        write_expr(f, condition)?;
        f.write_str(") {\n")?;
        write_scope(f, scope, indent_lvl + 1)?;
        write_indented(f, indent_lvl, "}")?;
      }

      ScopeInstr::Else { scope } => {
        f.write_str("else {\n")?;
        write_scope(f, scope, indent_lvl + 1)?;
        write_indented(f, indent_lvl, "}")?;
      }

      ScopeInstr::For {
        init_ty,
        init_handle,
        init_expr,
        condition,
        post_expr,
        scope,
      } => {
        f.write_str("for (")?;

        // initialization
        write_type(f, init_ty)?;
        f.write_str(" ")?;
        write_scoped_handle(f, init_handle)?;
        f.write_str(" = ")?;
        write_expr(f, init_expr)?;
        f.write_str("; ")?;

        // condition
        write_expr(f, condition)?;
        f.write_str("; ")?;

        // iteration; we basically write <init-expr> = <next-expr> in a fold-like way, so we need to re-use the
        // init_handle
        write_scoped_handle(f, init_handle)?;
        f.write_str(" = ")?;
        write_expr(f, post_expr)?;
        f.write_str(") {\n")?;

        // scope
        write_scope(f, scope, indent_lvl + 1)?;
        f.write_str("}")?;
      }

      ScopeInstr::While { condition, scope } => {
        f.write_str("while (")?;
        write_expr(f, condition)?;
        f.write_str(") {\n")?;
        write_scope(f, scope, indent_lvl + 1)?;
        write_indented(f, indent_lvl, "}")?;
      }

      ScopeInstr::MutateVar { var, expr } => {
        write_expr(f, var)?;
        f.write_str(" = ")?;
        write_expr(f, expr)?;
        f.write_str(";")?;
      }
    }

    f.write_str("\n")?;
  }

  Ok(())
}

fn write_constant(
  f: &mut impl fmt::Write,
  handle: u16,
  ty: &Type,
  constant: &ErasedExpr,
) -> Result<(), fmt::Error> {
  f.write_str("const ")?;
  write_type(f, ty)?;
  f.write_str(" ")?;
  write_scoped_handle(f, &ScopedHandle::global(handle))?;
  f.write_str(" = ")?;
  write_expr(f, constant)?;
  f.write_str(";\n")
}

fn write_input(f: &mut impl fmt::Write, name: &str, ty: &Type) -> Result<(), fmt::Error> {
  f.write_str("in ")?;
  write_type(f, ty)?;
  write!(f, " {};\n", name)
}

fn write_output(f: &mut impl fmt::Write, name: &str, ty: &Type) -> Result<(), fmt::Error> {
  f.write_str("out ")?;
  write_type(f, ty)?;
  write!(f, " {};\n", name)
}

fn write_uniform(f: &mut impl fmt::Write, name: &str, ty: &Type) -> Result<(), fmt::Error> {
  f.write_str("uniform ")?;
  write_type(f, ty)?;
  write!(f, " {};\n", name)
}

fn write_expr(f: &mut impl fmt::Write, expr: &ErasedExpr) -> Result<(), fmt::Error> {
  match expr {
    ErasedExpr::LitInt(x) => write!(f, "{}", x),
    ErasedExpr::LitUInt(x) => write!(f, "{}", x),
    ErasedExpr::LitFloat(x) => write!(f, "{}", write_f32(*x)),
    ErasedExpr::LitBool(x) => write!(f, "{}", x),

    ErasedExpr::LitInt2([x, y]) => write!(f, "ivec2({}, {})", x, y),
    ErasedExpr::LitUInt2([x, y]) => write!(f, "uvec2({}, {})", x, y),
    ErasedExpr::LitFloat2([x, y]) => write!(f, "vec2({}, {})", write_f32(*x), write_f32(*y)),

    ErasedExpr::LitBool2([x, y]) => write!(f, "bvec2({}, {})", x, y),

    ErasedExpr::LitInt3([x, y, z]) => write!(f, "ivec3({}, {}, {})", x, y, z),
    ErasedExpr::LitUInt3([x, y, z]) => write!(f, "uvec3({}, {}, {})", x, y, z),
    ErasedExpr::LitFloat3([x, y, z]) => write!(
      f,
      "vec3({}, {}, {})",
      write_f32(*x),
      write_f32(*y),
      write_f32(*z)
    ),
    ErasedExpr::LitBool3([x, y, z]) => write!(f, "bvec3({}, {}, {})", x, y, z),

    ErasedExpr::LitInt4([x, y, z, w]) => write!(f, "ivec4({}, {}, {}, {})", x, y, z, w),
    ErasedExpr::LitUInt4([x, y, z, w]) => write!(f, "uvec4({}, {}, {}, {})", x, y, z, w),
    ErasedExpr::LitFloat4([x, y, z, w]) => write!(
      f,
      "vec4({}, {}, {}, {})",
      write_f32(*x),
      write_f32(*y),
      write_f32(*z),
      write_f32(*w)
    ),

    ErasedExpr::LitBool4([x, y, z, w]) => write!(f, "bvec4({}, {}, {}, {})", x, y, z, w),

    ErasedExpr::LitM22(m) => write_matrix(f, "mat2", &m.0),
    ErasedExpr::LitM33(m) => write_matrix(f, "mat3", &m.0),
    ErasedExpr::LitM44(m) => write_matrix(f, "mat4", &m.0),

    ErasedExpr::Array(ty, items) => {
      write_type(f, ty)?;
      f.write_str("(")?;

      write_expr(f, &items[0])?;
      for item in &items[1..] {
        f.write_str(",")?;
        write_expr(f, item)?;
      }

      f.write_str(")")
    }

    ErasedExpr::Var(handle) => write_var(f, handle),

    ErasedExpr::Not(e) => {
      f.write_str("!")?;
      write_expr(f, e)
    }

    ErasedExpr::And(a, b) => {
      f.write_str("()")?;
      write_expr(f, a)?;
      f.write_str("&&")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Or(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" || ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Xor(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" ^^ ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::BitAnd(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" & ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::BitOr(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" | ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::BitXor(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" ^ ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Neg(e) => {
      f.write_str("-(")?;
      write_expr(f, e)?;
      f.write_str(")")
    }

    ErasedExpr::Add(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" + ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Sub(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" - ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Mul(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" * ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Div(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" / ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Rem(a, b) => {
      f.write_str("mod(")?;
      write_expr(f, a)?;
      f.write_str(", ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Shl(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" << ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Shr(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" >> ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Eq(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" == ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Neq(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" != ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Lt(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" < ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Lte(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" <= ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Gt(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" > ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::Gte(a, b) => {
      f.write_str("(")?;
      write_expr(f, a)?;
      f.write_str(" >= ")?;
      write_expr(f, b)?;
      f.write_str(")")
    }

    ErasedExpr::FunCall(fun, args) => {
      write_fun_handle(f, fun)?;
      f.write_str("(")?;

      if !args.is_empty() {
        write_expr(f, &args[0])?;
      }

      for arg in &args[1..] {
        f.write_str(", ")?;
        write_expr(f, arg)?;
      }

      f.write_str(")")
    }

    ErasedExpr::Swizzle(e, s) => {
      write_expr(f, e)?;
      f.write_str(".")?;
      write_swizzle(f, s)
    }

    ErasedExpr::Field { object, field } => {
      write_expr(f, object)?;
      f.write_str(".")?;
      write_expr(f, field)
    }

    ErasedExpr::ArrayLookup { object, index } => {
      write_expr(f, object)?;
      f.write_str("[")?;
      write_expr(f, index)?;
      f.write_str("]")
    }
  }
}

fn write_f32(f: f32) -> String {
  if f == 0. {
    return "0.".to_owned();
  }

  let mut s = f.to_string();

  if f.trunc() == 0. {
    s[1..].to_string()
  } else if f.fract() == 0. {
    s += ".";
    s
  } else {
    s
  }
}

fn write_swizzle(f: &mut impl fmt::Write, s: &Swizzle) -> Result<(), fmt::Error> {
  match s {
    Swizzle::D1(a) => write_swizzle_sel(f, a),

    Swizzle::D2(a, b) => {
      write_swizzle_sel(f, a)?;
      write_swizzle_sel(f, b)
    }

    Swizzle::D3(a, b, c) => {
      write_swizzle_sel(f, a)?;
      write_swizzle_sel(f, b)?;
      write_swizzle_sel(f, c)
    }

    Swizzle::D4(a, b, c, d) => {
      write_swizzle_sel(f, a)?;
      write_swizzle_sel(f, b)?;
      write_swizzle_sel(f, c)?;
      write_swizzle_sel(f, d)
    }
  }
}

fn write_swizzle_sel(f: &mut impl fmt::Write, d: &SwizzleSelector) -> Result<(), fmt::Error> {
  match d {
    SwizzleSelector::X => f.write_str("x"),
    SwizzleSelector::Y => f.write_str("y"),
    SwizzleSelector::Z => f.write_str("z"),
    SwizzleSelector::W => f.write_str("w"),
  }
}

fn write_fun_handle(f: &mut impl fmt::Write, fun: &ErasedFunHandle) -> Result<(), fmt::Error> {
  match fun {
    ErasedFunHandle::Vec2 => f.write_str("vec2"),
    ErasedFunHandle::Vec3 => f.write_str("vec3"),
    ErasedFunHandle::Vec4 => f.write_str("vec4"),
    ErasedFunHandle::Radians => f.write_str("radians"),
    ErasedFunHandle::Degrees => f.write_str("degrees"),
    ErasedFunHandle::Sin => f.write_str("sin"),
    ErasedFunHandle::Cos => f.write_str("cos"),
    ErasedFunHandle::Tan => f.write_str("tan"),
    ErasedFunHandle::ASin => f.write_str("asin"),
    ErasedFunHandle::ACos => f.write_str("acos"),
    ErasedFunHandle::ATan => f.write_str("atan"),
    ErasedFunHandle::SinH => f.write_str("asinh"),
    ErasedFunHandle::CosH => f.write_str("cosh"),
    ErasedFunHandle::TanH => f.write_str("tanh"),
    ErasedFunHandle::ASinH => f.write_str("asinh"),
    ErasedFunHandle::ACosH => f.write_str("acosh"),
    ErasedFunHandle::ATanH => f.write_str("atanh"),
    ErasedFunHandle::Pow => f.write_str("pow"),
    ErasedFunHandle::Exp => f.write_str("exp"),
    ErasedFunHandle::Exp2 => f.write_str("exp2"),
    ErasedFunHandle::Log => f.write_str("log"),
    ErasedFunHandle::Log2 => f.write_str("log2"),
    ErasedFunHandle::Sqrt => f.write_str("sqrt"),
    ErasedFunHandle::InverseSqrt => f.write_str("inversesqrt"),
    ErasedFunHandle::Abs => f.write_str("abs"),
    ErasedFunHandle::Sign => f.write_str("sign"),
    ErasedFunHandle::Floor => f.write_str("floor"),
    ErasedFunHandle::Trunc => f.write_str("trunc"),
    ErasedFunHandle::Round => f.write_str("round"),
    ErasedFunHandle::RoundEven => f.write_str("roundEven"),
    ErasedFunHandle::Ceil => f.write_str("ceil"),
    ErasedFunHandle::Fract => f.write_str("fract"),
    ErasedFunHandle::Min => f.write_str("min"),
    ErasedFunHandle::Max => f.write_str("max"),
    ErasedFunHandle::Clamp => f.write_str("clamp"),
    ErasedFunHandle::Mix => f.write_str("mix"),
    ErasedFunHandle::Step => f.write_str("step"),
    ErasedFunHandle::SmoothStep => f.write_str("smoothstep"),
    ErasedFunHandle::IsNan => f.write_str("isnan"),
    ErasedFunHandle::IsInf => f.write_str("isinf"),
    ErasedFunHandle::FloatBitsToInt => f.write_str("floatBitsToInt"),
    ErasedFunHandle::IntBitsToFloat => f.write_str("intBitsToFloat"),
    ErasedFunHandle::UIntBitsToFloat => f.write_str("uIntBitsToFloat"),
    ErasedFunHandle::FMA => f.write_str("fma"),
    ErasedFunHandle::Frexp => f.write_str("frexp"),
    ErasedFunHandle::Ldexp => f.write_str("ldexp"),
    ErasedFunHandle::PackUnorm2x16 => f.write_str("packUnorm2x16"),
    ErasedFunHandle::PackSnorm2x16 => f.write_str("packSnorm2x16"),
    ErasedFunHandle::PackUnorm4x8 => f.write_str("packUnorm4x8"),
    ErasedFunHandle::PackSnorm4x8 => f.write_str("packSnorm4x8"),
    ErasedFunHandle::UnpackUnorm2x16 => f.write_str("unpackUnorm2x16"),
    ErasedFunHandle::UnpackSnorm2x16 => f.write_str("unpackSnorm2x16"),
    ErasedFunHandle::UnpackUnorm4x8 => f.write_str("unpackUnorm4x8"),
    ErasedFunHandle::UnpackSnorm4x8 => f.write_str("unpackSnorm4x8"),
    ErasedFunHandle::PackHalf2x16 => f.write_str("packHalf2x16"),
    ErasedFunHandle::UnpackHalf2x16 => f.write_str("unpackHalf2x16"),
    ErasedFunHandle::Length => f.write_str("length"),
    ErasedFunHandle::Distance => f.write_str("distance"),
    ErasedFunHandle::Dot => f.write_str("dot"),
    ErasedFunHandle::Cross => f.write_str("cross"),
    ErasedFunHandle::Normalize => f.write_str("normalize"),
    ErasedFunHandle::FaceForward => f.write_str("faceforward"),
    ErasedFunHandle::Reflect => f.write_str("reflect"),
    ErasedFunHandle::Refract => f.write_str("refract"),
    ErasedFunHandle::VLt => f.write_str("lessThan"),
    ErasedFunHandle::VLte => f.write_str("lessThanEqual"),
    ErasedFunHandle::VGt => f.write_str("greaterThan"),
    ErasedFunHandle::VGte => f.write_str("greaterThanEqual"),
    ErasedFunHandle::VEq => f.write_str("equal"),
    ErasedFunHandle::VNeq => f.write_str("notEqual"),
    ErasedFunHandle::VAny => f.write_str("any"),
    ErasedFunHandle::VAll => f.write_str("all"),
    ErasedFunHandle::VNot => f.write_str("not"),
    ErasedFunHandle::UAddCarry => f.write_str("uaddCarry"),
    ErasedFunHandle::USubBorrow => f.write_str("usubBorrow"),
    ErasedFunHandle::UMulExtended => f.write_str("umulExtended"),
    ErasedFunHandle::IMulExtended => f.write_str("imulExtended"),
    ErasedFunHandle::BitfieldExtract => f.write_str("bitfieldExtract"),
    ErasedFunHandle::BitfieldInsert => f.write_str("bitfieldInsert"),
    ErasedFunHandle::BitfieldReverse => f.write_str("bitfieldReverse"),
    ErasedFunHandle::BitCount => f.write_str("bitCount"),
    ErasedFunHandle::FindLSB => f.write_str("findLSB"),
    ErasedFunHandle::FindMSB => f.write_str("findMSB"),
    ErasedFunHandle::EmitStreamVertex => f.write_str("EmitStreamVertex"),
    ErasedFunHandle::EndStreamPrimitive => f.write_str("EndStreamPrimitive"),
    ErasedFunHandle::EmitVertex => f.write_str("EmitVertex"),
    ErasedFunHandle::EndPrimitive => f.write_str("EndPrimitive"),
    ErasedFunHandle::DFDX => f.write_str("dfdx"),
    ErasedFunHandle::DFDY => f.write_str("dfdy"),
    ErasedFunHandle::DFDXFine => f.write_str("dfdxFine"),
    ErasedFunHandle::DFDYFine => f.write_str("dfdyFine"),
    ErasedFunHandle::DFDXCoarse => f.write_str("dfdxCoarse"),
    ErasedFunHandle::DFDYCoarse => f.write_str("dfdyCoarse"),
    ErasedFunHandle::FWidth => f.write_str("fwidth"),
    ErasedFunHandle::FWidthFine => f.write_str("fwidthFine"),
    ErasedFunHandle::FWidthCoarse => f.write_str("fwidthCoarse"),
    ErasedFunHandle::InterpolateAtCentroid => f.write_str("interpolateAtCentroid"),
    ErasedFunHandle::InterpolateAtSample => f.write_str("interpolateAtSample"),
    ErasedFunHandle::InterpolateAtOffset => f.write_str("interpolateAtOffset"),
    ErasedFunHandle::Barrier => f.write_str("barrier"),
    ErasedFunHandle::MemoryBarrier => f.write_str("memoryBarrier"),
    ErasedFunHandle::MemoryBarrierAtomic => f.write_str("memoryBarrierAtomic"),
    ErasedFunHandle::MemoryBarrierBuffer => f.write_str("memoryBarrierBuffer"),
    ErasedFunHandle::MemoryBarrierShared => f.write_str("memoryBarrierShared"),
    ErasedFunHandle::MemoryBarrierImage => f.write_str("memoryBarrierImage"),
    ErasedFunHandle::GroupMemoryBarrier => f.write_str("groupMemoryBarrier"),
    ErasedFunHandle::AnyInvocation => f.write_str("anyInvocation"),
    ErasedFunHandle::AllInvocations => f.write_str("allInvocations"),
    ErasedFunHandle::AllInvocationsEqual => f.write_str("allInvocationsEqual"),
    ErasedFunHandle::UserDefined(handle) => write_user_fun_handle(f, *handle),
  }
}

fn write_user_fun_handle(f: &mut impl fmt::Write, handle: u16) -> Result<(), fmt::Error> {
  write!(f, "fun_{}", handle)
}

fn write_var(f: &mut impl fmt::Write, handle: &ScopedHandle) -> Result<(), fmt::Error> {
  write_scoped_handle(f, handle)
}

fn write_scoped_handle(f: &mut impl fmt::Write, handle: &ScopedHandle) -> Result<(), fmt::Error> {
  match handle {
    ScopedHandle::BuiltIn(builtin) => write_builtin(f, builtin),

    ScopedHandle::Global(handle) => {
      write!(f, "glob_{}", handle)
    }

    ScopedHandle::FunArg(handle) => {
      write!(f, "arg_{}", handle)
    }

    ScopedHandle::FunVar { subscope, handle } => {
      write!(f, "var_{}_{}", subscope, handle)
    }

    ScopedHandle::Input(name) => f.write_str(name),

    ScopedHandle::Output(name) => f.write_str(name),

    ScopedHandle::Uniform(name) => f.write_str(name),
  }
}

fn write_builtin(f: &mut impl fmt::Write, builtin: &BuiltIn) -> Result<(), fmt::Error> {
  match builtin {
    BuiltIn::Vertex(builtin) => write_vert_builtin(f, builtin),
    BuiltIn::TessCtrl(builtin) => write_tess_ctrl_builtin(f, builtin),
    BuiltIn::TessEval(builtin) => write_tess_eval_builtin(f, builtin),
    BuiltIn::Geometry(builtin) => write_geo_builtin(f, builtin),
    BuiltIn::Fragment(builtin) => write_frag_builtin(f, builtin),
  }
}

fn write_vert_builtin(f: &mut impl fmt::Write, builtin: &VertexBuiltIn) -> Result<(), fmt::Error> {
  match builtin {
    VertexBuiltIn::VertexID => f.write_str("gl_VertexID"),
    VertexBuiltIn::InstanceID => f.write_str("gl_InstanceID"),
    VertexBuiltIn::BaseVertex => f.write_str("gl_BaseVertex"),
    VertexBuiltIn::BaseInstance => f.write_str("gl_BaseInstance"),
    VertexBuiltIn::Position => f.write_str("gl_Position"),
    VertexBuiltIn::PointSize => f.write_str("gl_PointSize"),
    VertexBuiltIn::ClipDistance => f.write_str("gl_ClipDistance"),
  }
}

fn write_tess_ctrl_builtin(
  f: &mut impl fmt::Write,
  builtin: &TessCtrlBuiltIn,
) -> Result<(), fmt::Error> {
  match builtin {
    TessCtrlBuiltIn::MaxPatchVerticesIn => f.write_str("gl_MaxPatchVerticesIn"),
    TessCtrlBuiltIn::PatchVerticesIn => f.write_str("gl_PatchVerticesIn"),
    TessCtrlBuiltIn::PrimitiveID => f.write_str("gl_PrimitiveID"),
    TessCtrlBuiltIn::InvocationID => f.write_str("gl_InvocationID"),
    TessCtrlBuiltIn::TessellationLevelOuter => f.write_str("gl_TessellationLevelOuter"),
    TessCtrlBuiltIn::TessellationLevelInner => f.write_str("gl_TessellationLevelInner"),
    TessCtrlBuiltIn::In => f.write_str("gl_In"),
    TessCtrlBuiltIn::Out => f.write_str("gl_Out"),
    TessCtrlBuiltIn::Position => f.write_str("gl_Position"),
    TessCtrlBuiltIn::PointSize => f.write_str("gl_PointSize"),
    TessCtrlBuiltIn::ClipDistance => f.write_str("gl_ClipDistance"),
    TessCtrlBuiltIn::CullDistance => f.write_str("gl_CullDistance"),
  }
}

fn write_tess_eval_builtin(
  f: &mut impl fmt::Write,
  builtin: &TessEvalBuiltIn,
) -> Result<(), fmt::Error> {
  match builtin {
    TessEvalBuiltIn::TessCoord => f.write_str("gl_TessCoord"),
    TessEvalBuiltIn::MaxPatchVerticesIn => f.write_str("gl_MaxPatchVerticesIn"),
    TessEvalBuiltIn::PatchVerticesIn => f.write_str("gl_PatchVerticesIn"),
    TessEvalBuiltIn::PrimitiveID => f.write_str("gl_PrimitiveID"),
    TessEvalBuiltIn::TessellationLevelOuter => f.write_str("gl_TessellationLevelOuter"),
    TessEvalBuiltIn::TessellationLevelInner => f.write_str("gl_TessellationLevelInner"),
    TessEvalBuiltIn::In => f.write_str("gl_In"),
    TessEvalBuiltIn::Out => f.write_str("gl_Out"),
    TessEvalBuiltIn::Position => f.write_str("gl_Position"),
    TessEvalBuiltIn::PointSize => f.write_str("gl_PointSize"),
    TessEvalBuiltIn::ClipDistance => f.write_str("gl_ClipDistance"),
    TessEvalBuiltIn::CullDistance => f.write_str("gl_CullDistance"),
  }
}

fn write_geo_builtin(f: &mut impl fmt::Write, builtin: &GeometryBuiltIn) -> Result<(), fmt::Error> {
  match builtin {
    GeometryBuiltIn::In => f.write_str("gl_In"),
    GeometryBuiltIn::Out => f.write_str("gl_Out"),
    GeometryBuiltIn::Position => f.write_str("gl_Position"),
    GeometryBuiltIn::PointSize => f.write_str("gl_PointSize"),
    GeometryBuiltIn::ClipDistance => f.write_str("gl_ClipDistance"),
    GeometryBuiltIn::CullDistance => f.write_str("gl_CullDistance"),
    GeometryBuiltIn::PrimitiveID => f.write_str("gl_PrimitiveID"),
    GeometryBuiltIn::PrimitiveIDIn => f.write_str("gl_PrimitiveIDIn"),
    GeometryBuiltIn::InvocationID => f.write_str("gl_InvocationID"),
    GeometryBuiltIn::Layer => f.write_str("gl_Layer"),
    GeometryBuiltIn::ViewportIndex => f.write_str("gl_ViewportIndex"),
  }
}

fn write_frag_builtin(
  f: &mut impl fmt::Write,
  builtin: &FragmentBuiltIn,
) -> Result<(), fmt::Error> {
  match builtin {
    FragmentBuiltIn::FragCoord => f.write_str("gl_FragCoord"),
    FragmentBuiltIn::FrontFacing => f.write_str("gl_FrontFacing"),
    FragmentBuiltIn::PointCoord => f.write_str("gl_PointCoord"),
    FragmentBuiltIn::SampleID => f.write_str("gl_SampleID"),
    FragmentBuiltIn::SamplePosition => f.write_str("gl_SamplePosition"),
    FragmentBuiltIn::SampleMaskIn => f.write_str("gl_SampleMaskIn"),
    FragmentBuiltIn::ClipDistance => f.write_str("gl_ClipDistance"),
    FragmentBuiltIn::CullDistance => f.write_str("gl_CullDistance"),
    FragmentBuiltIn::PrimitiveID => f.write_str("gl_PrimitiveID"),
    FragmentBuiltIn::Layer => f.write_str("gl_Layer"),
    FragmentBuiltIn::ViewportIndex => f.write_str("gl_ViewportIndex"),
    FragmentBuiltIn::FragDepth => f.write_str("gl_FragDepth"),
    FragmentBuiltIn::SampleMask => f.write_str("gl_SampleMask"),
    FragmentBuiltIn::HelperInvocation => f.write_str("gl_HelperInvocation"),
  }
}

fn write_prim_type(f: &mut impl fmt::Write, prim_ty: &PrimType) -> Result<(), fmt::Error> {
  let ty_str = match prim_ty {
    // ints
    PrimType::Int(Dim::Scalar) => "int",
    PrimType::Int(Dim::D2) => "ivec2",
    PrimType::Int(Dim::D3) => "ivec3",
    PrimType::Int(Dim::D4) => "ivec4",

    // uints
    PrimType::UInt(Dim::Scalar) => "uint",
    PrimType::UInt(Dim::D2) => "uvec2",
    PrimType::UInt(Dim::D3) => "uvec3",
    PrimType::UInt(Dim::D4) => "uvec4",

    // floats
    PrimType::Float(Dim::Scalar) => "float",
    PrimType::Float(Dim::D2) => "vec2",
    PrimType::Float(Dim::D3) => "vec3",
    PrimType::Float(Dim::D4) => "vec4",

    // booleans
    PrimType::Bool(Dim::Scalar) => "bool",
    PrimType::Bool(Dim::D2) => "bvec2",
    PrimType::Bool(Dim::D3) => "bvec3",
    PrimType::Bool(Dim::D4) => "bvec4",

    // matrices
    PrimType::Matrix(MatrixDim::D22) => "mat2",
    PrimType::Matrix(MatrixDim::D23) => "mat23",
    PrimType::Matrix(MatrixDim::D24) => "mat24",
    PrimType::Matrix(MatrixDim::D32) => "mat32",
    PrimType::Matrix(MatrixDim::D33) => "mat3",
    PrimType::Matrix(MatrixDim::D34) => "mat34",
    PrimType::Matrix(MatrixDim::D42) => "mat42",
    PrimType::Matrix(MatrixDim::D43) => "mat43",
    PrimType::Matrix(MatrixDim::D44) => "mat4",
  };

  f.write_str(ty_str)
}

fn write_type(f: &mut impl fmt::Write, ty: &Type) -> Result<(), fmt::Error> {
  write_prim_type(f, &ty.prim_ty)?;

  // array notation
  if !ty.array_dims.is_empty() {
    f.write_str("[")?;

    write!(f, "{}", &ty.array_dims[0])?;
    for dim in &ty.array_dims[1..] {
      write!(f, "][{}", dim)?;
    }

    f.write_str("]")
  } else {
    Ok(())
  }
}

fn write_indented(f: &mut impl fmt::Write, indent_lvl: usize, t: &str) -> Result<(), fmt::Error> {
  write_indent(f, indent_lvl)?;
  f.write_str(t)
}

fn write_indent(f: &mut impl fmt::Write, indent_lvl: usize) -> Result<(), fmt::Error> {
  write!(
    f,
    "{indent:<width$}",
    indent = " ",
    width = INDENT_SPACES * indent_lvl
  )
}

fn write_matrix<const M: usize, const N: usize>(
  f: &mut impl fmt::Write,
  ctor_name: &str,
  m: &[[f32; N]; M],
) -> Result<(), fmt::Error> {
  write!(f, "{}(", ctor_name)?;

  // first dimension
  write!(f, "{}", write_f32(m[0][0]))?;
  for i in 1..N {
    write!(f, ", {}", write_f32(m[0][i]))?;
  }

  // general case
  for y in 1..M {
    for i in 0..N {
      write!(f, ", {}", write_f32(m[y][i]))?;
    }
  }

  f.write_str(")")
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn matrices() {
    let mut output = String::new();

    write_matrix(&mut output, "mat2", &[[1., 2.], [3., 4.]]).unwrap();
    assert_eq!(output, "mat2(1., 2., 3., 4.)");

    output.clear();
    write_matrix(
      &mut output,
      "mat3",
      &[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
    )
    .unwrap();
    assert_eq!(output, "mat3(1., 2., 3., 4., 5., 6., 7., 8., 9.)");

    output.clear();
    write_matrix(
      &mut output,
      "mat4",
      &[
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.],
      ],
    )
    .unwrap();
    assert_eq!(
      output,
      "mat4(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.)"
    );
  }
}
