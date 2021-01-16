//! GLSL writers.

use crate::{
  BuiltIn, Dim, ErasedExpr, ErasedFun, ErasedFunHandle, ErasedReturn, ErasedScope, FragmentBuiltIn,
  GeometryBuiltIn, PrimType, ScopeInstr, ScopedHandle, Shader, ShaderDecl, Swizzle,
  SwizzleSelector, TessCtrlBuiltIn, TessEvalBuiltIn, Type, VertexBuiltIn,
};
use std::fmt;

// Number of space an indent level represents.
const INDENT_SPACES: usize = 2;

#[derive(Debug)]
pub enum WriteError {}

impl fmt::Display for WriteError {
  fn fmt(&self, _f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    Ok(()) // TODO: remove me
  }
}

impl std::error::Error for WriteError {}

pub fn write_shader_to_str(shader: &Shader) -> Result<String, WriteError> {
  let mut output = String::new();

  for decl in &shader.decls {
    match decl {
      ShaderDecl::Main(fun) => write_main_fun_to_str(&mut output, fun)?,
      ShaderDecl::FunDef(handle, fun) => write_fun_def_to_str(&mut output, *handle, fun)?,
      ShaderDecl::Const(handle, ty, ref constant) => {
        write_constant_to_str(&mut output, *handle, ty, constant)?
      }
      ShaderDecl::In(handle, ty) => write_input_to_str(&mut output, *handle, ty)?,
      ShaderDecl::Out(handle, ty) => write_output_to_str(&mut output, *handle, ty)?,
    }
  }

  Ok(output)
}

fn write_main_fun_to_str(output: &mut String, fun: &ErasedFun) -> Result<(), WriteError> {
  *output += "\nvoid main() {\n";
  write_scope_to_str(output, &fun.scope, 1)?;
  *output += "}";
  Ok(())
}

fn write_fun_def_to_str(
  output: &mut String,
  handle: u16,
  fun: &ErasedFun,
) -> Result<(), WriteError> {
  // just for aesthetics :')
  *output += "\n";

  let ret_expr = match &fun.ret {
    ErasedReturn::Void => {
      *output += "void";
      None
    }

    ErasedReturn::Expr(ref ty, expr) => {
      write_type_to_str(output, ty)?;
      Some(expr)
    }
  };

  *output += " ";
  write_fun_handle(output, handle)?;

  *output += "(";
  if !fun.args.is_empty() {
    write_type_to_str(output, &fun.args[0])?;
    *output += " arg_0";

    for (i, arg) in fun.args.iter().enumerate().skip(1) {
      *output += ", ";

      write_type_to_str(output, arg)?;
      *output += &format!(" arg_{}", i);
    }
  }
  *output += ") {\n";

  write_scope_to_str(output, &fun.scope, 1)?;

  if let Some(ref expr) = ret_expr {
    *output += &" ".repeat(INDENT_SPACES);
    *output += "return ";
    write_expr_to_str(output, expr)?;
    *output += ";";
  }

  *output += "\n}\n";
  Ok(())
}

fn write_scope_to_str(
  output: &mut String,
  scope: &ErasedScope,
  indent_lvl: usize,
) -> Result<(), WriteError> {
  let indent = " ".repeat(indent_lvl * INDENT_SPACES);

  for instr in &scope.instructions {
    // put the indent level
    *output += &indent;

    match instr {
      ScopeInstr::VarDecl {
        ty,
        handle,
        init_value,
      } => {
        write_type_to_str(output, ty)?;
        *output += " ";
        write_scoped_handle_to_str(output, handle)?;
        *output += " = ";
        write_expr_to_str(output, init_value)?;
        *output += ";";
      }

      ScopeInstr::Return(ret) => match ret {
        ErasedReturn::Void => *output += "return;",
        ErasedReturn::Expr(_, expr) => {
          *output += "return ";
          write_expr_to_str(output, expr)?;
          *output += ";";
        }
      },

      ScopeInstr::Continue => *output += "continue;",
      ScopeInstr::Break => *output += "break;",

      ScopeInstr::If { condition, scope } => {
        *output += "if (";
        write_expr_to_str(output, condition)?;
        *output += ") {\n";
        write_scope_to_str(output, scope, indent_lvl + 1)?;
        *output += "}";
      }

      ScopeInstr::ElseIf { condition, scope } => {
        *output += " else if (";
        write_expr_to_str(output, condition)?;
        *output += ") {\n";
        write_scope_to_str(output, scope, indent_lvl + 1)?;
        *output += "}";
      }

      ScopeInstr::Else { scope } => {
        *output += " else {\n";
        write_scope_to_str(output, scope, indent_lvl + 1)?;
        *output += "}";
      }

      ScopeInstr::For {
        init_ty,
        init_handle,
        init_expr,
        condition,
        post_expr,
        scope,
      } => {
        *output += "for (";

        // initialization
        write_type_to_str(output, init_ty)?;
        *output += " ";
        write_scoped_handle_to_str(output, init_handle)?;
        *output += " = ";
        write_expr_to_str(output, init_expr)?;
        *output += "; ";

        // condition
        write_expr_to_str(output, condition)?;
        *output += "; ";

        // iteration; we basically write <init-expr> = <next-expr> in a fold-like way, so we need to re-use the
        // init_handle
        write_scoped_handle_to_str(output, init_handle)?;
        *output += " = ";
        write_expr_to_str(output, post_expr)?;
        *output += ") {\n";

        // scope
        write_scope_to_str(output, scope, indent_lvl + 1)?;
        *output += "}";
      }

      ScopeInstr::While { condition, scope } => {
        *output += "while (";
        write_expr_to_str(output, condition)?;
        *output += ") {\n";
        write_scope_to_str(output, scope, indent_lvl + 1)?;
        *output += "}";
      }

      ScopeInstr::MutateVar { var, expr } => {
        write_expr_to_str(output, var)?;
        *output += " = ";
        write_expr_to_str(output, expr)?;
        *output += ";";
      }
    }

    *output += "\n";
  }

  Ok(())
}

fn write_constant_to_str(
  output: &mut String,
  handle: u16,
  ty: &Type,
  constant: &ErasedExpr,
) -> Result<(), WriteError> {
  *output += "const ";
  write_type_to_str(output, ty)?;
  *output += " ";
  write_scoped_handle_to_str(output, &ScopedHandle::global(handle))?;
  *output += " = ";
  write_expr_to_str(output, constant)?;
  *output += ";\n";

  Ok(())
}

fn write_input_to_str(output: &mut String, handle: u16, ty: &Type) -> Result<(), WriteError> {
  *output += "in ";
  write_type_to_str(output, ty)?;

  // the handle is treated as a global
  *output += " ";
  write_scoped_handle_to_str(output, &ScopedHandle::global(handle))?;

  *output += ";\n";

  Ok(())
}

fn write_output_to_str(output: &mut String, handle: u16, ty: &Type) -> Result<(), WriteError> {
  *output += "out ";
  write_type_to_str(output, ty)?;

  // the handle is treated as a global
  write_scoped_handle_to_str(output, &ScopedHandle::global(handle))?;

  *output += ";\n";

  Ok(())
}

fn write_expr_to_str(output: &mut String, expr: &ErasedExpr) -> Result<(), WriteError> {
  match expr {
    ErasedExpr::LitInt(x) => *output += &format!("{}", x),
    ErasedExpr::LitUInt(x) => *output += &format!("{}", x),
    ErasedExpr::LitFloat(x) => *output += &format!("{}", float_to_str(*x)),
    ErasedExpr::LitBool(x) => *output += &format!("{}", x),

    ErasedExpr::LitInt2([x, y]) => *output += &format!("ivec2({}, {})", x, y),
    ErasedExpr::LitUInt2([x, y]) => *output += &format!("uvec2({}, {})", x, y),
    ErasedExpr::LitFloat2([x, y]) => {
      *output += &format!("vec2({}, {})", float_to_str(*x), float_to_str(*y))
    }
    ErasedExpr::LitBool2([x, y]) => *output += &format!("bvec2({}, {})", x, y),

    ErasedExpr::LitInt3([x, y, z]) => *output += &format!("ivec3({}, {}, {})", x, y, z),
    ErasedExpr::LitUInt3([x, y, z]) => *output += &format!("uvec3({}, {}, {})", x, y, z),
    ErasedExpr::LitFloat3([x, y, z]) => {
      *output += &format!(
        "vec3({}, {}, {})",
        float_to_str(*x),
        float_to_str(*y),
        float_to_str(*z)
      )
    }
    ErasedExpr::LitBool3([x, y, z]) => *output += &format!("bvec3({}, {}, {})", x, y, z),

    ErasedExpr::LitInt4([x, y, z, w]) => *output += &format!("ivec4({}, {}, {}, {})", x, y, z, w),
    ErasedExpr::LitUInt4([x, y, z, w]) => *output += &format!("uvec4({}, {}, {}, {})", x, y, z, w),
    ErasedExpr::LitFloat4([x, y, z, w]) => {
      *output += &format!(
        "vec4({}, {}, {}, {})",
        float_to_str(*x),
        float_to_str(*y),
        float_to_str(*z),
        float_to_str(*w)
      )
    }
    ErasedExpr::LitBool4([x, y, z, w]) => *output += &format!("bvec4({}, {}, {}, {})", x, y, z, w),
    ErasedExpr::Array(ty, items) => {
      write_type_to_str(output, ty)?;
      *output += "(";

      write_expr_to_str(output, &items[0])?;
      for item in &items[1..] {
        *output += ", ";
        write_expr_to_str(output, item)?;
      }

      *output += ")";
    }

    ErasedExpr::MutVar(handle) => write_mut_var_to_str(output, handle)?,

    ErasedExpr::ImmutBuiltIn(builtin) => write_builtin_to_str(output, builtin)?,

    ErasedExpr::Not(e) => {
      *output += "!";
      write_expr_to_str(output, e)?;
    }

    ErasedExpr::And(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " && ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Or(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " || ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Xor(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " ^^ ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::BitAnd(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " & ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::BitOr(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " | ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::BitXor(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " ^ ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Neg(e) => {
      *output += "-(";
      write_expr_to_str(output, e)?;
      *output += ")";
    }

    ErasedExpr::Add(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " + ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Sub(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " - ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Mul(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " * ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Div(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " / ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Rem(a, b) => {
      *output += "mod(";
      write_expr_to_str(output, a)?;
      *output += ", ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Shl(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " << ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Shr(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " >> ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Eq(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " == ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Neq(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " != ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Lt(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " < ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Lte(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " <= ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Gt(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " > ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::Gte(a, b) => {
      *output += "(";
      write_expr_to_str(output, a)?;
      *output += " >= ";
      write_expr_to_str(output, b)?;
      *output += ")";
    }

    ErasedExpr::FunCall(f, args) => {
      write_fun_handle_to_str(output, f)?;
      *output += "(";

      if !args.is_empty() {
        write_expr_to_str(output, &args[0])?;
      }

      for arg in &args[1..] {
        *output += ", ";
        write_expr_to_str(output, arg)?;
      }

      *output += ")";
    }

    ErasedExpr::Swizzle(e, s) => {
      write_expr_to_str(output, e)?;
      *output += ".";
      write_swizzle_to_str(output, s)?;
    }

    ErasedExpr::Field { object, field } => {
      write_expr_to_str(output, object)?;
      *output += ".";
      write_expr_to_str(output, field)?;
    }

    ErasedExpr::ArrayLookup { object, index } => {
      write_expr_to_str(output, object)?;
      *output += "[";
      write_expr_to_str(output, index)?;
      *output += "]";
    }
  }

  Ok(())
}

fn float_to_str(f: f32) -> String {
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

fn write_swizzle_to_str(output: &mut String, s: &Swizzle) -> Result<(), WriteError> {
  match s {
    Swizzle::D1(a) => write_swizzle_sel_to_str(output, a)?,

    Swizzle::D2(a, b) => {
      write_swizzle_sel_to_str(output, a)?;
      write_swizzle_sel_to_str(output, b)?;
    }

    Swizzle::D3(a, b, c) => {
      write_swizzle_sel_to_str(output, a)?;
      write_swizzle_sel_to_str(output, b)?;
      write_swizzle_sel_to_str(output, c)?;
    }

    Swizzle::D4(a, b, c, d) => {
      write_swizzle_sel_to_str(output, a)?;
      write_swizzle_sel_to_str(output, b)?;
      write_swizzle_sel_to_str(output, c)?;
      write_swizzle_sel_to_str(output, d)?;
    }
  }

  Ok(())
}

fn write_swizzle_sel_to_str(output: &mut String, d: &SwizzleSelector) -> Result<(), WriteError> {
  match d {
    SwizzleSelector::X => *output += "x",
    SwizzleSelector::Y => *output += "y",
    SwizzleSelector::Z => *output += "z",
    SwizzleSelector::W => *output += "w",
  }

  Ok(())
}

fn write_fun_handle_to_str(output: &mut String, f: &ErasedFunHandle) -> Result<(), WriteError> {
  match f {
    ErasedFunHandle::Main => *output += "main",
    ErasedFunHandle::Radians => *output += "radians",
    ErasedFunHandle::Degrees => *output += "degrees",
    ErasedFunHandle::Sin => *output += "sin",
    ErasedFunHandle::Cos => *output += "cos",
    ErasedFunHandle::Tan => *output += "tan",
    ErasedFunHandle::ASin => *output += "asin",
    ErasedFunHandle::ACos => *output += "acos",
    ErasedFunHandle::ATan => *output += "atan",
    ErasedFunHandle::SinH => *output += "asinh",
    ErasedFunHandle::CosH => *output += "cosh",
    ErasedFunHandle::TanH => *output += "tanh",
    ErasedFunHandle::ASinH => *output += "asinh",
    ErasedFunHandle::ACosH => *output += "acosh",
    ErasedFunHandle::ATanH => *output += "atanh",
    ErasedFunHandle::Pow => *output += "pow",
    ErasedFunHandle::Exp => *output += "exp",
    ErasedFunHandle::Exp2 => *output += "exp2",
    ErasedFunHandle::Log => *output += "log",
    ErasedFunHandle::Log2 => *output += "log2",
    ErasedFunHandle::Sqrt => *output += "sqrt",
    ErasedFunHandle::InverseSqrt => *output += "inversesqrt",
    ErasedFunHandle::Abs => *output += "abs",
    ErasedFunHandle::Sign => *output += "sign",
    ErasedFunHandle::Floor => *output += "floor",
    ErasedFunHandle::Trunc => *output += "trunc",
    ErasedFunHandle::Round => *output += "round",
    ErasedFunHandle::RoundEven => *output += "roundEven",
    ErasedFunHandle::Ceil => *output += "ceil",
    ErasedFunHandle::Fract => *output += "fract",
    ErasedFunHandle::Min => *output += "min",
    ErasedFunHandle::Max => *output += "max",
    ErasedFunHandle::Clamp => *output += "clamp",
    ErasedFunHandle::Mix => *output += "mix",
    ErasedFunHandle::Step => *output += "step",
    ErasedFunHandle::SmoothStep => *output += "smoothstep",
    ErasedFunHandle::IsNan => *output += "isnan",
    ErasedFunHandle::IsInf => *output += "isinf",
    ErasedFunHandle::FloatBitsToInt => *output += "floatBitsToInt",
    ErasedFunHandle::IntBitsToFloat => *output += "intBitsToFloat",
    ErasedFunHandle::UIntBitsToFloat => *output += "uIntBitsToFloat",
    ErasedFunHandle::FMA => *output += "fma",
    ErasedFunHandle::Frexp => *output += "frexp",
    ErasedFunHandle::Ldexp => *output += "ldexp",
    ErasedFunHandle::PackUnorm2x16 => *output += "packUnorm2x16",
    ErasedFunHandle::PackSnorm2x16 => *output += "packSnorm2x16",
    ErasedFunHandle::PackUnorm4x8 => *output += "packUnorm4x8",
    ErasedFunHandle::PackSnorm4x8 => *output += "packSnorm4x8",
    ErasedFunHandle::UnpackUnorm2x16 => *output += "unpackUnorm2x16",
    ErasedFunHandle::UnpackSnorm2x16 => *output += "unpackSnorm2x16",
    ErasedFunHandle::UnpackUnorm4x8 => *output += "unpackUnorm4x8",
    ErasedFunHandle::UnpackSnorm4x8 => *output += "unpackSnorm4x8",
    ErasedFunHandle::PackHalf2x16 => *output += "packHalf2x16",
    ErasedFunHandle::UnpackHalf2x16 => *output += "unpackHalf2x16",
    ErasedFunHandle::Length => *output += "length",
    ErasedFunHandle::Distance => *output += "distance",
    ErasedFunHandle::Dot => *output += "dot",
    ErasedFunHandle::Cross => *output += "cross",
    ErasedFunHandle::Normalize => *output += "normalize",
    ErasedFunHandle::FaceForward => *output += "faceforward",
    ErasedFunHandle::Reflect => *output += "reflect",
    ErasedFunHandle::Refract => *output += "refract",
    ErasedFunHandle::VLt => *output += "lessThan",
    ErasedFunHandle::VLte => *output += "lessThanEqual",
    ErasedFunHandle::VGt => *output += "greaterThan",
    ErasedFunHandle::VGte => *output += "greaterThanEqual",
    ErasedFunHandle::VEq => *output += "equal",
    ErasedFunHandle::VNeq => *output += "notEqual",
    ErasedFunHandle::VAny => *output += "any",
    ErasedFunHandle::VAll => *output += "all",
    ErasedFunHandle::VNot => *output += "not",
    ErasedFunHandle::UAddCarry => *output += "uaddCarry",
    ErasedFunHandle::USubBorrow => *output += "usubBorrow",
    ErasedFunHandle::UMulExtended => *output += "umulExtended",
    ErasedFunHandle::IMulExtended => *output += "imulExtended",
    ErasedFunHandle::BitfieldExtract => *output += "bitfieldExtract",
    ErasedFunHandle::BitfieldInsert => *output += "bitfieldInsert",
    ErasedFunHandle::BitfieldReverse => *output += "bitfieldReverse",
    ErasedFunHandle::BitCount => *output += "bitCount",
    ErasedFunHandle::FindLSB => *output += "findLSB",
    ErasedFunHandle::FindMSB => *output += "findMSB",
    ErasedFunHandle::EmitStreamVertex => *output += "EmitStreamVertex",
    ErasedFunHandle::EndStreamPrimitive => *output += "EndStreamPrimitive",
    ErasedFunHandle::EmitVertex => *output += "EmitVertex",
    ErasedFunHandle::EndPrimitive => *output += "EndPrimitive",
    ErasedFunHandle::DFDX => *output += "dfdx",
    ErasedFunHandle::DFDY => *output += "dfdy",
    ErasedFunHandle::DFDXFine => *output += "dfdxFine",
    ErasedFunHandle::DFDYFine => *output += "dfdyFine",
    ErasedFunHandle::DFDXCoarse => *output += "dfdxCoarse",
    ErasedFunHandle::DFDYCoarse => *output += "dfdyCoarse",
    ErasedFunHandle::FWidth => *output += "fwidth",
    ErasedFunHandle::FWidthFine => *output += "fwidthFine",
    ErasedFunHandle::FWidthCoarse => *output += "fwidthCoarse",
    ErasedFunHandle::InterpolateAtCentroid => *output += "interpolateAtCentroid",
    ErasedFunHandle::InterpolateAtSample => *output += "interpolateAtSample",
    ErasedFunHandle::InterpolateAtOffset => *output += "interpolateAtOffset",
    ErasedFunHandle::Barrier => *output += "barrier",
    ErasedFunHandle::MemoryBarrier => *output += "memoryBarrier",
    ErasedFunHandle::MemoryBarrierAtomic => *output += "memoryBarrierAtomic",
    ErasedFunHandle::MemoryBarrierBuffer => *output += "memoryBarrierBuffer",
    ErasedFunHandle::MemoryBarrierShared => *output += "memoryBarrierShared",
    ErasedFunHandle::MemoryBarrierImage => *output += "memoryBarrierImage",
    ErasedFunHandle::GroupMemoryBarrier => *output += "groupMemoryBarrier",
    ErasedFunHandle::AnyInvocation => *output += "anyInvocation",
    ErasedFunHandle::AllInvocations => *output += "allInvocations",
    ErasedFunHandle::AllInvocationsEqual => *output += "allInvocationsEqual",
    ErasedFunHandle::UserDefined(handle) => write_fun_handle(output, *handle)?,
  }

  Ok(())
}

fn write_fun_handle(output: &mut String, handle: u16) -> Result<(), WriteError> {
  *output += &format!("fun_{}", handle);
  Ok(())
}

fn write_mut_var_to_str(output: &mut String, handle: &ScopedHandle) -> Result<(), WriteError> {
  write_scoped_handle_to_str(output, handle)
}

fn write_scoped_handle_to_str(
  output: &mut String,
  handle: &ScopedHandle,
) -> Result<(), WriteError> {
  match handle {
    ScopedHandle::BuiltIn(builtin) => write_builtin_to_str(output, builtin)?,

    ScopedHandle::Global(handle) => {
      *output += &format!("glob_{}", handle);
    }

    ScopedHandle::FunArg(handle) => {
      *output += &format!("arg_{}", handle);
    }

    ScopedHandle::FunVar { subscope, handle } => {
      *output += &format!("var_{}_{}", subscope, handle);
    }
  }

  Ok(())
}

fn write_builtin_to_str(output: &mut String, builtin: &BuiltIn) -> Result<(), WriteError> {
  match builtin {
    BuiltIn::Vertex(builtin) => write_vert_builtin_to_str(output, builtin),
    BuiltIn::TessCtrl(builtin) => write_tess_ctrl_builtin_to_str(output, builtin),
    BuiltIn::TessEval(builtin) => write_tess_eval_builtin_to_str(output, builtin),
    BuiltIn::Geometry(builtin) => write_geo_builtin_to_str(output, builtin),
    BuiltIn::Fragment(builtin) => write_frag_builtin_to_str(output, builtin),
  }
}

fn write_vert_builtin_to_str(
  output: &mut String,
  builtin: &VertexBuiltIn,
) -> Result<(), WriteError> {
  match builtin {
    VertexBuiltIn::VertexID => *output += "gl_VertexID",
    VertexBuiltIn::InstanceID => *output += "gl_InstanceID",
    VertexBuiltIn::BaseVertex => *output += "gl_BaseVertex",
    VertexBuiltIn::BaseInstance => *output += "gl_BaseInstance",
    VertexBuiltIn::Position => *output += "gl_Position",
    VertexBuiltIn::PointSize => *output += "gl_PointSize",
    VertexBuiltIn::ClipDistance => *output += "gl_ClipDistance",
  }

  Ok(())
}

fn write_tess_ctrl_builtin_to_str(
  output: &mut String,
  builtin: &TessCtrlBuiltIn,
) -> Result<(), WriteError> {
  match builtin {
    TessCtrlBuiltIn::MaxPatchVerticesIn => *output += "gl_MaxPatchVerticesIn",
    TessCtrlBuiltIn::PatchVerticesIn => *output += "gl_PatchVerticesIn",
    TessCtrlBuiltIn::PrimitiveID => *output += "gl_PrimitiveID",
    TessCtrlBuiltIn::InvocationID => *output += "gl_InvocationID",
    TessCtrlBuiltIn::TessellationLevelOuter => *output += "gl_TessellationLevelOuter",
    TessCtrlBuiltIn::TessellationLevelInner => *output += "gl_TessellationLevelInner",
    TessCtrlBuiltIn::In => *output += "gl_In",
    TessCtrlBuiltIn::Out => *output += "gl_Out",
    TessCtrlBuiltIn::Position => *output += "gl_Position",
    TessCtrlBuiltIn::PointSize => *output += "gl_PointSize",
    TessCtrlBuiltIn::ClipDistance => *output += "gl_ClipDistance",
    TessCtrlBuiltIn::CullDistance => *output += "gl_CullDistance",
  }

  Ok(())
}

fn write_tess_eval_builtin_to_str(
  output: &mut String,
  builtin: &TessEvalBuiltIn,
) -> Result<(), WriteError> {
  match builtin {
    TessEvalBuiltIn::TessCoord => *output += "gl_TessCoord",
    TessEvalBuiltIn::MaxPatchVerticesIn => *output += "gl_MaxPatchVerticesIn",
    TessEvalBuiltIn::PatchVerticesIn => *output += "gl_PatchVerticesIn",
    TessEvalBuiltIn::PrimitiveID => *output += "gl_PrimitiveID",
    TessEvalBuiltIn::TessellationLevelOuter => *output += "gl_TessellationLevelOuter",
    TessEvalBuiltIn::TessellationLevelInner => *output += "gl_TessellationLevelInner",
    TessEvalBuiltIn::In => *output += "gl_In",
    TessEvalBuiltIn::Out => *output += "gl_Out",
    TessEvalBuiltIn::Position => *output += "gl_Position",
    TessEvalBuiltIn::PointSize => *output += "gl_PointSize",
    TessEvalBuiltIn::ClipDistance => *output += "gl_ClipDistance",
    TessEvalBuiltIn::CullDistance => *output += "gl_CullDistance",
  }

  Ok(())
}

fn write_geo_builtin_to_str(
  output: &mut String,
  builtin: &GeometryBuiltIn,
) -> Result<(), WriteError> {
  match builtin {
    GeometryBuiltIn::In => *output += "gl_In",
    GeometryBuiltIn::Out => *output += "gl_Out",
    GeometryBuiltIn::Position => *output += "gl_Position",
    GeometryBuiltIn::PointSize => *output += "gl_PointSize",
    GeometryBuiltIn::ClipDistance => *output += "gl_ClipDistance",
    GeometryBuiltIn::CullDistance => *output += "gl_CullDistance",
    GeometryBuiltIn::PrimitiveID => *output += "gl_PrimitiveID",
    GeometryBuiltIn::PrimitiveIDIn => *output += "gl_PrimitiveIDIn",
    GeometryBuiltIn::InvocationID => *output += "gl_InvocationID",
    GeometryBuiltIn::Layer => *output += "gl_Layer",
    GeometryBuiltIn::ViewportIndex => *output += "gl_ViewportIndex",
  }

  Ok(())
}

fn write_frag_builtin_to_str(
  output: &mut String,
  builtin: &FragmentBuiltIn,
) -> Result<(), WriteError> {
  match builtin {
    FragmentBuiltIn::FragCoord => *output += "gl_FragCoord",
    FragmentBuiltIn::FrontFacing => *output += "gl_FrontFacing",
    FragmentBuiltIn::PointCoord => *output += "gl_PointCoord",
    FragmentBuiltIn::SampleID => *output += "gl_SampleID",
    FragmentBuiltIn::SamplePosition => *output += "gl_SamplePosition",
    FragmentBuiltIn::SampleMaskIn => *output += "gl_SampleMaskIn",
    FragmentBuiltIn::ClipDistance => *output += "gl_ClipDistance",
    FragmentBuiltIn::CullDistance => *output += "gl_CullDistance",
    FragmentBuiltIn::PrimitiveID => *output += "gl_PrimitiveID",
    FragmentBuiltIn::Layer => *output += "gl_Layer",
    FragmentBuiltIn::ViewportIndex => *output += "gl_ViewportIndex",
    FragmentBuiltIn::FragDepth => *output += "gl_FragDepth",
    FragmentBuiltIn::SampleMask => *output += "gl_SampleMask",
    FragmentBuiltIn::HelperInvocation => *output += "gl_HelperInvocation",
  }

  Ok(())
}

fn write_type_to_str(output: &mut String, ty: &Type) -> Result<(), WriteError> {
  let ty_str = match ty.prim_ty {
    PrimType::Int(Dim::Scalar) => "int",
    PrimType::Int(Dim::D2) => "ivec2",
    PrimType::Int(Dim::D3) => "ivec3",
    PrimType::Int(Dim::D4) => "ivec4",

    PrimType::UInt(Dim::Scalar) => "uint",
    PrimType::UInt(Dim::D2) => "uvec2",
    PrimType::UInt(Dim::D3) => "uvec3",
    PrimType::UInt(Dim::D4) => "uvec4",

    PrimType::Float(Dim::Scalar) => "float",
    PrimType::Float(Dim::D2) => "vec2",
    PrimType::Float(Dim::D3) => "vec3",
    PrimType::Float(Dim::D4) => "vec4",

    PrimType::Bool(Dim::Scalar) => "bool",
    PrimType::Bool(Dim::D2) => "bvec2",
    PrimType::Bool(Dim::D3) => "bvec3",
    PrimType::Bool(Dim::D4) => "bvec4",
  };
  *output += ty_str;

  // array notation
  if !ty.array_dims.is_empty() {
    *output += "[";

    *output += &ty.array_dims[0].to_string();
    for dim in &ty.array_dims[1..] {
      *output += &format!("][{}", dim);
    }

    *output += "]";
  }

  Ok(())
}
