use syn::{
  braced, parenthesized,
  parse::Parse,
  punctuated::Punctuated,
  token::{Brace, Paren},
  Expr, ExprAssign, ExprAssignOp, Ident, Token, Type,
};

/// A stage declaration with its environment.
#[derive(Debug)]
pub struct StageDecl {
  left_or: Token![|],
  input: FnArgItem,
  comma_input_token: Token![,],
  output: FnArgItem,
  comma_output_token: Token![,],
  env: FnArgItem,
  right_or: Token![|],
  brace_token: Brace,
  stage_item: StageItem,
}

impl Parse for StageDecl {
  fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
    let left_or = input.parse()?;
    let input_ = input.parse()?;
    let comma_input_token = input.parse()?;
    let output = input.parse()?;
    let comma_output_token = input.parse()?;
    let env = input.parse()?;
    let right_or = input.parse()?;

    let stage_input;
    let brace_token = braced!(stage_input in input);
    let stage_item = stage_input.parse()?;

    Ok(Self {
      left_or,
      input: input_,
      comma_input_token,
      output,
      comma_output_token,
      env,
      right_or,
      brace_token,
      stage_item,
    })
  }
}

/// A stage.
///
/// A stage contains global declarations.
#[derive(Debug)]
pub struct StageItem {
  glob_decl: Vec<ShaderDeclItem>,
}

impl Parse for StageItem {
  fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
    let mut glob_decl = Vec::new();

    loop {
      let lookahead = input.lookahead1();

      if lookahead.peek(Token![const]) {
        glob_decl.push(ShaderDeclItem::Const(input.parse()?));
      } else if lookahead.peek(Token![fn]) {
        glob_decl.push(ShaderDeclItem::FunDef(input.parse()?));
      } else {
        break Ok(StageItem { glob_decl });
      }
    }
  }
}

#[derive(Debug)]
pub enum ShaderDeclItem {
  Const(ConstItem),
  FunDef(FunDefItem),
}

#[derive(Debug)]
pub struct ConstItem {
  const_token: Token![const],
  ident: Ident,
  colon_token: Token![:],
  ty: Type,
  assign_token: Token![=],
  expr: Expr,
  semi_token: Token![;],
}

impl Parse for ConstItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let const_token = input.parse()?;
    let ident = input.parse()?;
    let colon_token = input.parse()?;
    let ty = input.parse()?;
    let assign_token = input.parse()?;
    let expr = input.parse()?;
    let semi_token = input.parse()?;

    Ok(Self {
      const_token,
      ident,
      colon_token,
      ty,
      assign_token,
      expr,
      semi_token,
    })
  }
}

#[derive(Debug)]
pub struct FnArgItem {
  ident: Ident,
  colon_token: Token![:],
  pound_token: Option<Token![#]>,
  ty: Type,
}

impl Parse for FnArgItem {
  fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
    let ident = input.parse()?;
    let colon_token = input.parse()?;
    let pound_token = input.parse()?;
    let ty = input.parse()?;

    Ok(Self {
      ident,
      colon_token,
      pound_token,
      ty,
    })
  }
}

#[derive(Debug)]
pub struct FunDefItem {
  fn_token: Token![fn],
  name: Ident,
  paren_token: Paren,
  args: Punctuated<FnArgItem, Token![,]>,
  arrow_token: Token![->],
  ret_ty: Type,
  brace_token: Brace,
  body: Punctuated<ScopeInstrItem, Token![;]>,
}

impl Parse for FunDefItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let fn_token = input.parse()?;
    let name = input.parse()?;

    let args_input;
    let paren_token = parenthesized!(args_input in input);
    let args = Punctuated::parse_terminated(&args_input)?;

    let arrow_token = input.parse()?;
    let ret_ty = input.parse()?;

    let body_input;
    let brace_token = braced!(body_input in input);
    let body = Punctuated::parse_terminated(&body_input)?;

    Ok(Self {
      fn_token,
      name,
      paren_token,
      args,
      arrow_token,
      ret_ty,
      brace_token,
      body,
    })
  }
}

#[derive(Debug)]
pub enum ScopeInstrItem {
  VarDecl(VarDeclItem),
  Return(ReturnItem),
  Continue(ContinueItem),
  Break(BreakItem),
  If(IfItem),
  Else(ElseTailItem),
  // For(ForItem), // TODO
  While(WhileItem),
  // MutateVar(MutateVarItem),
}

impl Parse for ScopeInstrItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let lookahead = input.lookahead1();

    let v = if lookahead.peek(Token![let]) {
      let v = input.parse()?;
      ScopeInstrItem::VarDecl(v)
    } else if lookahead.peek(Token![return]) {
      let v = input.parse()?;
      ScopeInstrItem::Return(v)
    } else if lookahead.peek(Token![continue]) {
      let v = input.parse()?;
      ScopeInstrItem::Continue(v)
    } else if lookahead.peek(Token![break]) {
      let v = input.parse()?;
      ScopeInstrItem::Break(v)
    } else if lookahead.peek(Token![if]) {
      let v = input.parse()?;
      ScopeInstrItem::If(v)
    } else if lookahead.peek(Token![else]) {
      let v = input.parse()?;
      ScopeInstrItem::If(v)
    } else {
      return Err(input.error(format!(
        "unknown kind of scope instruction: {:?}",
        lookahead.error()
      )));
    };

    Ok(v)
  }
}

#[derive(Debug)]
pub struct VarDeclItem {
  let_token: Token![let],
  name: Ident,
  ty: Option<(Token![:], Type)>,
  assign_token: Token![=],
  expr: Expr,
}

impl Parse for VarDeclItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let let_token = input.parse()?;
    let name = input.parse()?;

    let lookahead = input.lookahead1();
    let ty = if lookahead.peek(Token![:]) {
      let colon_token = input.parse()?;
      let ty = input.parse()?;
      Some((colon_token, ty))
    } else {
      None
    };
    let assign_token = input.parse()?;
    let expr = input.parse()?;

    Ok(Self {
      let_token,
      name,
      ty,
      assign_token,
      expr,
    })
  }
}

#[derive(Debug)]
pub struct ReturnItem {
  return_token: Option<Token![return]>,
  expr: Expr,
}

impl Parse for ReturnItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let return_token = input.parse()?;
    let expr = input.parse()?;

    Ok(Self { return_token, expr })
  }
}

#[derive(Debug)]
pub struct ContinueItem(Token![continue]);

impl Parse for ContinueItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let continue_token = input.parse()?;
    Ok(Self(continue_token))
  }
}

#[derive(Debug)]
pub struct BreakItem(Token![break]);

impl Parse for BreakItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let break_item = input.parse()?;
    Ok(Self(break_item))
  }
}

#[derive(Debug)]
pub struct IfItem {
  if_token: Token![if],
  paren_token: Paren,
  cond_expr: Expr,
  brace_token: Brace,
  body: Punctuated<ScopeInstrItem, Token![;]>,
}

impl Parse for IfItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let if_token = input.parse()?;

    let cond_input;
    let paren_token = parenthesized!(cond_input in input);
    let cond_expr = cond_input.parse()?;

    let body_input;
    let brace_token = braced!(body_input in input);
    let body = Punctuated::parse_terminated(&body_input)?;

    Ok(Self {
      if_token,
      paren_token,
      cond_expr,
      brace_token,
      body,
    })
  }
}

#[derive(Debug)]
pub struct ElseItem {
  else_token: Token![else],
  else_tail: ElseTailItem,
}

impl Parse for ElseItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let else_token = input.parse()?;
    let else_tail = input.parse()?;

    Ok(Self {
      else_token,
      else_tail,
    })
  }
}

#[derive(Debug)]
pub enum ElseTailItem {
  If(IfItem),

  Else {
    brace_token: Brace,
    body: Punctuated<ScopeInstrItem, Token![;]>,
  },
}

impl Parse for ElseTailItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let lookahead = input.lookahead1();

    let tail = if lookahead.peek(Token![if]) {
      let if_item = input.parse()?;
      ElseTailItem::If(if_item)
    } else {
      let body_input;
      let brace_token = braced!(body_input in input);
      let body = Punctuated::parse_terminated(&body_input)?;

      ElseTailItem::Else { brace_token, body }
    };

    Ok(tail)
  }
}

#[derive(Debug)]
pub struct WhileItem {
  while_token: Token![while],
  paren_token: Paren,
  cond_expr: Expr,
  brace_token: Brace,
  body: Punctuated<ScopeInstrItem, Token![;]>,
}

impl Parse for WhileItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let while_token = input.parse()?;

    let cond_input;
    let paren_token = parenthesized!(cond_input in input);
    let cond_expr = cond_input.parse()?;

    let body_input;
    let brace_token = braced!(body_input in input);
    let body = Punctuated::parse_terminated(&body_input)?;

    Ok(Self {
      while_token,
      paren_token,
      cond_expr,
      brace_token,
      body,
    })
  }
}

#[derive(Debug)]
pub enum MutateVarItem {
  Assign(ExprAssign),
  AssignOp(ExprAssignOp),
}
