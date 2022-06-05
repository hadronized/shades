use syn::{
  braced, parenthesized,
  parse::{Nothing, Parse},
  punctuated::Punctuated,
  token::{Brace, Paren},
  Expr, ExprAssign, ExprAssignOp, Ident, Token, Type,
};

/// A stage.
///
/// A stage contains global declarations.
#[derive(Debug)]
pub struct StageItem {
  glob_decl: Vec<ShaderDeclItem>,
}

#[derive(Debug)]
pub enum ShaderDeclItem {
  Const(ConstItem),
  FunDef(FunDefItem),
  Main(FunDefItem),
}

#[derive(Debug)]
pub struct ConstItem {
  const_token: Token![const],
  ident: Ident,
  colon_token: Token![:],
  ty: Type,
  assign_token: Token![=],
  expr: Expr,
}

impl Parse for ConstItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let const_token = input.parse()?;
    let ident = input.parse()?;
    let colon_token = input.parse()?;
    let ty = input.parse()?;
    let assign_token = input.parse()?;
    let expr = input.parse()?;

    Ok(Self {
      const_token,
      ident,
      colon_token,
      ty,
      assign_token,
      expr,
    })
  }
}

#[derive(Debug)]
pub struct FunDefItem {
  fn_token: Token![fn],
  name: Ident,
  paren_token: Paren,
  args: Nothing,
  brace_token: Brace,
  body: Punctuated<ScopeInstrItem, Token![;]>,
}

impl Parse for FunDefItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let fn_token = input.parse()?;
    let name = input.parse()?;
    let args_input;
    let paren_token = parenthesized!(args_input in input);
    let args = args_input.parse()?;
    let body_input;
    let brace_token = braced!(body_input in input);
    let body = body_input.parse()?;

    Ok(Self {
      fn_token,
      name,
      paren_token,
      args,
      brace_token,
      body,
    })
  }
}

#[derive(Debug)]
pub struct MainItem {
  fn_token: Token![fn],
  name: Ident,
  paren_token: Paren,
  args: Nothing,
  brace_token: Brace,
  body: Punctuated<ScopeInstrItem, Token![;]>,
}

#[derive(Debug)]
pub enum ScopeInstrItem {
  VarDecl(VarDeclItem),
  Return(ReturnItem),
  Continue(ContinueItem),
  Break(BreakItem),
  If(IfItem),
  ElseIf(ElseIfItem),
  Else(ElseItem),
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
    } else {
      return Err(input.error("unknown kind of scope instruction"));
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
  return_token: Token![return],
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

    let cond_content;
    let paren_token = parenthesized!(cond_content in input);
    let cond_expr = cond_content.parse()?;

    let body_content;
    let brace_token = braced!(body_content in input);
    let body = body_content.parse()?;

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
pub struct ElseIfItem {
  else_token: Token![else],
  if_token: Token![if],
  paren_token: Paren,
  cond_expr: Expr,
  brace_token: Brace,
  body: Punctuated<ScopeInstrItem, Token![;]>,
}

impl Parse for ElseIfItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let else_token = input.parse()?;
    let if_token = input.parse()?;

    let cond_content;
    let paren_token = parenthesized!(cond_content in input);
    let cond_expr = cond_content.parse()?;

    let body_content;
    let brace_token = braced!(body_content in input);
    let body = body_content.parse()?;

    Ok(Self {
      else_token,
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
  brace_token: Brace,
  body: Punctuated<ScopeInstrItem, Token![;]>,
}

impl Parse for ElseItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let else_token = input.parse()?;

    let body_content;
    let brace_token = parenthesized!(body_content in input);
    let body = body_content.parse()?;

    Ok(Self {
      else_token,
      brace_token,
      body,
    })
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

    let cond_content;
    let paren_token = parenthesized!(cond_content in input);
    let cond_expr = cond_content.parse()?;

    let body_content;
    let brace_token = braced!(body_content in input);
    let body = body_content.parsee()?;

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
