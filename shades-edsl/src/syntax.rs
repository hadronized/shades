use quote::{quote, ToTokens};
use syn::{
  braced, parenthesized,
  parse::Parse,
  parse_quote,
  punctuated::Punctuated,
  token::{Brace, Paren, Pound},
  Expr, ExprAssign, ExprAssignOp, Ident, Token, Type,
};

/// A stage declaration with its environment.
#[derive(Debug)]
pub struct StageDecl {
  stage_ty: Ident,
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

impl ToTokens for StageDecl {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let stage_ty = self.stage_ty.to_string();

    let q = match stage_ty.as_str() {
      "vertex" => {
        let input = FnArgItem {
          pound_token: Some(Pound::default()),
          ..self.input.clone()
        };
        let output = FnArgItem {
          pound_token: Some(Pound::default()),
          ..self.output.clone()
        };
        // TODO: env
        let stage = &self.stage_item;

        quote! {
          shades::stage::StageBuilder::new(|mut __builder, #input, #output| {
            #stage
          })
        }
      }

      stage => quote! { compile_error(format!("{stage} is not a valid stage type")) },
    };

    q.to_tokens(tokens);
  }
}

impl Parse for StageDecl {
  fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
    let stage_ty = input.parse()?;
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
      stage_ty,
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

impl ToTokens for StageItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let glob_decl = self.glob_decl.iter();
    let q = quote! { #(#glob_decl)* };

    q.to_tokens(tokens);
  }
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

impl ToTokens for ShaderDeclItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let q = match self {
      ShaderDeclItem::Const(const_item) => quote! { #const_item },
      ShaderDeclItem::FunDef(fundef_item) => quote! { #fundef_item },
    };

    q.to_tokens(tokens);
  }
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

impl ToTokens for ConstItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let ident = &self.ident;
    let ty = &self.ty;
    let expr = &self.expr;
    let q = quote! {
      let #ident: shades::expr::Expr<#ty> = __builder.constant(shades::expr::Expl::lit(#expr));
    };

    q.to_tokens(tokens);
  }
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

#[derive(Clone, Debug)]
pub struct FnArgItem {
  ident: Ident,
  colon_token: Token![:],
  pound_token: Option<Token![#]>,
  ty: Type,
}

impl ToTokens for FnArgItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let ident = &self.ident;
    let ty = if self.pound_token.is_some() {
      // if we have a #ty, we do not lift the type and use it as verbatim
      self.ty.clone()
    } else {
      // if we don’t have a #ty but just ty, we lift the type in Expr<#ty>
      let ty = &self.ty;
      parse_quote! { shades::expr::Expr<#ty> }
    };
    let q = quote! {
      #ident: #ty
    };

    q.to_tokens(tokens);
  }
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
  body: ScopeInstrItems,
}

impl ToTokens for FunDefItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let name = &self.name;
    let args = self.args.iter();
    // let ret_ty = &self.ret_ty;
    let body = &self.body;

    let q = if name.to_string() == "main" {
      quote! {
        __builder.main_fun(|__scope: shades::scope::Scope<()>, #(#args),*| { #body });
      }
    } else {
      quote! {
       let #name = __builder.fun(|__scope: shades::scope::Scope<()>, #(#args),*| { #body });
      }
    };

    q.to_tokens(tokens);
  }
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
    let body = body_input.parse()?;

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
pub struct ScopeInstrItems {
  items: Vec<ScopeInstrItem>,
}

impl ToTokens for ScopeInstrItems {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let items = &self.items;
    let q = quote! { #(#items)*};
    q.to_tokens(tokens);
  }
}

impl Parse for ScopeInstrItems {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let mut items = Vec::new();

    loop {
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
        break;
      };

      items.push(v);
    }

    Ok(Self { items })
  }
}

#[derive(Debug)]
pub enum ScopeInstrItem {
  VarDecl(VarDeclItem),
  Return(ReturnItem),
  Continue(ContinueItem),
  Break(BreakItem),
  If(IfItem),
  // For(ForItem), // TODO
  While(WhileItem),
  // MutateVar(MutateVarItem),
}

impl ToTokens for ScopeInstrItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let q = match self {
      ScopeInstrItem::VarDecl(decl) => {
        // FIXME: lift the whole expression
        let name = &decl.name;
        let expr = &decl.expr;

        if let Some((_, ty)) = &decl.ty {
          quote! {
            let #name: shades::var::Var<#ty> = __scope.var(shades::expr::Expr::from(#expr));
          }
        } else {
          quote! {
            let #name = __scope.var(shades::expr::Expr::from(#expr));
          }
        }
      }

      ScopeInstrItem::Return(ret) => {
        let expr = &ret.expr;

        quote! {
          __scope.leave(shades::expr::Expr::from(#expr));
        }
      }

      ScopeInstrItem::Continue(_) => {
        quote! {
          __scope.loop_continue();
        }
      }

      ScopeInstrItem::Break(_) => {
        quote! {
          __scope.loop_break();
        }
      }

      ScopeInstrItem::If(if_item) => {
        quote! { #if_item }
      }

      ScopeInstrItem::While(while_item) => {
        let cond = &while_item.cond_expr;
        let body = &while_item.body;
        quote! {
          __scope.loop_while(#cond, |__scope| {
            #body
          })
        }
      }
    };

    q.to_tokens(tokens);
  }
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
  semi_token: Token![;],
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
    let semi_token = input.parse()?;

    Ok(Self {
      let_token,
      name,
      ty,
      assign_token,
      expr,
      semi_token,
    })
  }
}

#[derive(Debug)]
pub struct ReturnItem {
  return_token: Option<Token![return]>,
  expr: Expr,
  semi_token: Token![;],
}

impl Parse for ReturnItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let return_token = input.parse()?;
    let expr = input.parse()?;
    let semi_token = input.parse()?;

    Ok(Self {
      return_token,
      expr,
      semi_token,
    })
  }
}

#[derive(Debug)]
pub struct ContinueItem {
  continue_token: Token![continue],
  semi_token: Token![;],
}

impl Parse for ContinueItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let continue_token = input.parse()?;
    let semi_token = input.parse()?;
    Ok(Self {
      continue_token,
      semi_token,
    })
  }
}

#[derive(Debug)]
pub struct BreakItem {
  break_token: Token![break],
  semi_token: Token![;],
}

impl Parse for BreakItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let break_token = input.parse()?;
    let semi_token = input.parse()?;
    Ok(Self {
      break_token,
      semi_token,
    })
  }
}

#[derive(Debug)]
pub struct IfItem {
  if_token: Token![if],
  paren_token: Paren,
  cond_expr: Expr,
  brace_token: Brace,
  body: ScopeInstrItems,
  else_item: Option<ElseItem>,
}

impl IfItem {
  fn to_tokens_as_part_of_else(&self, tokens: &mut proc_macro2::TokenStream) {
    let cond = &self.cond_expr;
    let body = &self.body;

    let q = quote! {
      .or_else(#cond, |__scope| {
        #body
      })
    };

    q.to_tokens(tokens);

    match self.else_item {
      Some(ref else_item) => {
        let else_tail = &else_item.else_tail;
        quote! { #else_tail }.to_tokens(tokens);
      }

      None => {
        quote! { ; }.to_tokens(tokens);
      }
    }
  }
}

impl ToTokens for IfItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let cond = &self.cond_expr;
    let body = &self.body;

    let q = quote! {
    __scope.when(#cond, |__scope| {
      #body
    }) };

    q.to_tokens(tokens);

    match self.else_item {
      Some(ref else_item) => {
        let else_tail = &else_item.else_tail;
        quote! { #else_tail }.to_tokens(tokens);
      }

      None => {
        quote! { ; }.to_tokens(tokens);
      }
    }
  }
}

impl Parse for IfItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let if_token = input.parse()?;

    let cond_input;
    let paren_token = parenthesized!(cond_input in input);
    let cond_expr = cond_input.parse()?;

    let body_input;
    let brace_token = braced!(body_input in input);
    let body = body_input.parse()?;

    let else_item = if input.lookahead1().peek(Token![else]) {
      Some(input.parse()?)
    } else {
      None
    };

    Ok(Self {
      if_token,
      paren_token,
      cond_expr,
      brace_token,
      body,
      else_item,
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
  If(Box<IfItem>),

  Else {
    brace_token: Brace,
    body: ScopeInstrItems,
  },
}

impl ToTokens for ElseTailItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    match self {
      ElseTailItem::If(if_item) => {
        if_item.to_tokens_as_part_of_else(tokens);
      }

      ElseTailItem::Else { body, .. } => {
        let q = quote! {
          .or(|__scope| { #body });
        };

        q.to_tokens(tokens);
      }
    }
  }
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
      let body = body_input.parse()?;

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
  body: ScopeInstrItems,
}

impl Parse for WhileItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let while_token = input.parse()?;

    let cond_input;
    let paren_token = parenthesized!(cond_input in input);
    let cond_expr = cond_input.parse()?;

    let body_input;
    let brace_token = braced!(body_input in input);
    let body = body_input.parse()?;

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
