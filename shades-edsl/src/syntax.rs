use quote::{quote, ToTokens};
use syn::{
  braced, parenthesized,
  parse::Parse,
  parse_quote,
  punctuated::Punctuated,
  token::{Brace, Paren},
  visit_mut::VisitMut,
  BinOp, Expr, Ident, Token, Type,
};

/// A stage declaration with its environment.
#[derive(Debug)]
pub struct StageDecl {
  stage_ty: Type,
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

impl StageDecl {
  pub fn mutate(&mut self) {
    for item in &mut self.stage_item.glob_decl {
      item.mutate();
    }
  }
}

impl ToTokens for StageDecl {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let stage_ty = &self.stage_ty;
    let mut input = self.input.clone();
    let mut output = self.output.clone();
    let mut env = self.env.clone();
    let stage = &self.stage_item;

    let in_ty = input.ty.clone();
    let out_ty = output.ty.clone();
    let env_ty = env.ty.clone();
    input.ty = parse_quote! { <#stage_ty as shades::stage::ShaderModule<#in_ty, #out_ty>>::Inputs };
    output.ty =
      parse_quote! { <#stage_ty as shades::stage::ShaderModule<#in_ty, #out_ty>>::Outputs };
    env.ty = parse_quote! { <#env_ty as shades::env::Environment>::Env };

    let q = quote! {
      shades::stage::ModBuilder::<#stage_ty, #in_ty, #out_ty, #env_ty>::new_stage(|mut __builder, #input, #output, #env| {
        #stage
      })
    };

    q.to_tokens(tokens);
  }
}

impl Parse for StageDecl {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
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

impl StageItem {
  fn mutate(&mut self) {
    for decl in &mut self.glob_decl {
      decl.mutate();
    }
  }
}

impl ToTokens for StageItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let glob_decl = self.glob_decl.iter();
    let q = quote! { #(#glob_decl)* };

    q.to_tokens(tokens);
  }
}

impl Parse for StageItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
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

impl ShaderDeclItem {
  fn mutate(&mut self) {
    match self {
      ShaderDeclItem::Const(const_item) => const_item.mutate(),
      ShaderDeclItem::FunDef(fun_def_item) => fun_def_item.mutate(),
    }
  }
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

impl ConstItem {
  fn mutate(&mut self) {
    ExprVisitor.visit_type_mut(&mut self.ty);
    ExprVisitor.visit_expr_mut(&mut self.expr);
  }
}

impl ToTokens for ConstItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let ident = &self.ident;
    let ty = &self.ty;
    let expr = &self.expr;
    let q = quote! {
      let #ident: #ty = __builder.constant(#expr);
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

impl FnArgItem {
  fn mutate(&mut self) {
    if self.pound_token.is_none() {
      // if we don’t have a #ty but just ty, we lift the type in Expr<#ty>
      ExprVisitor.visit_type_mut(&mut self.ty);
    };
  }
}

impl ToTokens for FnArgItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let ident = &self.ident;
    let ty = &self.ty;
    let q = quote! {
      #ident: #ty
    };

    q.to_tokens(tokens);
  }
}

impl Parse for FnArgItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
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
  ret_ty: Option<(Token![->], Type)>,
  brace_token: Brace,
  body: ScopeInstrItems,
}

impl FunDefItem {
  fn mutate(&mut self) {
    for arg in self.args.iter_mut() {
      arg.mutate();
    }

    self.body.mutate();
  }
}

impl ToTokens for FunDefItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let name = &self.name;
    let body = &self.body;

    if name.to_string() == "main" {
      let q = quote! {
        // scope everything to prevent leaking them out
        {
          // function scope
          let mut __scope: shades::scope::Scope<()> = shades::scope::Scope::new(0);

          // roll down the statements
          #body

          // create the function definition
          use shades::erased::Erased as _;
          let erased = shades::fun::ErasedFun::new(Vec::new(), None, __scope.to_erased());
          let fundef = shades::fun::FunDef::<(), ()>::new(erased);
          __builder.main_fun(fundef)
        }
      };

      q.to_tokens(tokens);
      return;
    }

    let args_ty = self.args.iter().map(|arg| &arg.ty);

    // function argument types
    let fn_args_ty = self.args.iter().map(|arg| {
      let ty = &arg.ty;
      quote! { <#ty as shades::types::ToType>::ty() }
    });

    // function argument expression declarations
    let args_expr_decls = self.args.iter().enumerate().map(|(k, arg)| {
      let ty = &arg.ty;
      let ident = &arg.ident;
      quote! { let #ident: #ty = shades::expr::Expr::new_fun_arg(#k as u16); }
    });

    // function return type
    let ret_ty = &self.ret_ty;
    let (quoted_ret_ty, real_ret_ty) = match ret_ty {
      None => (quote! { None }, quote! { () }),

      Some((_, ret_ty)) => {
        if let Type::Tuple(..) = ret_ty {
          (quote! { None }, quote! { () })
        } else {
          (
            quote! { Some(<#ret_ty as shades::types::ToType>::ty()) },
            quote! { shades::expr::Expr<#ret_ty> },
          )
        }
      }
    };

    let q = quote! {
      // scope everything to prevent leaking them out
      let #name = {
        // function scope
        let mut __scope: shades::scope::Scope<#real_ret_ty> = shades::scope::Scope::new(0);

        // create the function argument expressions so that #body can reference them
        #(#args_expr_decls)*

        // roll down the statements
        #body

        // create the function definition
        use shades::erased::Erased as _;
        let erased = shades::fun::ErasedFun::new(vec![#(#fn_args_ty),*], #quoted_ret_ty, __scope.to_erased());
        let fundef = shades::fun::FunDef::<#real_ret_ty, (#(#args_ty),*)>::new(erased);
        __builder.fun(fundef)
      };
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

    let ret_ty = if input.peek(Token![->]) {
      let arrow_token = input.parse()?;
      let ret_ty = input.parse()?;
      Some((arrow_token, ret_ty))
    } else {
      None
    };

    let body_input;
    let brace_token = braced!(body_input in input);
    let body = body_input.parse()?;

    Ok(Self {
      fn_token,
      name,
      paren_token,
      args,
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

impl ScopeInstrItems {
  fn mutate(&mut self) {
    for item in &mut self.items {
      item.mutate();
    }
  }
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

    while !input.is_empty() {
      let lookahead = input.lookahead1();

      let v = if lookahead.peek(Token![let]) {
        let v = input.parse()?;
        ScopeInstrItem::VarDecl(v)
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
        // try to parse a mutate var first, otherwise fallback on return
        let input_ = input.fork();
        if input_.parse::<MutateVarItem>().is_ok() {
          // advance the original input
          let v = input.parse()?;
          ScopeInstrItem::MutateVar(v)
        } else {
          let v = input.parse()?;
          ScopeInstrItem::Return(v)
        }
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
  MutateVar(MutateVarItem),
}

impl ScopeInstrItem {
  fn mutate(&mut self) {
    match self {
      ScopeInstrItem::VarDecl(var_decl) => var_decl.mutate(),
      ScopeInstrItem::Return(ret) => ret.mutate(),
      ScopeInstrItem::If(if_) => if_.mutate(),
      ScopeInstrItem::While(while_) => while_.mutate(),
      ScopeInstrItem::MutateVar(mutate_var) => mutate_var.mutate(),
      _ => (),
    }
  }
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
            let #name: #ty = __scope.var(#expr);
          }
        } else {
          quote! {
            let #name = __scope.var(#expr);
          }
        }
      }

      ScopeInstrItem::Return(ret) => {
        let expr = ret.expr();

        quote! {
          __scope.leave(#expr);
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

      ScopeInstrItem::MutateVar(mutate_var) => {
        quote! {
          #mutate_var;
        }
      }
    };

    q.to_tokens(tokens);
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

impl VarDeclItem {
  fn mutate(&mut self) {
    if let Some((_, ref mut ty)) = self.ty {
      ExprVisitor.visit_type_mut(ty);
    }

    ExprVisitor.visit_expr_mut(&mut self.expr);
  }
}

impl Parse for VarDeclItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let let_token = input.parse()?;
    let name = input.parse()?;

    let ty = if input.peek(Token![:]) {
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

// FIXME: this encoding is wrong; it should be either an expression, or return + expr + semi
#[derive(Debug)]
pub enum ReturnItem {
  Return {
    return_token: Option<Token![return]>,
    expr: Expr,
    semi_token: Token![;],
  },

  Implicit {
    expr: Expr,
  },
}

impl ReturnItem {
  fn expr(&self) -> &Expr {
    match self {
      ReturnItem::Return { expr, .. } => expr,
      ReturnItem::Implicit { expr } => expr,
    }
  }
  fn mutate(&mut self) {
    match self {
      ReturnItem::Return { expr, .. } => ExprVisitor.visit_expr_mut(expr),
      ReturnItem::Implicit { expr } => ExprVisitor.visit_expr_mut(expr),
    }
  }
}

impl Parse for ReturnItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    if input.peek(Token![return]) {
      let return_token = input.parse()?;
      let expr = input.parse()?;
      let semi_token = input.parse()?;

      Ok(ReturnItem::Return {
        return_token,
        expr,
        semi_token,
      })
    } else {
      let expr = input.parse()?;

      Ok(ReturnItem::Implicit { expr })
    }
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
  fn mutate(&mut self) {
    ExprVisitor.visit_expr_mut(&mut self.cond_expr);

    self.body.mutate();

    if let Some(ref mut else_item) = self.else_item {
      else_item.mutate();
    }
  }
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

    let else_item = if input.peek(Token![else]) {
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

impl ElseItem {
  fn mutate(&mut self) {
    self.else_tail.mutate();
  }
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

impl ElseTailItem {
  fn mutate(&mut self) {
    match self {
      ElseTailItem::If(if_item) => if_item.mutate(),
      ElseTailItem::Else { body, .. } => body.mutate(),
    }
  }
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
    let tail = if input.peek(Token![if]) {
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

impl WhileItem {
  fn mutate(&mut self) {
    ExprVisitor.visit_expr_mut(&mut self.cond_expr);
    self.body.mutate();
  }
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
pub enum MutateVarAssignToken {
  Assign(Token![=]),
  AssignBinOp(BinOp),
}

impl Parse for MutateVarAssignToken {
  fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
    if input.peek(Token![=]) {
      let token = input.parse()?;
      Ok(Self::Assign(token))
    } else {
      let bin_op = input.parse()?;
      Ok(Self::AssignBinOp(bin_op))
    }
  }
}

#[derive(Debug)]
pub struct MutateVarItem {
  ident: Ident,
  assign_token: MutateVarAssignToken,
  expr: Expr,
  semi_token: Token![;],
}

impl MutateVarItem {
  fn mutate(&mut self) {
    ExprVisitor.visit_expr_mut(&mut self.expr);
  }
}

impl ToTokens for MutateVarItem {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let ident = &self.ident;
    let assign_token = &self.assign_token;
    let expr = &self.expr;

    let bin_op = match assign_token {
      MutateVarAssignToken::Assign(_) => quote! { None },
      MutateVarAssignToken::AssignBinOp(bin_op) => match bin_op {
        syn::BinOp::AddEq(_) => quote! { shades::scope::MutateBinOp::Add },
        syn::BinOp::SubEq(_) => quote! { shades::scope::MutateBinOp::Sub },
        syn::BinOp::MulEq(_) => quote! { shades::scope::MutateBinOp::Mul },
        syn::BinOp::DivEq(_) => quote! { shades::scope::MutateBinOp::Div },
        syn::BinOp::RemEq(_) => quote! { shades::scope::MutateBinOp::Rem },
        syn::BinOp::BitXorEq(_) => quote! { shades::scope::MutateBinOp::Xor },
        syn::BinOp::BitAndEq(_) => quote! { shades::scope::MutateBinOp::And },
        syn::BinOp::BitOrEq(_) => quote! { shades::scope::MutateBinOp::Or },
        syn::BinOp::ShlEq(_) => quote! { shades::scope::MutateBinOp::Shl },
        syn::BinOp::ShrEq(_) => quote! { shades::scope::MutateBinOp::Shr },
        _ => quote! { compile_error!("expecting +=, -=, *=, /=, %=, ^=, &=, |=, <<= or >>=") },
      },
    };
    let q = quote! {
      __scope.set(#ident, #bin_op, #expr)
    };

    q.to_tokens(tokens);
  }
}

impl Parse for MutateVarItem {
  fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
    let ident = input.parse()?;
    let assign_token = input.parse()?;
    let expr = input.parse()?;
    let semi_token = input.parse()?;

    Ok(Self {
      ident,
      assign_token,
      expr,
      semi_token,
    })
  }
}

/// A visitor that mutates expressions / literals so that we can wrap them in Expr. If the expression is already in
/// shades::expr::Expr, the expression is left unchanged.
struct ExprVisitor;

impl VisitMut for ExprVisitor {
  fn visit_type_mut(&mut self, ty: &mut Type) {
    *ty = parse_quote! { shades::expr::Expr<#ty> };
  }

  fn visit_expr_mut(&mut self, i: &mut Expr) {
    match i {
      Expr::Array(a) => {
        for expr in &mut a.elems {
          self.visit_expr_mut(expr);
        }
      }

      Expr::Assign(a) => {
        self.visit_expr_mut(&mut a.right);
      }

      Expr::AssignOp(a) => {
        self.visit_expr_mut(&mut a.right);
      }

      Expr::Binary(b) => {
        self.visit_expr_mut(&mut b.left);
        self.visit_expr_mut(&mut b.right);
      }

      Expr::Block(b) => {
        for stmt in &mut b.block.stmts {
          match stmt {
            syn::Stmt::Expr(expr) | syn::Stmt::Semi(expr, _) => {
              self.visit_expr_mut(expr);
            }

            _ => (),
          }
        }
      }

      Expr::Box(e) => {
        self.visit_expr_mut(&mut e.expr);
      }

      Expr::Call(c) => {
        for arg in &mut c.args {
          self.visit_expr_mut(arg);
        }
      }

      Expr::Cast(c) => {
        self.visit_expr_mut(&mut c.expr);
      }

      Expr::Lit(_) => {
        *i = parse_quote! { shades::expr::Expr::from(#i) };
      }

      Expr::MethodCall(c) => {
        for arg in &mut c.args {
          self.visit_expr_mut(arg);
        }
      }

      Expr::Paren(p) => {
        self.visit_expr_mut(&mut p.expr);
      }

      Expr::Return(r) => {
        if let Some(ref mut expr) = r.expr {
          self.visit_expr_mut(expr);
        }
      }

      Expr::Unary(u) => {
        self.visit_expr_mut(&mut u.expr);
      }

      Expr::Path(p) => {
        let e = parse_quote! { #p.clone() };
        *i = e;
      }

      _ => (),
    }
  }
}
