#![allow(dead_code)] // FIXME: remove before publishing

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
  parse::{Parse, ParseStream},
  parse_macro_input, Expr, Ident, Token, Type, UseTree,
};

#[derive(Debug)]
struct TopLevel {
  uses: Vec<Use>,
  constants: Vec<Constant>,
}

impl ToTokens for TopLevel {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    for item in &self.uses {
      item.to_tokens(tokens);
    }

    for item in &self.constants {
      item.to_tokens(tokens);
    }
  }
}

impl Parse for TopLevel {
  fn parse(input: ParseStream) -> Result<Self, syn::Error> {
    let mut uses = Vec::new();
    let mut constants = Vec::new();

    loop {
      let lookahead = input.lookahead1();

      if lookahead.peek(Token![use]) {
        uses.push(input.parse()?);
        let _: Token![;] = input.parse()?;
      } else if lookahead.peek(Token![const]) {
        constants.push(input.parse()?);
        let _: Token![;] = input.parse()?;
      } else {
        break;
      }
    }

    Ok(TopLevel { uses, constants })
  }
}

#[derive(Debug)]
struct Use {
  use_token: Token![use],
  tree: UseTree,
}

impl ToTokens for Use {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let q = quote! {
      // TODO
    };

    q.to_tokens(tokens);
  }
}

impl Parse for Use {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let use_token = input.parse()?;
    let tree = input.parse()?;

    Ok(Use { use_token, tree })
  }
}

#[derive(Debug)]
struct Constant {
  const_token: Token![const],
  ident: Ident,
  colon_token: Token![:],
  ty: Type,
  equal_token: Token![=],
  expr: Expr,
}

impl ToTokens for Constant {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    let ident = &self.ident;
    let ty = &self.ty;
    let expr = &self.expr;
    let q = quote! {
      const #ident: shades::expr::Expr<#ty> = shades::expr::Expr::<#ty>::lit(#expr);
    };

    q.to_tokens(tokens);
  }
}

impl Parse for Constant {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let const_token = input.parse()?;
    let ident = input.parse()?;
    let colon_token = input.parse()?;
    let ty = input.parse()?;
    let equal_token = input.parse()?;
    let expr = input.parse()?;

    Ok(Constant {
      const_token,
      ident,
      colon_token,
      ty,
      equal_token,
      expr,
    })
  }
}

#[proc_macro]
pub fn shades(tokens: TokenStream) -> TokenStream {
  let parsed = parse_macro_input!(tokens as TopLevel);
  parsed.to_token_stream().into()
}
