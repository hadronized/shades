#![allow(dead_code)] // FIXME: remove before publishing

pub mod constant;
use std::collections::HashMap;

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
  parse::{Parse, ParseStream},
  parse_macro_input, parse_quote,
  visit_mut::VisitMut,
  Expr, Ident, Item, ItemFn, Token, Type, UseTree,
};

#[derive(Debug)]
struct TopLevel {
  uses: Vec<Use>,
  constants: Vec<Constant>,
  fns: Vec<Fun>,
}

impl TopLevel {
  fn mutate(&mut self) {
    // TODO: uses
    // TODO: constants

    for fun in &mut self.fns {
      fun.mutate();
    }
  }
}

impl ToTokens for TopLevel {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    for item in &self.uses {
      item.to_tokens(tokens);
    }

    for item in &self.constants {
      item.to_tokens(tokens);
    }

    for item in &self.fns {
      item.to_tokens(tokens);
    }
  }
}

impl Parse for TopLevel {
  fn parse(input: ParseStream) -> Result<Self, syn::Error> {
    let mut uses = Vec::new();
    let mut constants = Vec::new();
    let mut fns = Vec::new();

    loop {
      let lookahead = input.lookahead1();

      if lookahead.peek(Token![use]) {
        uses.push(input.parse()?);
        let _: Token![;] = input.parse()?;
      } else if lookahead.peek(Token![const]) {
        constants.push(input.parse()?);
        let _: Token![;] = input.parse()?;
      } else if lookahead.peek(Token![fn]) {
        fns.push(input.parse()?);
      } else {
        break;
      }
    }

    Ok(TopLevel {
      uses,
      constants,
      fns,
    })
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
struct Fun {
  item_fn: Item,
}

impl Fun {
  fn mutate(&mut self) {
    if let Item::Fn(item_fn) = &self.item_fn {
      let sig = &item_fn.sig;

      // arguments (rank, ident, ty)
      let args = sig
        .inputs
        .iter()
        .map(|arg| match arg {
          syn::FnArg::Receiver(_) => panic!("functions taking self are forbidden"), // TODO

          syn::FnArg::Typed(ty) => {
            let ident = match &*ty.pat {
              syn::Pat::Ident(pat_ident) => pat_ident.ident.clone(),
              _ => parse_quote! { compile_error!("incorrect ident type") },
            };
            let ty = &*ty.ty;

            (ident, ty)
          }
        })
        .enumerate()
        .map(|(rank, (ident, ty))| (rank, ident, ty));

      // ranked idents, which maps a name to its index
      let ranked_idents = args
        .clone()
        .map(|(rank, ident, _)| (ident, rank))
        .collect::<HashMap<_, _>>();

      // list of type arguments required by shades
      let ty_list = args.map(|(_, _, ty)| ty.clone()).collect::<Vec<_>>();

      // return type
      let ret = match sig.output {
        syn::ReturnType::Default => ().into(),
        syn::ReturnType::Type(_, ty) => todo!(),
      };
    }
  }
}

impl Parse for Fun {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let item_fn: ItemFn = input.parse()?;
    Ok(Fun {
      item_fn: Item::Fn(item_fn),
    })
  }
}

impl ToTokens for Fun {
  fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
    self.item_fn.to_tokens(tokens);
  }
}

#[proc_macro]
pub fn shades(tokens: TokenStream) -> TokenStream {
  let mut parsed = parse_macro_input!(tokens as TopLevel);
  parsed.mutate();
  parsed.to_token_stream().into()
}
