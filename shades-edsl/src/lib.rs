#![allow(dead_code)] // FIXME: remove before publishing

mod syntax;

use proc_macro::TokenStream;
use quote::ToTokens;
use syn::parse_macro_input;

use crate::syntax::StageDecl;

#[proc_macro]
pub fn shades(tokens: TokenStream) -> TokenStream {
  let mut stage = parse_macro_input!(tokens as StageDecl);
  stage.mutate();
  let ast = stage.into_token_stream();
  eprintln!("{}", ast.to_string());
  ast.into_token_stream().into()
}

#[proc_macro]
pub fn shades_def(tokens: TokenStream) -> TokenStream {
  todo!()
}
