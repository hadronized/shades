#![allow(dead_code)] // FIXME: remove before publishing

mod syntax;

use proc_macro::TokenStream;
use syn::parse_macro_input;

use crate::syntax::StageDecl;

#[proc_macro]
pub fn shades(tokens: TokenStream) -> TokenStream {
  let stage = parse_macro_input!(tokens as StageDecl);
  panic!("{:#?}", stage);
}

#[proc_macro]
pub fn shades_def(tokens: TokenStream) -> TokenStream {
  todo!()
}
