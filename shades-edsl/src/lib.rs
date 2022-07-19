mod syntax;

use proc_macro::TokenStream;
use quote::ToTokens;
use syn::parse_macro_input;

use crate::syntax::StageDecl;

#[proc_macro]
pub fn shades(tokens: TokenStream) -> TokenStream {
  let mut stage = parse_macro_input!(tokens as StageDecl);

  stage.mutate();

  stage.into_token_stream().into()
}
