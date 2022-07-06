use crate::types::Type;

/// Environment.
pub trait Environment {
  type Env;

  fn env() -> Self::Env;

  fn env_set() -> Vec<(u16, Type)>;
}

impl Environment for () {
  type Env = ();

  fn env() -> Self::Env {
    ()
  }

  fn env_set() -> Vec<(u16, Type)> {
    Vec::new()
  }
}
