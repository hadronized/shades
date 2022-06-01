use crate::types::Type;

/// Environment.
pub trait Environment {
  type Env;

  fn env() -> Self::Env;

  fn env_set() -> Vec<(usize, Type)>;
}

impl Environment for () {
  type Env = ();

  fn env() -> Self::Env {
    ()
  }

  fn env_set() -> Vec<(usize, Type)> {
    Vec::new()
  }
}
