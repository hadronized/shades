//! Shader environments.
//!
//! An environment is a set of expressions available to a shader stage to manipulate values coming from the outside of
//! the stage. It’s akin to system environment (i.e. [`std::env::var`]), but for shaders.
//!
//! Environment variables are typically set by the backend technology, such as OpenGL or Vulkan.

use crate::types::Type;

/// Environment.
///
/// Environment types must provide [`Environment::Env`], which is the type that will be available on the EDSL side. The
/// reason for that is to authorize creating proc-macro to automatically derive [`Environment`] for their types without
/// having to use [`Expr`] types.
///
/// You can use [`()`] if you do not want to use any environment.
pub trait Environment {
  /// Environment type available on the EDSL side.
  type Env;

  /// Get the environment.
  fn env() -> Self::Env;

  /// Environment set.
  ///
  /// This is the list of environment variables available in [`Environment::Env`]. The mapping is a [`String`] to [`Type`].
  /// You can easily get the [`Type`] by using [`ToType::ty`].
  fn env_set() -> Vec<(String, Type)>;
}

impl Environment for () {
  type Env = ();

  fn env() -> Self::Env {
    ()
  }

  fn env_set() -> Vec<(String, Type)> {
    Vec::new()
  }
}
