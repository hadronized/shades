//! Types that support type erasure.
//!
//! Type erasure is important when typing provides the safe abstraction over raw data, and that you need to manipulate
//! the raw data without the safe typing abstraction. It’s typical once you have passed the typing interface
//! (user-side) and that you know that you won’t violate invariants.

/// Associated erased type.
pub trait Erased {
  /// Erased version.
  type Erased;

  /// Consumer the typed interface and return the erased version.
  fn to_erased(self) -> Self::Erased;

  /// Immutable access to the erased version.
  fn erased(&self) -> &Self::Erased;

  /// Mutable access to the erased version.
  fn erased_mut(&mut self) -> &mut Self::Erased;
}
