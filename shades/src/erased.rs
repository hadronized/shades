//! Types that support type erasure.

pub trait Erased {
  type Erased;

  fn erased(&self) -> &Self::Erased;
  fn erased_mut(&mut self) -> &mut Self::Erased;
}
