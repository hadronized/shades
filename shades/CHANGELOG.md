# 0.3.6

> Jul 12, 2021

- Allow to call `Neg` and `Not` on `Var<T>`.
- Add `HasX`, `HasY`, `HasZ` and `HasW` to make it easier to get `VN<_>` values.
- Add `Bounded` to the public interface.
- Fix `lit!` for `V2`.
- Add support for array literals.

# 0.3.5

> Feb 21, 2021

- Change the way `writer::glsl` writers work internally: they now use `std::fmt::Write` instead of using directly
  `String`.
- Add the `shades::writer::glsl::write_shader` that uses `std::fmt::Write` directly.

# 0.3.4

> Feb 20, 2021

- Fix matrices types in the GLSL writer.

# 0.3.3

> Feb 20, 2021

- Fix matrices binary operators.

# 0.3.2

> Feb 20, 2021

- Add matrices.

# 0.3.1

> Feb 19, 2021

- Add the `uniforms!` macro to safely create uniforms.

# 0.3

> Feb 19, 2021

- Add matrices and make `PrimType` a non-exhaustive `enum` so that we can add more types later without breaking the API.

# 0.2

> Feb 15, 2021

- Changed the API to make it easier to use.
- Documented as much things as possible. It’s likely some things are still missing. They will be added, if any, in minor
  bumps of the library.

# 0.1

> Jan 18, 2021

- Initial revision.
