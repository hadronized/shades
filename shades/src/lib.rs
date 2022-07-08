//! Shades, a shading language EDSL in vanilla Rust.
//!
//! This crate provides an [EDSL] to build [shaders], leveraging the Rust compiler (`rustc`) and its type system to ensure
//! soundness and typing. Because shaders are written in Rust, this crate is completely language agnostic: it can in theory
//! target any shading language – the current tier-1 language being [GLSL]. The EDSL allows to statically type shaders
//! while still generating the actual shading code at runtime.
//!
//! # Motivation
//!
//! In typical graphics libraries and engines, shaders are _opaque strings_ – either hard-coded in the program, read from
//! a file at runtime, constructed via fragments of strings concatenated with each others, etc. The strings are passed to
//! the graphics drivers, which will _compile_ and _link_ the code at runtime. It is the responsibility of the runtime
//! (i.e. the graphics library, engine or the application) to check for errors and react correctly. Shading languages can
//! also be compiled _off-line_, and their bytecode is then used at runtime (c.f. SPIR-V).
//!
//! For a lot of people, that has proven okay for decades and even allowed _live coding_: because the shading code is
//! loaded at runtime, it is possible to re-load, re-compile and re-link it every time a change happens. However, this comes
//! with non-negligible drawbacks:
//!
//! - The shading code is often checked at runtime. In this case, ill-written shaders won’t be visible by programmers until
//!   the runtime is executed and the GPU driver refuses the shading code.
//! - When compiled off-line and transpiled to bytecode, extra specialized tooling is required (such as an external program,
//!   a language extension, etc.).
//! - Writing shaders implies learning a new language. The most widespread shading language is [GLSL] but others exist,
//!   meaning that people will have to learn specialized languages and, most of the time, weaker compilation systems. For
//!   instance, [GLSL] doesn’t have anything natively to include other [GLSL] files and it’s an old C-like language.
//! - Even though the appeal of using a language in a dynamic way can seem appealing, going from a dynamic language and
//!   using it in a statically manner is not an easy task. However, going the other way around (from a static to dynamic)
//!   is much much simpler. In other terms: it is possible to live-reload a compiled language with the help of low-level
//!   system primitives, such as `dlopen`, `dlsym`, etc. It’s more work but it’s possible. And
//!   [Rust can do it too](https://crates.io/crates/libloading).
//!
//! The author ([@phaazon]) of this crate thinks that shading code is still code, and that it should be treated as such.
//! It’s easy to see the power of live-coding / reloading, but it’s more important to provide a shading code that is
//! statically proven sound and with less bugs that without the static check. Also, as stated above, using a compiled
//! approach doesn’t prevent from writing a relocatable object, compiled isolated and reloaded at runtime, providing
//! roughly the same functionality as live-coding.
//!
//! Another important point is the choice of using an EDSL. Some people would argue that Rust has other interesting and
//! powerful ways to achieve the same goal. It is important to notice that this crate doesn’t provide a compiler to compile
//! Rust code to a shading language. Instead, it provides a Rust crate that will still generate the shading code at runtime.
//! Several crates following a different approach:
//!
//! - You can use the [glsl](https://crates.io/crates/glsl) and [glsl-quasiquote](https://crates.io/crates/glsl-quasiquote)
//!   crates. The first one is a parser for GLSL and the second one allows you to write GLSL in a quasi-quoter
//!   (`glsl! { /* here */  }`) and get it compiled and checked at runtime. It’s still [GLSL], though, and the
//!   possibilities of runtime combinations are much less than an EDSL. Also, [glsl] doesn’t provide semantic analysis,
//!   so you are left implementing that on your own (and it’s a lot of work).
//! - You can use the [rust-gpu] project. It’s a similar project but they use a `rustc` toolchain, compiling Rust code
//!   representing GPU code. It requires a specific toolchain and doesn’t operate at the same level of this crate — it can
//!   even compile a large part of the `core` library.
//!
//! ## Influences
//!
//! - [blaze-html], a [Haskell] [EDSL] to build HTML in [Haskell].
//! - [selda], a [Haskell] [EDSL] to build SQL queries and execute them without writing SQL strings. This current crate is
//!   very similar in the approach.
//!
//! # The AST crate and the EDSL crate
//!
//! This crate ([shades]) is actually half of the solution. [shades] provides the AST code. Using only [shades] requires
//! you to learn all the types that represent a shading language AST. It can be tedious and counter-productive. For
//! instance, writing a simple loop with [shades] requires several function calls and a weird syntax.
//!
//! Up to version `shades-0.3.6`, you didn’t really have a choice and you had to use that weird syntax. Some macros were
//! available to simplify things, along with nightly extensions to call functions and declare them by hacking around
//! lambdas. However, starting from `shades-0.4`, a new crate appeared: [shades-edsl].
//!
//! [shades-edsl] solves the above problem by removing the “friendly” interface from [shades], reserving [shades]’
//! interface for being called by [shades-edsl] (or adventurous users!). [shades-edsl] is a procedural macro parsing
//! regular Rust code and generating Rust code using [shades]. For instance, a function definition using [shades] only
//! requires you to know about `FunDef`, `ErasedFunDef`, `Return`, `ErasedReturn`, how arguments are represented, etc.
//! It can be very frustrating for people who just want to write the code. With [shades-edsl]? It’s as simple as:
//!
//! ```rust
//! fn add(a: i32, b: i32) -> i32 {
//!   a + b
//! }
//! ```
//!
//! If that function definiton is in a `shades!` block, it will be completely changed to use [shades] instead.
//!
//! # Why you would love this
//!
//! If you like type systems, languages and basically hacking compilers (writing code for your compiler to generate the
//! runtime code!), then it’s likely you’ll like this crate. Among all the features you will find:
//!
//! - Use vanilla Rust. Because this crate is language-agnostic, the whole thing you need to know to get started is to
//!   write Rust. You don’t have to learn [GLSL] to use this crate — even though you still need to understand the concept
//!   of shaders, what they are, how they work, etc. But the _encoding of those concepts_ is now encapsulated by a native
//!   Rust crate.
//! - Types used to represent shading types are basic and native Rust types, such as `bool`, `f32` or `[T; N]`.
//! - Write a more functional code rather than imperative code!
//! - Catch semantic bugs within `rustc`. For instance, assigning a `bool` to a `f32` in your shader code will trigger a
//!   `rustc` error, so that kind of errors won’t leak to your runtime.
//! - Make some code impossible to write. For instance, you will not be able to use in a _vertex shader_ expressions only
//!   valid in the context of a _fragment shader_, as this is not possible by their own definitions.
//! - Extend and add more items to famous shading languages. For instance, [GLSL] doesn’t have a `π` constant. This
//!   situation is fixed so you will never have to write `π` decimals by yourself anymore.
//! - Because you write Rust, benefit from all the language type candies, composability, extensibility and soundness.
//! - Using the proc-macro EDSL, you write shading code without even knowing it, since that crate reinterprets Rust code
//!   into AST nodes. You will not have to care about all the types and functions defined in this crate! You want to
//!   create a function in the shading language? You don’t have to use `FunDef`, simply use a regular `fn` Rust
//!   function. The EDSL does the rest.
//!
//! # Why you wouldn’t love this
//!
//! The crate is, as of nowadays, still very experimental. Here’s a list of things you might dislike about the crate:
//!
//! - Some people would argue that writing [GLSL] is much faster and simpler to write, and they would be partially right.
//!   However, you would need to learn [GLSL] in the first place; you wouldn’t be able to target SPIR-V; you wouldn’t have
//!   a solution to the static typing problem; etc.
//! - In the case of a runtime compilation / linking failure of your shading code, debugging it might be challenging, as
//!   all the identifiers (with a few exceptions) are generated for you. It’ll make it harder to understand the generated
//!   code.
//!
//! [@phaazon]: https://github.com/phaazon
//! [EDSL]: https://en.wikipedia.org/wiki/Domain-specific_language#External_and_Embedded_Domain_Specific_Languages
//! [shaders]: https://en.wikipedia.org/wiki/Shader
//! [GLSL]: https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf
//! [Haskell]: https://www.haskell.org
//! [blaze-html]: http://hackage.haskell.org/package/blaze-html
//! [selda]: http://hackage.haskell.org/package/selda
//! [proc-macro]: https://doc.rust-lang.org/reference/procedural-macros.html
//! [rust-gpu]: https://github.com/EmbarkStudios/rust-gpu
//! [do-notation]: https://crates.io/crates/do-notation
//! [shades]: https://crates.io/crates/shades
//! [shades-edsl]: https://crates.io/crates/shades-edsl

pub mod builtin;
pub mod env;
pub mod erased;
pub mod expr;
pub mod fun;
pub mod input;
pub mod output;
pub mod scope;
pub mod shader;
pub mod stage;
pub mod stdlib;
pub mod swizzle;
pub mod types;
pub mod var;
pub mod writer;
