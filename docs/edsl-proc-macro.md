# EDSL procedural macro

This is a design document trying to explore how to ease building EDSL nodes by leveraging the power of procedural
macros. Especially, this document explores the possibility to annotate and automatically lift expressions and
statement, removing a lot of abstractions that currently exists just to please `rustc`.

<!-- vim-markdown-toc GFM -->

* [Summary](#summary)
* [Context & problems](#context--problems)
* [Solution](#solution)
  * [Automatic scope prefixing](#automatic-scope-prefixing)
  * [Automatic literals lifting and variables transformation](#automatic-literals-lifting-and-variables-transformation)
  * [Sharable blocks](#sharable-blocks)
  * [The EDSL](#the-edsl)

<!-- vim-markdown-toc -->

# Summary

# Context & problems

At the time of writing this document, the design of `shades` works but is fairly difficult to work with. For instance,
creating a new function requires to either use a nightly compiler (to be able to use the nightly implementors of
`FnOnce`, `Fn` and `FnMut`), or to use an ugly `.call((expr0, expr1, …))` syntax. Another problem is the presence of
complex traits, such has `FunBuilder`.

The main big problems are:

- Type ascriptions, which are required pretty much everywhere.
- “Magic” macro to lift literals as expressions, such as in `lit!(1) + 2`.
- Side-effects (setting a variable) requires using a function (`.set()`), because there is no way to override the
  assignment operator in Rust.
- Everything that is a scope must be prefixed with the scope object (most of the time, `s.`), and the scope type must be
  ascripted as well.

# Solution

In order to resolve all the problems from above, we are going to analysis each point and suggest a solution for each of
them.

## Automatic scope prefixing

The current code is problematic:

```rust
  shader.main_fun(|s: &mut Scope<()>| {
    let x = s.var(1.);

    // …
```

Here, we can see that creating the `main` function requires to create an `s: Scope<()>`, with type ascription, and we
have to use that scope whenever we want to do anything with it. Instead, something that would be nicer would be to
completely hide the scope behind the scene, in a monadic way:

```rust
// …
    let x = 1.;
```

## Automatic literals lifting and variables transformation

In the current code, we have a lot of `.clone()` and `.into()` in order to convert from `Var` to `Expr`. On the same
idea, we also have a lot of weird `lit!(x) + y`. We propose to simply completely remove that and replace any literal
value with the following:

```rust
Expr::from(lit)
```

For instance, the following:

```rust
1 + 2
```

will then be replaced by:

```rust
Expr::from(1) + Expr::from(2)
```

Similarly, downcasting a `Var` to an `Expr` will always be done via `Expr::from`. Whenever an expression is expected,
`Expr::from` will be used. So assuming `v` is a `Var`, the following:

```rust
let x = v + 3;
```

will be replaced by:

```rust
let x = Expr::from(&v) + Expr::from(3);
```

## Sharable blocks

One of the main pain points of GLSL is that it’s impossible to share things. For instance, if we want to set `PI` and
use it by importing it, we have to basically copy a string line and paste it whenever it’s needed. It’s a very old way
of doing and it doesn’t scale / is error-prone.

Instead, we are going to support importing symbols inside shader stages. In order to do so, we will hijack `use`
statements.

## The EDSL

The EDSL will be written as a procedural macro, and we will have a couple of them. We want to be able to write a _shader
stage_ using the macro, but we also want to be able to write code that can be `use`d by other pieces of code. So we will
need two proc-macros:

- `shades!`, that will create a shader stage.
- `shades_def!`, that will create a new definition that can be used by other `shades_def!` or `shades!` blocks.
