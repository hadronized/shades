use std::ops::Deref;

use crate::{
  builtin::{
    BuiltIn, FragmentBuiltIn, GeometryBuiltIn, TessCtrlBuiltIn, TessEvalBuiltIn, VertexBuiltIn,
  },
  expr::ErasedExpr,
  expr::Expr,
  scope::ScopedHandle,
  types::{Type, V4},
  var::Var,
};

/// Outputs.
pub trait Outputs {
  type Out;

  fn output() -> Self::Out;

  fn output_set() -> Vec<(u16, Type)>;
}

impl Outputs for () {
  type Out = ();

  fn output() -> Self::Out {
    ()
  }

  fn output_set() -> Vec<(u16, Type)> {
    Vec::new()
  }
}

/// Vertex shader environment outputs.
#[derive(Debug)]
pub struct VertexShaderOutputs<O> {
  /// 4D position of the vertex.
  pub position: Var<V4<f32>>,

  /// Point size of the vertex.
  pub point_size: Var<f32>,

  // Clip distances to user-defined plans.
  pub clip_distance: Var<[f32]>,

  // User-defined outputs.
  pub user: O,
}

impl<O> VertexShaderOutputs<O> {
  pub(crate) const fn new(user: O) -> Self {
    let position = Var(Expr::new(ErasedExpr::new_builtin(BuiltIn::Vertex(
      VertexBuiltIn::Position,
    ))));
    let point_size = Var(Expr::new(ErasedExpr::new_builtin(BuiltIn::Vertex(
      VertexBuiltIn::PointSize,
    ))));
    let clip_distance = Var(Expr::new(ErasedExpr::new_builtin(BuiltIn::Vertex(
      VertexBuiltIn::ClipDistance,
    ))));

    Self {
      position,
      point_size,
      clip_distance,
      user,
    }
  }
}

impl<O> Deref for VertexShaderOutputs<O> {
  type Target = O;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Tessellation control shader outputs.
#[derive(Debug)]
pub struct TessCtrlShaderOutputs<O> {
  /// Outer tessellation levels.
  pub tess_level_outer: Var<[f32; 4]>,

  /// Inner tessellation levels.
  pub tess_level_inner: Var<[f32; 2]>,

  /// Array of per-vertex output variables.
  pub output: Var<[TessControlPerVertexOut]>,

  pub user: O,
}

impl<O> TessCtrlShaderOutputs<O> {
  pub(crate) const fn new(user: O) -> Self {
    let tess_level_outer = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::TessellationLevelOuter,
    )));
    let tess_level_inner = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::TessellationLevelInner,
    )));
    let output = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::Out,
    )));

    Self {
      tess_level_outer,
      tess_level_inner,
      output,
      user,
    }
  }
}

impl<O> Deref for TessCtrlShaderOutputs<O> {
  type Target = O;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Output tessellation control shader environment.
#[derive(Debug)]
pub struct TessControlPerVertexOut;

impl Expr<TessControlPerVertexOut> {
  /// 4D position of the vertex.
  pub fn position(&self) -> Var<V4<f32>> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::Position,
      ))),
    };

    Var(Expr::new(expr))
  }

  /// Point size of the vertex.
  pub fn point_size(&self) -> Var<f32> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::PointSize,
      ))),
    };

    Var(Expr::new(expr))
  }

  /// Clip distances to user-defined planes.
  pub fn clip_distance(&self) -> Var<[f32]> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::ClipDistance,
      ))),
    };

    Var(Expr::new(expr))
  }

  /// Cull distances to user-defined planes.
  pub fn cull_distance(&self) -> Var<[f32]> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::CullDistance,
      ))),
    };

    Var(Expr::new(expr))
  }
}

/// Tessellation evalution shader outputs.
#[derive(Debug)]
pub struct TessEvalShaderOutputs<O> {
  // outputs
  /// 4D position of the vertex.
  pub position: Var<V4<f32>>,

  /// Point size of the vertex.
  pub point_size: Var<f32>,

  /// Clip distances to user-defined planes.
  pub clip_distance: Var<[f32]>,

  /// Cull distances to user-defined planes.
  pub cull_distance: Var<[f32]>,

  pub user: O,
}

impl<O> TessEvalShaderOutputs<O> {
  pub(crate) const fn new(user: O) -> Self {
    let position = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::Position,
    )));
    let point_size = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::PointSize,
    )));
    let clip_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::ClipDistance,
    )));
    let cull_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::ClipDistance,
    )));

    Self {
      position,
      point_size,
      clip_distance,
      cull_distance,
      user,
    }
  }
}

impl<O> Deref for TessEvalShaderOutputs<O> {
  type Target = O;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Geometry shader outputs.
#[derive(Debug)]
pub struct GeometryShaderOutputs<O> {
  /// Output 4D vertex position.
  pub position: Var<V4<f32>>,

  /// Output vertex point size.
  pub point_size: Var<f32>,

  /// Output clip distances to user-defined planes.
  pub clip_distance: Var<[f32]>,

  /// Output cull distances to user-defined planes.
  pub cull_distance: Var<[f32]>,

  /// Primitive ID to write to in.
  pub primitive_id: Var<i32>,

  /// Layer to write to in.
  pub layer: Var<i32>,

  /// Viewport index to write to.
  pub viewport_index: Var<i32>,

  pub user: O,
}

impl<O> GeometryShaderOutputs<O> {
  pub(crate) const fn new(user: O) -> Self {
    let position = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::Position,
    )));
    let point_size = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::PointSize,
    )));
    let clip_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::ClipDistance,
    )));
    let cull_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::CullDistance,
    )));
    let primitive_id = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::PrimitiveID,
    )));
    let layer = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::Layer,
    )));
    let viewport_index = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::ViewportIndex,
    )));

    Self {
      position,
      point_size,
      clip_distance,
      cull_distance,
      primitive_id,
      layer,
      viewport_index,
      user,
    }
  }
}

impl<O> Deref for GeometryShaderOutputs<O> {
  type Target = O;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Fragment shader outputs.
///
/// This type contains everything you have access to when writing a fragment shader.
#[derive(Debug)]
pub struct FragmentShaderOutputs<O> {
  /// Depth of the fragment.
  pub frag_depth: Var<f32>,

  /// Sample mask of the fragment.
  pub sample_mask: Var<[i32]>,

  pub user: O,
}

impl<O> FragmentShaderOutputs<O> {
  pub(crate) const fn new(user: O) -> Self {
    let frag_depth = Var::new(ScopedHandle::BuiltIn(BuiltIn::Fragment(
      FragmentBuiltIn::FragDepth,
    )));
    let sample_mask = Var::new(ScopedHandle::BuiltIn(BuiltIn::Fragment(
      FragmentBuiltIn::SampleMask,
    )));

    Self {
      frag_depth,
      sample_mask,
      user,
    }
  }
}

impl<O> Deref for FragmentShaderOutputs<O> {
  type Target = O;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn array_lookup() {
    let vertex = VertexShaderOutputs::new(());
    let clip_dist_expr = vertex.clip_distance.at(1);

    assert_eq!(
      clip_dist_expr.erased,
      ErasedExpr::ArrayLookup {
        object: Box::new(vertex.clip_distance.erased.clone()),
        index: Box::new(ErasedExpr::LitInt(1)),
      }
    );
  }
}
