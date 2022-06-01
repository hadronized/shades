use std::ops::Deref;

use crate::{
  builtin::{BuiltIn, TessCtrlBuiltIn, TessEvalBuiltIn, VertexBuiltIn, GeometryBuiltIn, FragmentBuiltIn},
  types::{V4, V2, V3, Type},
  expr::{ErasedExpr, Expr},
};

/// Inputs.
pub trait Inputs {
  type In;

  fn input() -> Self::In;
  fn input_set() -> Vec<(usize, Type)>;
}

impl Inputs for () {
  type In = ();

  fn input() -> Self::In {
    ()
  }

  fn input_set() -> Vec<(usize, Type)> {
    Vec::new()
  }
}

/// Vertex shader environment inputs.
#[derive(Debug)]
pub struct VertexShaderInputs<I> {
  /// ID of the current vertex.
  pub vertex_id: Expr<i32>,

  /// Instance ID of the current vertex.
  pub instance_id: Expr<i32>,

  /// Base vertex offset.
  pub base_vertex: Expr<i32>,

  /// Base instance vertex offset.
  pub base_instance: Expr<i32>,

  // User-defined inputs.
  pub user: I,
}

impl<I> VertexShaderInputs<I> {
  pub(crate) const fn new(user: I) -> Self {
    let vertex_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Vertex(
      VertexBuiltIn::VertexID,
    )));
    let instance_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Vertex(
      VertexBuiltIn::InstanceID,
    )));
    let base_vertex = Expr::new(ErasedExpr::new_builtin(BuiltIn::Vertex(
      VertexBuiltIn::BaseVertex,
    )));
    let base_instance = Expr::new(ErasedExpr::new_builtin(BuiltIn::Vertex(
      VertexBuiltIn::BaseInstance,
    )));

    Self {
      vertex_id,
      instance_id,
      base_vertex,
      base_instance,
      user,
    }
  }
}

impl<I> Deref for VertexShaderInputs<I> {
  type Target = I;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Tessellation control shader inputs.
#[derive(Debug)]
pub struct TessCtrlShaderInputs<I> {
  /// Maximum number of vertices per patch.
  pub max_patch_vertices_in: Expr<i32>,

  /// Number of vertices for the current patch.
  pub patch_vertices_in: Expr<i32>,

  /// ID of the current primitive.
  pub primitive_id: Expr<i32>,

  /// ID of the current tessellation control shader invocation.
  pub invocation_id: Expr<i32>,

  /// Array of per-vertex input expressions.
  pub input: Expr<[TessControlPerVertexIn]>,

  pub user: I,
}

impl<I> TessCtrlShaderInputs<I> {
  pub(crate) const fn new(user: I) -> Self {
    let max_patch_vertices_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::MaxPatchVerticesIn,
    )));
    let patch_vertices_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::PatchVerticesIn,
    )));
    let primitive_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::PrimitiveID,
    )));
    let invocation_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::InvocationID,
    )));
    let input = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::In,
    )));

    Self {
      max_patch_vertices_in,
      patch_vertices_in,
      primitive_id,
      invocation_id,
      input,
      user,
    }
  }
}

impl<I> Deref for TessCtrlShaderInputs<I> {
  type Target = I;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Read-only, input tessellation control shader environment.
#[derive(Debug)]
pub struct TessControlPerVertexIn;

impl Expr<TessControlPerVertexIn> {
  pub fn position(&self) -> Expr<V4<f32>> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::Position,
      ))),
    };

    Expr::new(erased)
  }

  pub fn point_size(&self) -> Expr<f32> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::PointSize,
      ))),
    };

    Expr::new(erased)
  }

  pub fn clip_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::ClipDistance,
      ))),
    };

    Expr::new(erased)
  }

  pub fn cull_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::CullDistance,
      ))),
    };

    Expr::new(erased)
  }
}

/// Tessellation evalution shader inputs.
#[derive(Debug)]
pub struct TessEvalShaderInputs<I> {
  /// Number of vertices in the current patch.
  pub patch_vertices_in: Expr<i32>,

  /// ID of the current primitive.
  pub primitive_id: Expr<i32>,

  /// Tessellation coordinates of the current vertex.
  pub tess_coord: Expr<V3<f32>>,

  /// Outer tessellation levels.
  pub tess_level_outer: Expr<[f32; 4]>,

  /// Inner tessellation levels.
  pub tess_level_inner: Expr<[f32; 2]>,

  /// Array of per-evertex expressions.
  pub input: Expr<[TessEvaluationPerVertexIn]>,

  pub user: I,
}

impl<I> TessEvalShaderInputs<I> {
  pub(crate) const fn new(user: I) -> Self {
    let patch_vertices_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::PatchVerticesIn,
    )));
    let primitive_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::PrimitiveID,
    )));
    let tess_coord = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::TessCoord,
    )));
    let tess_level_outer = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::TessellationLevelOuter,
    )));
    let tess_level_inner = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::TessellationLevelInner,
    )));
    let input = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::In,
    )));

    Self {
      patch_vertices_in,
      primitive_id,
      tess_coord,
      tess_level_outer,
      tess_level_inner,
      input,
      user,
    }
  }
}

impl<I> Deref for TessEvalShaderInputs<I> {
  type Target = I;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Tessellation evaluation per-vertex expression.
#[derive(Debug)]
pub struct TessEvaluationPerVertexIn;

impl Expr<TessEvaluationPerVertexIn> {
  /// 4D position of the vertex.
  pub fn position(&self) -> Expr<V4<f32>> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::Position,
      ))),
    };

    Expr::new(erased)
  }

  /// Point size of the vertex.
  pub fn point_size(&self) -> Expr<f32> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::PointSize,
      ))),
    };

    Expr::new(erased)
  }

  /// Clip distances to user-defined planes.
  pub fn clip_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::ClipDistance,
      ))),
    };

    Expr::new(erased)
  }

  /// Cull distances to user-defined planes.
  pub fn cull_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::CullDistance,
      ))),
    };

    Expr::new(erased)
  }
}

/// Geometry shader inputs.
#[derive(Debug)]
pub struct GeometryShaderInputs<I> {
  /// Contains the index of the current primitive.
  pub primitive_id_in: Expr<i32>,

  /// ID of the current invocation of the geometry shader.
  pub invocation_id: Expr<i32>,

  /// Read-only environment for each vertices.
  pub input: Expr<[GeometryPerVertexIn]>,

  pub user: I,
}

impl<I> GeometryShaderInputs<I> {
  pub(crate) const fn new(user: I) -> Self {
    let primitive_id_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
      GeometryBuiltIn::PrimitiveIDIn,
    )));
    let invocation_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
      GeometryBuiltIn::InvocationID,
    )));
    let input = Expr::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
      GeometryBuiltIn::In,
    )));

    Self {
      primitive_id_in,
      invocation_id,
      input,
      user,
    }
  }
}

impl<I> Deref for GeometryShaderInputs<I> {
  type Target = I;

  fn deref(&self) -> &Self::Target {
    &self.user
  }
}

/// Read-only, input geometry shader environment.
#[derive(Debug)]
pub struct GeometryPerVertexIn;

impl Expr<GeometryPerVertexIn> {
  /// Provides 4D the position of the vertex.
  pub fn position(&self) -> Expr<V4<f32>> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::Position,
      ))),
    };

    Expr::new(erased)
  }

  /// Provides the size point of the vertex if it’s currently being rendered in point mode.
  pub fn point_size(&self) -> Expr<f32> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::PointSize,
      ))),
    };

    Expr::new(erased)
  }

  /// Clip distances to user planes of the vertex.
  pub fn clip_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::ClipDistance,
      ))),
    };

    Expr::new(erased)
  }

  /// Cull distances to user planes of the vertex.
  pub fn cull_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::CullDistance,
      ))),
    };

    Expr::new(erased)
  }
}

/// Fragment shader inputs.
///
/// This type contains everything you have access to when writing a fragment shader.
#[derive(Debug)]
pub struct FragmentShaderInputs<I> {
  /// Fragment coordinate in the framebuffer.
  pub frag_coord: Expr<V4<f32>>,

  /// Whether the fragment is front-facing.
  pub front_facing: Expr<bool>,

  /// Clip distances to user planes.
  ///
  /// This is an array giving the clip distances to each of the user clip planes.
  pub clip_distance: Expr<[f32]>,

  /// Cull distances to user planes.
  ///
  /// This is an array giving the cull distances to each of the user clip planes.
  pub cull_distance: Expr<[f32]>,

  /// Contains the 2D coordinates of a fragment within a point primitive.
  pub point_coord: Expr<V2<f32>>,

  /// ID of the primitive being currently rendered.
  pub primitive_id: Expr<i32>,

  /// ID of the sample being currently rendered.
  pub sample_id: Expr<i32>,

  /// Sample 2D coordinates.
  pub sample_position: Expr<V2<f32>>,

  /// Contains the computed sample coverage mask for the current fragment.
  pub sample_mask_in: Expr<i32>,

  /// Layer the fragment will be written to.
  pub layer: Expr<i32>,

  /// Viewport index the fragment will be written to.
  pub viewport_index: Expr<i32>,

  /// Indicates whether we are in a helper invocation of a fragment shader.
  pub helper_invocation: Expr<bool>,

  pub user: I
}

impl<I> FragmentShaderInputs<I> {
  pub(crate) const fn new(user: I) -> Self {
    let frag_coord = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::FragCoord,
    )));
    let front_facing = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::FrontFacing,
    )));
    let clip_distance = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::ClipDistance,
    )));
    let cull_distance = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::CullDistance,
    )));
    let point_coord = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::PointCoord,
    )));
    let primitive_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::PrimitiveID,
    )));
    let sample_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::SampleID,
    )));
    let sample_position = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::SamplePosition,
    )));
    let sample_mask_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::SampleMaskIn,
    )));
    let layer = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::Layer,
    )));
    let viewport_index = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::ViewportIndex,
    )));
    let helper_invocation = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::HelperInvocation,
    )));

    Self {
      frag_coord,
      front_facing,
      clip_distance,
      cull_distance,
      point_coord,
      primitive_id,
      sample_id,
      sample_position,
      sample_mask_in,
      layer,
      viewport_index,
      helper_invocation,
      user,
    }
  }
}

impl<I> Deref for FragmentShaderInputs<I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
    &self.user
    }
}

#[cfg(test)]
mod test {
  use crate::lit;
use super::*;

  #[test]
  fn vertex_id_commutative() {
    let vertex = VertexShaderInputs::new(());

    let x = lit!(1);
    let _ = &vertex.vertex_id + &x;
    let _ = x + vertex.vertex_id;
  }

}

