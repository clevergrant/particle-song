// Inflated line prefix: wider fill (20px half-width) to create smooth outline envelope.
struct LineSegment {
  startPos: vec2<f32>,
  endPos: vec2<f32>,
  osmId: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

struct CentroidParams {
  resolution: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) along: f32,
  @location(1) @interpolate(flat) osmId: u32,
};

@group(0) @binding(0) var<storage, read> lines: array<LineSegment>;
@group(0) @binding(1) var<uniform> params: CentroidParams;

const CORNERS = array<vec2<f32>, 6>(
  vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(0.0, 1.0),
  vec2(0.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corner = CORNERS[vertexIndex];
  let seg = lines[instanceIndex];
  let dir = seg.endPos - seg.startPos;
  let len = length(dir);
  let tangent = dir / max(len, 0.001);
  let normal = vec2<f32>(-tangent.y, tangent.x);

  let halfWidth = 20.0; // inflated to match particle fill inflation
  // Extend endpoints by halfWidth for rounded caps
  let pos = seg.startPos - tangent * halfWidth + tangent * corner.x * (len + halfWidth * 2.0) + normal * corner.y * halfWidth;

  var clip = (pos / params.resolution) * 2.0 - 1.0;
  clip.y = -clip.y;

  let depthRank = (seg.osmId >> 8u) & 0xFFu;
  var out: VertexOutput;
  out.position = vec4<f32>(clip, f32(depthRank) / 255.0, 1.0);
  out.along = corner.x;
  out.osmId = seg.osmId & 0xFFu;
  return out;
}
