// Inflated centroid prefix: 2.5x radius for fill/seed passes (matches particle inflation).
// osmId encodes: bits 0-7 = organism ID, bits 8-15 = depth rank.
struct OrganismCentroid {
  pos: vec2<f32>,
  radius: f32,
  osmId: u32,
};

struct CentroidParams {
  resolution: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) osmId: u32,
};

@group(0) @binding(0) var<storage, read> centroids: array<OrganismCentroid>;
@group(0) @binding(1) var<uniform> params: CentroidParams;

const CORNERS = array<vec2<f32>, 6>(
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
  vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corner = CORNERS[vertexIndex];
  let c = centroids[instanceIndex];

  var out: VertexOutput;
  out.uv = corner;
  out.osmId = c.osmId & 0xFFu;

  let depthRank = (c.osmId >> 8u) & 0xFFu;
  let pos = c.pos + corner * c.radius * 2.5;
  var clip = (pos / params.resolution) * 2.0 - 1.0;
  clip.y = -clip.y;
  out.position = vec4<f32>(clip, f32(depthRank) / 255.0, 1.0);

  return out;
}
