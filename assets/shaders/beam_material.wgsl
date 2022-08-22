#import bevy_sprite::mesh2d_types
#import bevy_sprite::mesh2d_view_bindings

struct BeamMaterial {
    color: vec4<f32>,
    pattern: u32,
};

@group(1) @binding(0)
var<uniform> material: BeamMaterial;
@group(1) @binding(1)
var texture: texture_2d<f32>;
@group(1) @binding(2)
var texture_sampler: sampler;

@group(2) @binding(0)
var<uniform> mesh: Mesh2d;

struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    #import bevy_sprite::mesh2d_vertex_output
};

@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let i = clamp(u32(in.uv.x * 16.0), 0u, 15u);
    let bit = 1u << i;
    if ((bit & material.pattern) != 0u) {
        return material.color;
    }
    discard;
}