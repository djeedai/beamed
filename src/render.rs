use bevy::{
    prelude::Color,
    reflect::TypeUuid,
    render::render_resource::{AsBindGroup, ShaderRef},
    sprite::Material2d,
};

#[derive(Default, AsBindGroup, TypeUuid, Debug, Clone)]
#[uuid = "050ce6ca-080a-4d8c-b6b5-b5bab7560d8f"]
pub struct BeamMaterial {
    #[uniform(0)]
    pub color: [Color; 16],
    #[uniform(0)]
    pub pattern: u32,
}

impl Material2d for BeamMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/beam_material.wgsl".into()
    }
}
