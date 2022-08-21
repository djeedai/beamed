use bevy::{
    app::CoreStage,
    asset::{AssetStage, LoadState},
    gltf::{Gltf, GltfMesh},
    input::gamepad::GamepadButtonType,
    pbr::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
        render_resource::{AddressMode, SamplerDescriptor},
        texture::ImageSampler,
    },
    sprite::Mesh2dHandle,
    window::WindowId,
};
//use bevy_atmosphere::*;
use bevy_kira_audio::{
    Audio as KiraAudio, AudioChannel as KiraAudioChannel, AudioPlugin as KiraAudioPlugin,
    AudioSource as KiraAudioSource, PlaybackState,
};
use bevy_tweening::{lens::*, *};
//use heron::prelude::*;
use leafwing_input_manager::prelude::*;
//use rand::prelude::*;
use std::{f32::consts::PI, time::Duration};

pub struct GamePlugin;

use crate::{menu::AudioManager, AppState};

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ScoreEvent>()
            .register_type::<Cursor>()
            .init_resource::<FixupImages>()
            .init_resource::<AudioRes>()
            // .add_plugin(bevy_atmosphere::AtmospherePlugin {
            //     dynamic: true,
            //     ..default()
            // })
            .add_plugin(InputManagerPlugin::<PlayerAction>::default())
            .add_system_set_to_stage(
                CoreStage::First,
                SystemSet::on_update(AppState::InGame).with_system(fixup_images),
            )
            .add_system_set_to_stage(
                CoreStage::Update,
                SystemSet::on_enter(AppState::InGame).with_system(game_setup),
            )
            .add_system_set_to_stage(
                CoreStage::Update,
                SystemSet::on_update(AppState::InGame).with_system(update_cursor),
            );
    }
}

fn update_cursor(
    board_query: Query<&Board>,
    mut cursor_query: Query<(&mut Transform, &mut Cursor)>,
    input_query: Query<&ActionState<PlayerAction>>,
) {
    let board = board_query.single();
    let size = board.size();
    let half_size = (size - 1) / 2;

    let (mut transform, mut cursor) = cursor_query.single_mut();

    let input_state = input_query.single();
    if input_state.just_pressed(PlayerAction::MoveLeft) {
        cursor.pos.x -= 1;
    }
    if input_state.just_pressed(PlayerAction::MoveRight) {
        cursor.pos.x += 1;
    }
    if input_state.just_pressed(PlayerAction::MoveDown) {
        cursor.pos.y -= 1;
    }
    if input_state.just_pressed(PlayerAction::MoveUp) {
        cursor.pos.y += 1;
    }

    cursor.pos = cursor.pos.clamp(-half_size, half_size);
    //trace!("size={:?} half={:?} pos={:?}", size, half_size, cursor.pos);

    transform.translation =
        (cursor.pos.as_vec2() * board.cell_size).extend(transform.translation.z);
}

#[derive(Actionlike, PartialEq, Eq, Clone, Copy, Hash, Debug)]
enum PlayerAction {
    // Cursor
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    PlaceSelectedItem,
    // Inventory
    SelectNextItem,
    SelectPrevItem,
}

#[derive(Component)]
struct Player;

#[derive(Component, Default)]
struct MainCamera {
    screen_bounds: UiRect<f32>,
}

impl MainCamera {
    pub fn update_screen_bounds(
        &mut self,
        projection: &PerspectiveProjection,
        transform: &Transform,
    ) {
        let camera_half_height = (projection.fov * transform.translation.z * 0.5).abs();
        let camera_half_width = (camera_half_height * projection.aspect_ratio).abs();
        self.screen_bounds.left = -camera_half_width;
        self.screen_bounds.right = camera_half_width;
        self.screen_bounds.bottom = -camera_half_height;
        self.screen_bounds.top = camera_half_height;
        println!(
            "Screen bounds changed: cw/2={} ch/2={} bounds={:?}",
            camera_half_width, camera_half_height, self.screen_bounds
        );
    }
}

#[derive(Component, Default, Debug, Reflect)]
#[reflect(Component)]
struct Cursor {
    pos: IVec2,
}

#[derive(Component, Debug)]
struct Board {
    size: IVec2,
    cell_size: Vec2,
}

impl Board {
    pub fn new(size: IVec2) -> Self {
        Self {
            size,
            cell_size: Vec2::splat(32.),
        }
    }

    pub fn size(&self) -> IVec2 {
        self.size
    }

    /// Create a quad with proper UVs.
    fn create_grid_mesh(&self) -> Mesh {
        let half_size = (self.size - 1) / 2;
        let min = -half_size;
        let max = half_size + 1;

        let offset = Vec2::splat(0.5);
        let vertices = vec![
            ((IVec2::new(min.x, min.y).as_vec2() - offset) * self.cell_size)
                .extend(0.)
                .to_array(),
            ((IVec2::new(max.x, min.y).as_vec2() - offset) * self.cell_size)
                .extend(0.)
                .to_array(),
            ((IVec2::new(min.x, max.y).as_vec2() - offset) * self.cell_size)
                .extend(0.)
                .to_array(),
            ((IVec2::new(max.x, max.y).as_vec2() - offset) * self.cell_size)
                .extend(0.)
                .to_array(),
        ];
        let normals = [[0.; 3]; 4].to_vec();
        let indices = vec![0, 1, 2, 2, 1, 3];
        let uvs = vec![
            IVec2::new(min.x, min.y).as_vec2().to_array(),
            IVec2::new(max.x, min.y).as_vec2().to_array(),
            IVec2::new(min.x, max.y).as_vec2().to_array(),
            IVec2::new(max.x, max.y).as_vec2().to_array(),
        ];

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs));
        mesh.set_indices(Some(Indices::U32(indices)));
        mesh
    }
}

#[derive(Component)]
struct ScoreCounter(u32);

pub struct ScoreEvent(pub u32);

#[derive(Component)]
struct GameOverText;

#[derive(Default)]
pub(crate) struct SfxAudio;

#[derive(Default)]
struct AudioRes {
    sound_move_cursor: Handle<KiraAudioSource>,
}

fn game_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    audio: Res<KiraAudio>,
    sfx_audio: Res<KiraAudioChannel<SfxAudio>>,
    windows: Res<Windows>,
    mut audio_res: ResMut<AudioRes>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut fixup_images: ResMut<FixupImages>,
) {
    println!("game_setup");

    sfx_audio.set_volume(0.5);

    //audio_res.sound_move_cursor = asset_server.load("sounds/move_cursor.ogg");

    // Main camera
    commands
        .spawn_bundle(Camera2dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 1.),
            ..Default::default()
        })
        .insert(MainCamera::default());

    // Input
    let mut input_map = InputMap::default();
    input_map.insert(KeyCode::Up, PlayerAction::MoveUp);
    input_map.insert(KeyCode::W, PlayerAction::MoveUp);
    input_map.insert(GamepadButtonType::DPadUp, PlayerAction::MoveUp);
    input_map.insert(KeyCode::Down, PlayerAction::MoveDown);
    input_map.insert(KeyCode::S, PlayerAction::MoveDown);
    input_map.insert(GamepadButtonType::DPadDown, PlayerAction::MoveDown);
    input_map.insert(KeyCode::Left, PlayerAction::MoveLeft);
    input_map.insert(KeyCode::A, PlayerAction::MoveLeft);
    input_map.insert(GamepadButtonType::DPadDown, PlayerAction::MoveLeft);
    input_map.insert(KeyCode::Right, PlayerAction::MoveRight);
    input_map.insert(KeyCode::D, PlayerAction::MoveRight);
    input_map.insert(GamepadButtonType::DPadDown, PlayerAction::MoveRight);
    input_map.insert(KeyCode::Space, PlayerAction::PlaceSelectedItem);
    input_map.insert(KeyCode::Return, PlayerAction::PlaceSelectedItem);
    #[cfg(not(debug_assertions))] // only in release, otherwise annoying with egui inspector
    input_map.insert(MouseButton::Left, PlayerAction::PlaceSelectedItem);

    // Cursor
    let cursor_image = asset_server.load("textures/cursor.png");
    commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                custom_size: Some(Vec2::splat(32.)),
                ..default()
            },
            texture: cursor_image,
            ..default()
        })
        .insert(Cursor::default())
        .insert(Name::new("cursor"))
        .insert_bundle(InputManagerBundle::<PlayerAction> {
            action_state: ActionState::default(),
            input_map,
        });

    // Board
    let grid_image = asset_server.load("textures/grid.png");
    fixup_images
        .images
        .push((grid_image.clone(), AddressMode::Repeat, AddressMode::Repeat));
    let size = IVec2::new(7, 7);
    let board = Board::new(size);
    let mesh = board.create_grid_mesh();
    let mesh: Mesh2dHandle = meshes.add(mesh).into();
    commands
        .spawn_bundle(ColorMesh2dBundle {
            mesh,
            material: materials.add(ColorMaterial {
                color: Color::WHITE,
                texture: Some(grid_image),
            }),
            ..default()
        })
        .insert(board)
        .insert(Name::new("board"));
}

#[derive(Default)]
struct FixupImages {
    images: Vec<(Handle<Image>, AddressMode, AddressMode)>,
}

fn fixup_images(mut fixup_images: ResMut<FixupImages>, mut images: ResMut<Assets<Image>>) {
    let mut images_temp = vec![];
    std::mem::swap(&mut fixup_images.images, &mut images_temp);
    for (handle, mode_u, mode_v) in &images_temp {
        trace!("fixup: {:?} u={:?} v={:?}", handle, mode_u, mode_v);
        let mut image = images.get_mut(&handle.clone());
        if let Some(image) = image {
            image.sampler_descriptor = ImageSampler::Descriptor(SamplerDescriptor {
                address_mode_u: *mode_u,
                address_mode_v: *mode_v,
                ..default()
            });
            trace!("fixup applied!");
        } else {
            fixup_images.images.push((handle.clone(), *mode_u, *mode_v));
        }
    }
}
