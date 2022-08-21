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
use std::{f32::consts::PI, time::Duration, usize};

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

fn update_cursor(
    board_query: Query<&Board>,
    input_query: Query<&ActionState<PlayerAction>>,
    mut cursor_query: Query<(&mut Transform, &mut Cursor), Without<InventoryCursor>>,
    mut inventory_query: Query<&mut Inventory>,
    mut inventory_cursor_query: Query<(&mut Transform, &mut InventoryCursor), Without<Cursor>>,
    mut cell_query: Query<(&mut Cell, &mut Handle<Image>), Without<Slot>>,
    mut slot_query: Query<(&mut Slot, &mut Handle<Image>), Without<Cell>>,
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

    let mut inventory = inventory_query.single_mut();

    let item_changed = if input_state.just_pressed(PlayerAction::SelectPrevItem) {
        inventory.select_prev()
    } else if input_state.just_pressed(PlayerAction::SelectNextItem) {
        inventory.select_next()
    } else {
        false
    };

    if item_changed {
        let (mut transform, mut cursor) = inventory_cursor_query.single_mut();
        if let Some(index) = inventory.selected_index() {
            transform.translation.x = index as f32 * board.cell_size().x;
        }
    }

    if input_state.just_pressed(PlayerAction::PlaceSelectedItem) {
        trace!("Place item...");
        // Get the cell under the cursor, where the new item goes
        let cell_entity = board.cell_at(cursor.pos);
        if let Ok((mut cell, mut cell_image)) = cell_query.get_mut(cell_entity) {
            // Check the cell is empty (don't overwrite!)
            if cell.item.is_none() {
                // Get the selected inventory slot entity
                if let Some(slot_entity) = inventory.selected() {
                    // Get the actual Slot component for that entity, and its image
                    if let Ok((mut slot, mut slot_image)) = slot_query.get_mut(slot_entity) {
                        // Try to take 1 item from that slot
                        if let Some(item) = slot.try_take(1) {
                            // Success! Place item on board in cell
                            *cell_image = item.image.clone();
                            cell.item = Some(item);
                            // If slot is emtpy, clear its image
                            if slot.is_emtpy() {
                                *slot_image = inventory.empty_slot_image().clone();
                            }
                        } else {
                            debug!(
                                "Slot #{} has no more item.",
                                inventory.selected_index().unwrap_or(0)
                            );
                        }
                    } else {
                        warn!("Failed to find Slot component at Entity {:?}.", slot_entity);
                    }
                } else {
                    debug!("No slot selected in inventory, cannot place item.");
                }
            } else {
                debug!(
                    "Cell at cursor pos {:?} already contains an item, cannot place another one.",
                    cursor.pos
                );
            }
        } else {
            warn!("Failed to find cell at cursor pos {:?}", cursor.pos);
        }
    }
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

#[derive(Component, Debug, Default, Clone)]
struct Cell {
    ipos: IVec2,
    item: Option<Item>,
}

impl Cell {
    pub fn new(ipos: IVec2) -> Self {
        Self { ipos, item: None }
    }
}

#[derive(Component, Debug)]
struct Board {
    size: IVec2,
    cell_size: Vec2,
    cells: Vec<Entity>,
}

impl Board {
    pub fn new(size: IVec2) -> Self {
        let count = size.x as usize * size.y as usize;
        Self {
            size,
            cell_size: Vec2::splat(32.),
            cells: vec![],
        }
    }

    pub fn set_cells(&mut self, cells: Vec<Entity>) {
        assert_eq!(self.size.x as usize * self.size.y as usize, cells.len());
        self.cells = cells;
    }

    /// Iterator over all the positions of the cells on the grid.
    ///
    /// The iterators yields cell positions by row.
    pub fn grid_iter(&self) -> impl Iterator<Item = IVec2> {
        let r = self.rect();
        (r.0.y..r.1.y)
            .flat_map(move |j| (r.0.x..r.1.x).map(move |i| (i, j)))
            .map(|(i, j)| IVec2::new(i, j))
    }

    #[inline]
    pub fn size(&self) -> IVec2 {
        self.size
    }

    #[inline]
    pub fn rect(&self) -> (IVec2, IVec2) {
        // FIXME - works only on odd size...
        let half_size = (self.size + 1) / 2;
        (-half_size + 1, half_size)
    }

    #[inline]
    pub fn cell_size(&self) -> Vec2 {
        self.cell_size
    }

    /// Get the center of the cell located at the specified board position.
    #[inline]
    pub fn to_world(&self, ipos: IVec2) -> Vec2 {
        (ipos.as_vec2() * self.cell_size) - Vec2::splat(0.5)
    }

    /// Get the board cell position corresponding to a world location.
    ///
    /// If the input position lies outside the board, returns `None`.
    pub fn to_board(&self, pos: Vec2) -> Option<IVec2> {
        let pos = (pos + Vec2::splat(0.5)) / self.cell_size;
        let ipos = pos.as_ivec2();
        let r = self.rect();
        if ipos.x >= r.0.x && ipos.x <= r.1.x && ipos.y >= r.0.y && ipos.y <= r.1.y {
            Some(ipos)
        } else {
            None
        }
    }

    /// Get the board cell position corresponding to a world location.
    ///
    /// If the input position lies outside the board, returns the position of the closest cell.
    pub fn to_board_clamped(&self, pos: Vec2) -> IVec2 {
        let pos = (pos + Vec2::splat(0.5)) / self.cell_size;
        let ipos = pos.as_ivec2();
        let r = self.rect();
        ipos.clamp(r.0, r.1)
    }

    #[inline]
    pub fn cells(&self) -> &[Entity] {
        &self.cells
    }

    pub fn try_cell_at(&self, ipos: IVec2) -> Option<Entity> {
        let r = self.rect();
        if ipos.x >= r.0.x && ipos.x <= r.1.x && ipos.y >= r.0.y && ipos.y <= r.1.y {
            let index = (ipos.y - r.0.y) * self.size.x + (ipos.x - r.0.x);
            Some(self.cells[index as usize])
        } else {
            None
        }
    }

    pub fn cell_at(&self, ipos: IVec2) -> Entity {
        let min = self.rect().0;
        let index = (ipos.y - min.y) * self.size.x + (ipos.x - min.x);
        trace!(
            "ipos={:?} min={:?} size={:?} index={}",
            ipos,
            min,
            self.size,
            index
        );
        self.cells[index as usize]
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
    input_map.insert(GamepadButtonType::South, PlayerAction::PlaceSelectedItem);
    input_map.insert(KeyCode::Q, PlayerAction::SelectPrevItem);
    input_map.insert(GamepadButtonType::LeftTrigger, PlayerAction::SelectPrevItem);
    input_map.insert(KeyCode::E, PlayerAction::SelectNextItem);
    input_map.insert(
        GamepadButtonType::RightTrigger,
        PlayerAction::SelectNextItem,
    );
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
            texture: cursor_image.clone(),
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
    let mut board = Board::new(size);

    // Inventory
    let item0 = asset_server.load("textures/halver.png");
    let item1 = asset_server.load("textures/inverter.png");
    let item2 = asset_server.load("textures/into_red.png");
    let mut children = vec![];
    for (index, (maybe_item, count)) in [
        (
            Some(Item {
                image: item0.clone(),
            }),
            3,
        ),
        (
            Some(Item {
                image: item1.clone(),
            }),
            1,
        ),
        (None, 0),
        (
            Some(Item {
                image: item2.clone(),
            }),
            2,
        ),
    ]
    .iter()
    .enumerate()
    {
        let count = if maybe_item.is_none() { 0 } else { *count };
        let pos = Vec3::new(index as f32 * board.cell_size().x, 0., 0.);
        let texture = if let Some(item) = maybe_item {
            item.image.clone()
        } else {
            grid_image.clone()
        };
        children.push(
            commands
                .spawn_bundle(SpriteBundle {
                    transform: Transform::from_translation(pos),
                    sprite: Sprite {
                        custom_size: Some(Vec2::splat(32.)),
                        ..default()
                    },
                    texture,
                    ..default()
                })
                .insert(Slot {
                    item: maybe_item.clone(),
                    count,
                })
                .insert(Name::new(format!("slot#{}", index)))
                .id(),
        );
    }
    let mut inventory = Inventory::new(grid_image.clone());
    inventory.set_slots(children.clone());
    let slot_count = inventory.slots().len();
    let offset = (Vec2::new(-(slot_count as f32) / 2. + 0.5, -(size.y as f32) / 2. - 3.)
        * board.cell_size())
    .extend(0.);
    children.push(
        commands
            .spawn_bundle(SpriteBundle {
                sprite: Sprite {
                    custom_size: Some(Vec2::splat(32.)),
                    ..default()
                },
                texture: cursor_image.clone(),
                ..default()
            })
            .insert(InventoryCursor)
            .id(),
    );
    commands
        .spawn_bundle(SpatialBundle {
            transform: Transform::from_translation(offset),
            ..default()
        }) // needed for children to be visible
        .insert(inventory)
        .insert(Name::new("inventory"))
        .push_children(&children[..]);

    // Board
    let mut children = vec![];
    for ipos in board.grid_iter() {
        let pos = board.to_world(ipos).extend(0.);
        let cell = commands
            .spawn_bundle(SpriteBundle {
                sprite: Sprite {
                    custom_size: Some(Vec2::splat(32.)),
                    ..default()
                },
                texture: grid_image.clone(),
                transform: Transform::from_translation(pos),
                ..default()
            })
            .insert(Cell::new(ipos))
            .insert(Name::new(format!("cell({},{})", ipos.x, ipos.y)))
            .id();
        children.push(cell);
    }
    board.set_cells(children.clone());
    commands
        .spawn_bundle(SpatialBundle::default())
        .insert(board)
        .insert(Name::new("board"))
        .push_children(&children[..]);
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

#[derive(Default, Debug, Clone, PartialEq)]
struct Item {
    image: Handle<Image>,
}

#[derive(Component, Default, Debug)]
struct Slot {
    item: Option<Item>,
    count: usize,
}

impl Slot {
    pub fn try_take(&mut self, count: usize) -> Option<Item> {
        if count <= self.count {
            self.count -= count;
            self.item.clone()
        } else {
            None
        }
    }

    pub fn is_emtpy(&self) -> bool {
        self.count == 0
    }
}

#[derive(Component, Default)]
struct Inventory {
    empty_slot_image: Handle<Image>,
    slots: Vec<Entity>,
    selected_index: usize,
}

impl Inventory {
    pub fn new(empty_slot_image: Handle<Image>) -> Self {
        Self {
            empty_slot_image,
            ..default()
        }
    }

    pub fn empty_slot_image(&self) -> &Handle<Image> {
        &self.empty_slot_image
    }

    pub fn set_slots(&mut self, slots: Vec<Entity>) {
        self.slots = slots;
        let count = self.slots.len();
        self.selected_index = self.selected_index.min(count - 1);
    }

    pub fn slots(&self) -> &[Entity] {
        &self.slots
    }

    pub fn select(&mut self, slot_index: usize) -> bool {
        let count = self.slots.len();
        if slot_index >= count {
            false
        } else if self.selected_index != slot_index {
            self.selected_index = slot_index;
            true
        } else {
            false
        }
    }

    pub fn select_prev(&mut self) -> bool {
        let count = self.slots.len();
        if count == 0 {
            false
        } else {
            let prev = (self.selected_index + count - 1) % count;
            if self.selected_index != prev {
                self.selected_index = prev;
                true
            } else {
                false
            }
        }
    }

    pub fn select_next(&mut self) -> bool {
        let count = self.slots.len();
        if count == 0 {
            false
        } else {
            let next = (self.selected_index + 1) % count;
            if self.selected_index != next {
                self.selected_index = next;
                true
            } else {
                false
            }
        }
    }

    pub fn selected(&self) -> Option<Entity> {
        if self.selected_index < self.slots.len() {
            Some(self.slots[self.selected_index])
        } else {
            None
        }
    }

    pub fn selected_index(&self) -> Option<usize> {
        if self.selected_index < self.slots.len() {
            Some(self.selected_index)
        } else {
            None
        }
    }
}

#[derive(Component, Default, Debug, Reflect)]
#[reflect(Component)]
struct InventoryCursor;
