use bevy::{
    app::CoreStage,
    asset::{AssetStage, LoadState},
    ecs::system::Resource,
    gltf::{Gltf, GltfMesh},
    input::gamepad::GamepadButtonType,
    pbr::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
        render_resource::{AddressMode, SamplerDescriptor},
        texture::ImageSampler,
    },
    sprite::{Material2dPlugin, MaterialMesh2dBundle, Mesh2dHandle},
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

use crate::{menu::AudioManager, render::BeamMaterial, AppState};

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(Material2dPlugin::<BeamMaterial>::default())
            .add_event::<ScoreEvent>()
            .add_event::<PlaceItemEvent>()
            .register_type::<Cursor>()
            .init_resource::<FixupImages>()
            .init_resource::<AudioRes>()
            .init_resource::<ItemDatabase>()
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
                SystemSet::on_update(AppState::InGame)
                    .with_system(update_cursor)
                    .with_system(update_board.after(update_cursor)),
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
    TurnItemLeft,
    TurnItemRight,
    PlaceSelectedItem,
    // Inventory
    SelectNextItem,
    SelectPrevItem,
}

#[derive(Debug)]
struct PlaceItemEvent {
    item_id: ItemId,
    ipos: IVec2,
    orient: Orient,
}

fn update_cursor(
    board_query: Query<&Board>,
    input_query: Query<&ActionState<PlayerAction>>,
    mut cursor_query: Query<
        (&mut Transform, &mut Cursor, &mut Animator<Transform>),
        (Without<InventoryCursor>, Without<Cell>),
    >,
    mut inventory_query: Query<&mut Inventory>,
    mut inventory_cursor_query: Query<
        (&mut Transform, &mut InventoryCursor),
        (Without<Cursor>, Without<Cell>),
    >,
    cell_query: Query<&Cell>,
    mut slot_query: Query<(&mut Slot, &mut Handle<Image>), Without<Cell>>,
    database: Res<ItemDatabase>,
    mut place_item_event_writer: EventWriter<PlaceItemEvent>,
    //
    mut meshes: ResMut<Assets<Mesh>>,
    mut beam_materials: ResMut<Assets<BeamMaterial>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut commands: Commands,
) {
    let board = board_query.single();
    let size = board.size();
    let half_size = (size - 1) / 2;

    let (mut transform, mut cursor, mut animator) = cursor_query.single_mut();

    let input_state = input_query.single();

    let mut pos = cursor.pos;
    if input_state.just_pressed(PlayerAction::MoveLeft) {
        pos.x -= 1;
    }
    if input_state.just_pressed(PlayerAction::MoveRight) {
        pos.x += 1;
    }
    if input_state.just_pressed(PlayerAction::MoveDown) {
        pos.y -= 1;
    }
    if input_state.just_pressed(PlayerAction::MoveUp) {
        pos.y += 1;
    }
    if input_state.just_pressed(PlayerAction::TurnItemLeft) {
        cursor.turn_to(transform.rotation, -1, &mut *animator);
    }
    if input_state.just_pressed(PlayerAction::TurnItemRight) {
        cursor.turn_to(transform.rotation, 1, &mut *animator);
    }

    pos = pos.clamp(-half_size, half_size);
    //trace!("size={:?} half={:?} pos={:?}", size, half_size, cursor.pos);

    if pos != cursor.pos {
        cursor.move_to(transform.translation, pos, board.cell_size, &mut *animator);
    }

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
        if let Ok(cell) = cell_query.get(cell_entity) {
            // Check the cell is empty (don't overwrite!)
            if cell.item.is_none() {
                // Get the selected inventory slot entity
                if let Some(slot_entity) = inventory.selected() {
                    // Get the actual Slot component for that entity, and its image
                    if let Ok((mut slot, mut slot_image)) = slot_query.get_mut(slot_entity) {
                        // Try to take 1 item from that slot
                        if let Some(item_id) = slot.try_take(1) {
                            // Send event to board to place a new item
                            place_item_event_writer.send(PlaceItemEvent {
                                item_id,
                                ipos: cursor.pos,
                                orient: cursor.orient,
                            });

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
                    "Cell at pos {:?} already contains an item, cannot place another one.",
                    cursor.pos
                );
            }
        } else {
            warn!("Failed to find cell at pos {:?}", cursor.pos);
        }
    }
}

/// Update the Board, adding/removing items and ticking all.
fn update_board(
    database: Res<ItemDatabase>,
    mut board_query: Query<&mut Board>,
    mut cell_query: Query<(&mut Cell, &mut Handle<Image>, &mut Transform)>,
    mut place_item_event_reader: EventReader<PlaceItemEvent>,
) {
    let mut board = board_query.single_mut();
    let size = board.size();
    let half_size = (size - 1) / 2;

    // Place new items on board
    for ev in place_item_event_reader.iter() {
        // Get the cell under the cursor, where the new item goes
        let cell_entity = board.cell_at(ev.ipos);
        trace!(
            "PlaceItem: ipos={:?} cell_entity={:?}",
            ev.ipos,
            cell_entity
        );
        if let Ok((mut cell, mut cell_image, mut cell_transform)) = cell_query.get_mut(cell_entity)
        {
            // Check the cell is empty (don't overwrite!)
            if cell.item.is_none() {
                // Success!
                let item = database.get(ev.item_id);

                // Update cell, for graphics
                *cell_image = item.image.clone();
                cell_transform.rotation = ev.orient.into();
                cell.item = Some(ev.item_id);

                // Update board, for logic
                trace!(
                    "board.add() ipos={:?} orient={:?} item_id={:?}",
                    ev.ipos,
                    ev.orient,
                    ev.item_id
                );
                board.add(ev.ipos, ev.orient, cell_entity, ev.item_id, item);

                FIXME - instead of trying to directly connect inputs, let's break down existing
                beams that intersect the position of the new tile, and conect inputs that way.
                Then we can connect outputs with raycast as below.

                // Connect inputs of new item
                for input in &item.inputs {
                    trace!("Input: port={:?}", input.port);
                    match input.port {
                        Port::Single(in_orient) => {
                            let global_in_orient = ev.orient + in_orient;
                            trace!(
                                "+ item_orient={:?} local_orient={:?} global_orient={:?}",
                                ev.orient,
                                in_orient,
                                global_in_orient
                            );

                            if let Some((out_ipos, out_item_orient, out_item_id)) =
                                board.find(ev.ipos, global_in_orient)
                            {
                                trace!(
                                    "Found item: ipos={:?} orient={:?} id={:?}",
                                    out_ipos,
                                    out_item_orient,
                                    out_item_id
                                );

                                let out_item = database.get(out_item_id);

                                // Calculate the orientation of the output, which is the opposite
                                // of the orientation of the search.
                                let global_out_orient = global_in_orient.reversed();
                                trace!("global_out_orient = {:?}", global_out_orient);

                                // Calculate the orientation locally for the output port/item
                                let local_out_orient = global_out_orient - out_item_orient;
                                trace!("local_out_orient = {:?}", local_out_orient);

                                // Check if the item has an output to connect to
                                for output in &out_item.outputs {
                                    trace!("+ output = {:?}", output.port);

                                    if output.port.can_connect_from(local_out_orient) {
                                        trace!("Connect!");

                                        //board.connect();

                                        // let mesh: Mesh2dHandle = meshes
                                        //     .add(shape::Quad::new(Vec2::new(140., 4.)).into())
                                        //     .into();
                                        // commands.spawn_bundle(MaterialMesh2dBundle {
                                        //     mesh,
                                        //     material: beam_materials.add(BeamMaterial {
                                        //         color: Color::PURPLE,
                                        //         pattern: 0xAAF0,
                                        //     }),
                                        //     //material: color_materials.add(Color::PURPLE.into()),
                                        //     ..default()
                                        // });
                                    }
                                }

                                // Beams cannot cross an item, so stop search here
                                break;
                            }
                        }
                        Port::PassThrough(pto) => {}
                        Port::Any => {}
                    }
                }
            } else {
                debug!(
                    "Cell at pos {:?} already contains an item, cannot place another one.",
                    ev.ipos
                );
            }
        } else {
            warn!("Failed to find cell at pos {:?}", ev.ipos);
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
    orient: Orient,
}

impl Cursor {
    pub fn move_to(
        &mut self,
        start: Vec3,
        end: IVec2,
        cell_size: Vec2,
        animator: &mut Animator<Transform>,
    ) {
        // Move authoritative position
        self.pos = end;

        // Animate visual position
        let lens = TransformPositionLens {
            start,
            end: (end.as_vec2() * cell_size).extend(start.z),
        };
        animator.set_tweenable(Tween::new(
            EaseFunction::QuadraticInOut,
            TweeningType::Once,
            Duration::from_millis(200),
            lens,
        ));
    }

    pub fn turn_to(&mut self, start: Quat, dir: i32, animator: &mut Animator<Transform>) {
        if dir < 0 {
            self.orient.turn_right();
        } else if dir > 0 {
            self.orient.turn_left();
        }
        let lens = TransformRotationLens {
            start,
            end: self.orient.into(),
        };
        animator.set_tweenable(Tween::new(
            EaseFunction::QuadraticInOut,
            TweeningType::Once,
            Duration::from_millis(200),
            lens,
        ));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect, Default)]
enum Orient {
    #[default]
    Top = 0,
    Right = 1,
    Bottom = 2,
    Left = 3,
}

impl Orient {
    pub fn to_angle(&self) -> f32 {
        match self {
            Orient::Top => 0.,
            Orient::Right => PI / 2.,
            Orient::Bottom => PI,
            Orient::Left => 3. * PI / 2.,
        }
    }

    pub fn to_dir(&self) -> IVec2 {
        match self {
            Orient::Top => IVec2::Y,
            Orient::Right => IVec2::X,
            Orient::Bottom => IVec2::NEG_Y,
            Orient::Left => IVec2::NEG_X,
        }
    }

    pub fn turn_left(&mut self) {
        *self = match self {
            Orient::Top => Orient::Left,
            Orient::Right => Orient::Top,
            Orient::Bottom => Orient::Right,
            Orient::Left => Orient::Bottom,
        };
    }

    pub fn turn_right(&mut self) {
        *self = match self {
            Orient::Top => Orient::Right,
            Orient::Right => Orient::Bottom,
            Orient::Bottom => Orient::Left,
            Orient::Left => Orient::Top,
        };
    }

    pub fn reversed(&self) -> Orient {
        match self {
            Orient::Top => Orient::Bottom,
            Orient::Right => Orient::Left,
            Orient::Bottom => Orient::Top,
            Orient::Left => Orient::Right,
        }
    }

    fn from_raw(value: i32) -> Orient {
        match value {
            0 => Orient::Top,
            1 => Orient::Right,
            2 => Orient::Bottom,
            3 => Orient::Left,
            _ => panic!(),
        }
    }
}

impl From<Orient> for Quat {
    fn from(orient: Orient) -> Quat {
        Quat::from_rotation_z(orient.to_angle())
    }
}

impl std::ops::Add for Orient {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let value = ((self as i32) + (rhs as i32)) % 4;
        Orient::from_raw(value)
    }
}

impl std::ops::Sub for Orient {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let value = ((self as i32) + 4 - (rhs as i32)) % 4;
        Orient::from_raw(value)
    }
}

#[derive(Component, Debug, Default, Clone)]
struct Cell {
    ipos: IVec2,
    item: Option<ItemId>,
    orient: Orient,
}

impl Cell {
    pub fn new(ipos: IVec2) -> Self {
        Self { ipos, ..default() }
    }
}

#[derive(Component, Debug)]
struct Board {
    size: IVec2,
    cell_size: Vec2,
    /// Entity for drawning the cell.
    cells: Vec<Entity>,
    /// Entity for an item, if any.
    items: Vec<Option<(Orient, Entity, ItemId)>>,
}

impl Board {
    pub fn new(size: IVec2) -> Self {
        let count = size.x as usize * size.y as usize;
        Self {
            size,
            cell_size: Vec2::splat(32.),
            cells: vec![], // TEMP; until set_cells() called
            items: vec![None; count],
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

    fn index(&self, ipos: IVec2) -> usize {
        let min = self.rect().0;
        ((ipos.y - min.y) * self.size.x + (ipos.x - min.x)) as usize
    }

    pub fn cell_at(&self, ipos: IVec2) -> Entity {
        let index = self.index(ipos);
        trace!("ipos={:?} size={:?} index={}", ipos, self.size, index);
        self.cells[index]
    }

    /// Add a new instance of the given item onto the board.
    pub fn add(
        &mut self,
        ipos: IVec2,
        orient: Orient,
        entity: Entity,
        item_id: ItemId,
        _item: &Item,
    ) {
        let index = self.index(ipos);
        self.items[index] = Some((orient, entity, item_id));
    }

    pub fn try_get(&self, ipos: IVec2) -> GetBoard {
        let r = self.rect();
        if ipos.x >= r.0.x && ipos.x <= r.1.x && ipos.y >= r.0.y && ipos.y <= r.1.y {
            let index = self.index(ipos);
            if let Some(item_tuple) = self.items[index] {
                GetBoard::Item(item_tuple)
            } else {
                GetBoard::Empty
            }
        } else {
            GetBoard::Outside
        }
    }

    /// Find a tile search from an existing starting tile alongside the given orientation.
    pub fn find(&self, start: IVec2, orient: Orient) -> Option<(IVec2, Orient, ItemId)> {
        let r = self.rect();
        let index = self.index(start);
        let (local_orient, _, _) = self.items[index].unwrap();

        // Get the world-space orientation of the search from its local orientation
        // and the orientation of the start tile itself.
        let global_orient = local_orient + orient;
        let dir = orient.to_dir();

        // Check all tiles in the given direction until a connection is made or
        // the border of the board is reached without finding one.
        let mut ipos = start + dir;
        loop {
            match self.try_get(ipos) {
                // No tile at this position, continue checking next one in same direction
                GetBoard::Empty => ipos += dir,
                // Reached the border of the board, found nothing
                GetBoard::Outside => return None,
                // Reached an item
                GetBoard::Item((local_orient, entity, item_id)) => {
                    return Some((ipos, local_orient, item_id))
                }
            }
        }
    }
}

enum GetBoard {
    Empty,
    Outside,
    Item((Orient, Entity, ItemId)),
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
    mut database: ResMut<ItemDatabase>,
) {
    println!("game_setup");

    sfx_audio.set_volume(0.5);

    //audio_res.sound_move_cursor = asset_server.load("sounds/move_cursor.ogg");

    // Populate item database
    for (id, name, inputs, outputs, gate) in [
        (
            "halver",
            "Halver",
            vec![InputPort {
                port: PassThroughOrient::Horizontal.into(),
            }],
            vec![OutputPort {
                port: PassThroughOrient::Horizontal.into(),
            }],
            Box::new(BitManipulator::new(0xF0F0, BitOp::None)) as Box<dyn Gate>,
        ),
        (
            "inverter",
            "Inverter",
            vec![InputPort {
                port: PassThroughOrient::Horizontal.into(),
            }],
            vec![OutputPort {
                port: PassThroughOrient::Horizontal.into(),
            }],
            Box::new(BitManipulator::new(0xFFFF, BitOp::Not)),
        ),
        (
            "filter_red",
            "Filter",
            vec![InputPort {
                port: PassThroughOrient::Any.into(),
            }],
            vec![OutputPort {
                port: PassThroughOrient::Any.into(),
            }],
            Box::new(Filter::new(BitColor::Red)),
        ),
        (
            "emit",
            "Emitter",
            vec![],
            vec![OutputPort {
                port: Orient::Top.into(),
            }],
            Box::new(Emitter::new(BitColor::Red, 1)),
        ),
        // (
        //     "multi_emit",
        //     "Multi-Emitter",
        //     vec![],
        //     vec![OutputPort {
        //         port: Port::Any,
        //     }],
        // ),
    ] {
        let path = format!("textures/{}.png", id);
        let image = asset_server.load(&path);
        database.add(name, image, inputs, outputs, gate);
    }

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
    input_map.insert(KeyCode::Q, PlayerAction::TurnItemLeft);
    input_map.insert(GamepadButtonType::DPadLeft, PlayerAction::TurnItemLeft);
    input_map.insert(KeyCode::E, PlayerAction::TurnItemRight);
    input_map.insert(GamepadButtonType::DPadRight, PlayerAction::TurnItemRight);
    input_map.insert(KeyCode::Space, PlayerAction::PlaceSelectedItem);
    input_map.insert(KeyCode::Return, PlayerAction::PlaceSelectedItem);
    input_map.insert(GamepadButtonType::South, PlayerAction::PlaceSelectedItem);
    input_map.insert(
        UserInput::chord([KeyCode::Tab, KeyCode::LShift]),
        PlayerAction::SelectPrevItem,
    );
    input_map.insert(GamepadButtonType::LeftTrigger, PlayerAction::SelectPrevItem);
    input_map.insert(KeyCode::Tab, PlayerAction::SelectNextItem);
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
        .insert(Animator::<Transform>::default())
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
    let mut children = vec![];
    for (index, (maybe_item, count)) in [
        (Some(ItemId(0)), 3),
        (Some(ItemId(1)), 1),
        (None, 0),
        (Some(ItemId(2)), 2),
        (Some(ItemId(3)), 1),
    ]
    .iter()
    .enumerate()
    {
        let count = if maybe_item.is_none() { 0 } else { *count };
        let pos = Vec3::new(index as f32 * board.cell_size().x, 0., 0.);
        let texture = if let Some(item_id) = maybe_item {
            database.get(*item_id).image.clone()
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
        .spawn_bundle(SpatialBundle::default()) // needed for children to be visible
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

trait Gate: Send + Sync + 'static {
    fn tick(&mut self, inputs: &[InputBeam], outputs: &mut [OutputBeam]);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassThroughOrient {
    Horizontal,
    Vertical,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Port {
    Single(Orient),
    PassThrough(PassThroughOrient),
    Any,
}

impl Port {
    pub fn can_connect_from(&self, orient: Orient) -> bool {
        match *self {
            Port::Single(or) => or == orient,
            Port::PassThrough(pto) => {
                (pto == PassThroughOrient::Horizontal
                    && (orient == Orient::Left || orient == Orient::Right))
                    || (pto == PassThroughOrient::Vertical
                        && (orient == Orient::Top || orient == Orient::Bottom))
                    || (pto == PassThroughOrient::Any)
            }
            Port::Any => true,
        }
    }
}

impl From<Orient> for Port {
    fn from(orient: Orient) -> Port {
        Port::Single(orient)
    }
}

impl From<PassThroughOrient> for Port {
    fn from(pto: PassThroughOrient) -> Port {
        Port::PassThrough(pto)
    }
}

#[derive(Debug, Clone, Copy)]
struct InputPort {
    port: Port,
}

#[derive(Debug, Clone, Copy)]
struct OutputPort {
    port: Port,
}

#[derive(Debug, Clone, Copy)]
struct InputBeam<'a> {
    port: &'a InputPort,
    pattern: Option<BitPattern>,
}

#[derive(Debug, Clone, Copy)]
struct OutputBeam<'a> {
    port: &'a OutputPort,
    pattern: Option<BitPattern>,
}

struct Item {
    name: String,
    image: Handle<Image>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    gate: Box<dyn Gate>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct ItemId(u32);

struct ItemDatabase {
    items: Vec<Item>,
}

impl Default for ItemDatabase {
    fn default() -> Self {
        ItemDatabase { items: vec![] }
    }
}

impl ItemDatabase {
    pub fn add(
        &mut self,
        name: &str,
        image: Handle<Image>,
        inputs: Vec<InputPort>,
        outputs: Vec<OutputPort>,
        gate: Box<dyn Gate>,
    ) -> ItemId {
        let id = ItemId(self.items.len() as u32);
        self.items.push(Item {
            name: name.to_string(),
            image,
            inputs,
            outputs,
            gate,
        });
        id
    }

    pub fn get(&self, id: ItemId) -> &Item {
        let index = id.0 as usize;
        &self.items[index]
    }

    pub fn try_get(&self, id: ItemId) -> Option<&Item> {
        let index = id.0 as usize;
        if index < self.items.len() {
            Some(&self.items[index])
        } else {
            None
        }
    }
}

#[derive(Component, Default, Debug)]
struct Slot {
    item: Option<ItemId>,
    count: usize,
}

impl Slot {
    pub fn try_take(&mut self, count: usize) -> Option<ItemId> {
        if count <= self.count {
            self.count -= count;
            self.item
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum BitColor {
    #[default]
    White,
    Red,
    Yellow,
    Green,
    Blue,
    Violet,
}

impl BitColor {
    #[inline]
    pub fn new(raw: u8) -> Self {
        match raw {
            0 => BitColor::White,
            1 => BitColor::Red,
            2 => BitColor::Yellow,
            3 => BitColor::Green,
            4 => BitColor::Blue,
            5 => BitColor::Violet,
            _ => panic!("Invalid BitColor raw value {}", raw),
        }
    }

    #[inline]
    pub fn raw(&self) -> u8 {
        match &self {
            BitColor::White => 0,
            BitColor::Red => 1,
            BitColor::Yellow => 2,
            BitColor::Green => 3,
            BitColor::Blue => 4,
            BitColor::Violet => 5,
        }
    }
}

impl From<BitColor> for Color {
    fn from(bc: BitColor) -> Color {
        match bc {
            BitColor::White => Color::WHITE,
            BitColor::Red => Color::RED,
            BitColor::Yellow => Color::YELLOW,
            BitColor::Green => Color::GREEN,
            BitColor::Blue => Color::BLUE,
            BitColor::Violet => Color::VIOLET,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct Bit {
    value: u8,
}

impl Bit {
    #[inline]
    pub fn new(color: BitColor, thickness: u8) -> Self {
        Self {
            value: (color.raw() & 0xF) | ((thickness & 0xF) << 4),
        }
    }

    #[inline]
    pub fn color(&self) -> BitColor {
        BitColor::new(self.value & 0xF)
    }

    #[inline]
    pub fn set_color(&mut self, color: BitColor) {
        self.value |= color.raw();
    }

    #[inline]
    pub fn thickness(&self) -> u8 {
        (self.value & 0xF0) >> 4
    }

    #[inline]
    pub fn set_thickness(&mut self, thickness: u8) {
        self.value |= (thickness & 0xF) << 4;
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct BitPattern {
    pattern: u16,
    colors: [BitColor; 16],
    thicknesses: [u8; 16],
}

impl BitPattern {
    /// Create a simple pattern of constant color and thickness.
    pub fn simple(color: BitColor, pattern: u16, thickness: u8) -> Self {
        Self {
            pattern,
            colors: [color; 16],
            thicknesses: [thickness; 16],
        }
    }

    pub fn monochrome(&self) -> Option<BitColor> {
        let mut ret: Option<BitColor> = None;
        for i in 0..16 {
            if self.pattern & (1u16 << i) != 0 {
                let color = self.colors[i];
                if let Some(prev_color) = &ret {
                    if *prev_color != color {
                        return None;
                    }
                } else {
                    ret = Some(color);
                }
            }
        }
        ret
    }

    pub fn thickness(&self) -> Option<u8> {
        let mut ret: Option<u8> = None;
        for i in 0..16 {
            if self.pattern & (1u16 << i) != 0 {
                let thickness = self.thicknesses[i];
                if let Some(prev_thickness) = &ret {
                    if *prev_thickness != thickness {
                        return None;
                    }
                } else {
                    ret = Some(thickness);
                }
            }
        }
        ret
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum BitOp {
    #[default]
    None,
    Not,
    And,
    Or,
    Xor,
}

#[derive(Component, Default, Debug)]
struct Emitter {
    bit: Bit,
}

impl Emitter {
    pub fn new(color: BitColor, thickness: u8) -> Self {
        Self {
            bit: Bit::new(color, thickness),
        }
    }
}

impl Gate for Emitter {
    fn tick(&mut self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) {
        // Create continuous bit pattern
        let pattern = BitPattern::simple(self.bit.color(), 0xFFFF, self.bit.thickness());

        // Assign to (single) output
        let output = &mut outputs[0];
        output.pattern = Some(pattern);
    }
}

/// A bit manipulator gate applying a bit operation and pattern to its single input.
///
/// Only works with None/Not/And ops.
#[derive(Component, Default, Debug)]
struct BitManipulator {
    pattern: u16,
    op: BitOp,
}

impl BitManipulator {
    pub fn new(pattern: u16, op: BitOp) -> Self {
        Self { pattern, op }
    }
}

impl Gate for BitManipulator {
    fn tick(&mut self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) {
        let input = &inputs[0];
        let output = &mut outputs[0];
        output.pattern = None;
        let pattern = if let Some(pattern) = input.pattern {
            pattern
        } else {
            return;
        };
        match self.op {
            BitOp::None => (),
            BitOp::Not => {
                if let Some(color) = pattern.monochrome() {
                    if let Some(thickness) = pattern.thickness() {
                        let mut ret = pattern;
                        ret.pattern = !ret.pattern;
                        for i in 0..16 {
                            if ret.pattern & (1u16 << i) != 0 {
                                ret.thicknesses[i] = thickness;
                                ret.colors[i] = color;
                            }
                        }
                        output.pattern = Some(ret);
                    }
                }
            }
            BitOp::And => {
                let mut ret = pattern;
                ret.pattern &= self.pattern;
                output.pattern = Some(ret);
            }
            BitOp::Or => {
                unimplemented!()
            }
            BitOp::Xor => {
                unimplemented!()
            }
        }
    }
}

#[derive(Component, Debug)]
struct Filter {
    color: BitColor,
}

impl Filter {
    pub fn new(color: BitColor) -> Self {
        Self { color }
    }
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            color: BitColor::White,
        }
    }
}

impl Gate for Filter {
    fn tick(&mut self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) {
        // TODO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_manipulator() {
        let input = InputPort {
            port: Orient::Top.into(),
        };
        let mut inputs = [InputBeam {
            port: &input,
            pattern: None,
        }];
        let output = OutputPort {
            port: Orient::Bottom.into(),
        };
        let mut outputs = [OutputBeam {
            port: &output,
            pattern: None,
        }];

        {
            let mut none = BitManipulator::new(0xFFFF, BitOp::None);
            let input_pattern = BitPattern::simple(BitColor::Red, 0xFFFF, 16);
            inputs[0].pattern = Some(input_pattern);
            none.tick(&inputs, &mut outputs);
            assert_eq!(1, outputs.len());
            let output = &outputs[0];
            assert!(output.pattern.is_none());
        }

        {
            let mut not = BitManipulator::new(0xFFFF, BitOp::Not);
            let input_pattern = BitPattern::simple(BitColor::Red, 0xBEEF, 1);
            inputs[0].pattern = Some(input_pattern);
            not.tick(&inputs, &mut outputs);
            assert_eq!(1, outputs.len());
            let output = &outputs[0];
            assert!(output.pattern.is_some());
            let pattern = output.pattern.unwrap();
            let mono = pattern.monochrome();
            assert!(mono.is_some());
            let color = mono.unwrap();
            assert_eq!(BitColor::Red, color);
        }
    }
}
