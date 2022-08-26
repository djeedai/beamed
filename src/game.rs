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
            .add_event::<RebuildBeamsEvent>()
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
                    .with_system(update_board.after(update_cursor))
                    .with_system(rebuild_beams.after(update_board)),
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

#[derive(Debug)]
struct RebuildBeamsEvent;

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
    mut commands: Commands,
    database: Res<ItemDatabase>,
    mut board_query: Query<&mut Board>,
    mut cell_query: Query<(&mut Cell, &mut Handle<Image>, &mut Transform), Without<Beam>>,
    mut beam_query: Query<(Entity, &mut Beam, &mut Transform, &Mesh2dHandle), Without<Cell>>,
    mut place_item_event_reader: EventReader<PlaceItemEvent>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rebuild_beams_event_writer: EventWriter<RebuildBeamsEvent>,
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

                let cell_size = board.cell_size();

                // Update board, for logic
                trace!(
                    "board.add() ipos={:?} orient={:?} item_id={:?}",
                    ev.ipos,
                    ev.orient,
                    ev.item_id
                );
                let _ = board.add(ev.ipos, ev.orient, cell_entity, ev.item_id, item);
                rebuild_beams_event_writer.send(RebuildBeamsEvent);
            }
        }
    }
}

/// Rebuild all beams on the board after some tile changed
fn rebuild_beams(
    mut commands: Commands,
    database: Res<ItemDatabase>,
    mut board_query: Query<&mut Board>,
    mut cell_query: Query<(&mut Cell, &mut Handle<Image>, &mut Transform), Without<Beam>>,
    mut beam_query: Query<Entity, With<Beam>>,
    mut rebuild_beams_event_reader: EventReader<RebuildBeamsEvent>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut beam_materials: ResMut<Assets<BeamMaterial>>,
) {
    // Drain all event and early out if none
    if rebuild_beams_event_reader.iter().last().is_none() {
        return;
    }

    // Clear all existing beam entities
    for entity in beam_query.iter() {
        commands.entity(entity).despawn_recursive();
    }

    // Rebuild all beams
    let mut board = board_query.single_mut();
    let cell_size = board.cell_size();
    if let Some(beams) = board.rebuild_beams(&*database) {
        for beam in beams {
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
            beam.rebuild_mesh(&mut mesh, board.cell_size());
            let mesh = Mesh2dHandle(meshes.add(mesh));
            let material = beam_materials.add(BeamMaterial {
                color: beam.pattern.colors[0].into(), // FIXME - multi-color
                pattern: beam.pattern.pattern as u32,
            });
            let z = 0.5; // above everything else
            let orient = Orient::from_dir(beam.end - beam.start); // FIXME - this is the GLOBAL orient!
            let offset = orient.to_dir().as_vec2() * 0.5;
            commands
                .spawn_bundle(MaterialMesh2dBundle {
                    mesh,
                    material,
                    transform: Transform {
                        translation: ((offset + beam.start.as_vec2()) * cell_size).extend(z),
                        rotation: orient.into(),
                        ..default()
                    },
                    ..default()
                })
                .insert(Name::new(format!("beam@{:?}", beam.start)))
                .insert(beam);
        }
    }

    // // Connect outputs of new item
    // for output in &item.outputs {
    //     trace!("Output: port={:?}", output.port);
    //     match output.port {
    //         Port::Single(out_orient) => {
    //             let global_out_orient = ev.orient + out_orient;
    //             trace!(
    //                 "+ item_orient={:?} local_orient={:?} global_orient={:?}",
    //                 ev.orient,
    //                 out_orient,
    //                 global_out_orient
    //             );

    //             if let Some(in_tile) = board.find(ev.ipos, global_out_orient) {
    //                 trace!(
    //                     "Found tile: ipos={:?} orient={:?} id={:?}",
    //                     in_tile.ipos,
    //                     in_tile.orient,
    //                     in_tile.item_id,
    //                 );

    //                 let in_item = database.get(in_tile.item_id);

    //                 // Calculate the orientation of the output, which is the opposite
    //                 // of the orientation of the search.
    //                 let global_in_orient = global_out_orient.reversed();
    //                 trace!("global_in_orient = {:?}", global_in_orient);

    //                 // Calculate the orientation locally for the output port/item
    //                 let local_in_orient = global_in_orient - in_tile.orient;
    //                 trace!("local_in_orient = {:?}", local_in_orient);

    //                 // Check if the item has an output to connect to
    //                 for input in &in_item.inputs {
    //                     trace!("+ output = {:?}", output.port);

    //                     if input.port.can_connect_from(local_in_orient) {
    //                         trace!("Connect!");

    //                         //out_tile.

    //                         // let mesh: Mesh2dHandle = meshes
    //                         //     .add(shape::Quad::new(Vec2::new(140., 4.)).into())
    //                         //     .into();
    //                         // commands.spawn_bundle(MaterialMesh2dBundle {
    //                         //     mesh,
    //                         //     material: beam_materials.add(BeamMaterial {
    //                         //         color: Color::PURPLE,
    //                         //         pattern: 0xAAF0,
    //                         //     }),
    //                         //     //material: color_materials.add(Color::PURPLE.into()),
    //                         //     ..default()
    //                         // });
    //                     }
    //                 }

    //                 // Beams cannot cross an item, so stop search here
    //                 break;
    //             } else {
    //                 // Didn't find any tile. If emitter, a beam until board side
    //                 board.add_beam(new_tile)
    //             }
    //         }
    //         Port::PassThrough(pto) => {}
    //         Port::Any => {}
    //     }
    // }

    // // Connect inputs of new item
    // for input in &item.inputs {
    //     trace!("Input: port={:?}", input.port);
    //     match input.port {
    //         Port::Single(in_orient) => {
    //             let global_in_orient = ev.orient + in_orient;
    //             trace!(
    //                 "+ item_orient={:?} local_orient={:?} global_orient={:?}",
    //                 ev.orient,
    //                 in_orient,
    //                 global_in_orient
    //             );

    //             if let Some(out_tile) = board.find(ev.ipos, global_in_orient) {
    //                 trace!(
    //                     "Found tile: ipos={:?} orient={:?} id={:?}",
    //                     out_tile.ipos,
    //                     out_tile.orient,
    //                     out_tile.item_id,
    //                 );

    //                 let out_item = database.get(out_tile.item_id);

    //                 // Calculate the orientation of the output, which is the opposite
    //                 // of the orientation of the search.
    //                 let global_out_orient = global_in_orient.reversed();
    //                 trace!("global_out_orient = {:?}", global_out_orient);

    //                 // Calculate the orientation locally for the output port/item
    //                 let local_out_orient = global_out_orient - out_tile.orient;
    //                 trace!("local_out_orient = {:?}", local_out_orient);

    //                 // Check if the item has an output to connect to
    //                 for output in &out_item.outputs {
    //                     trace!("+ output = {:?}", output.port);

    //                     if output.port.can_connect_from(local_out_orient) {
    //                         trace!("Connect!");

    //                         out_tile.

    //                         // let mesh: Mesh2dHandle = meshes
    //                         //     .add(shape::Quad::new(Vec2::new(140., 4.)).into())
    //                         //     .into();
    //                         // commands.spawn_bundle(MaterialMesh2dBundle {
    //                         //     mesh,
    //                         //     material: beam_materials.add(BeamMaterial {
    //                         //         color: Color::PURPLE,
    //                         //         pattern: 0xAAF0,
    //                         //     }),
    //                         //     //material: color_materials.add(Color::PURPLE.into()),
    //                         //     ..default()
    //                         // });
    //                     }
    //                 }

    //                 // Beams cannot cross an item, so stop search here
    //                 break;
    //             }
    //         }
    //         Port::PassThrough(pto) => {}
    //         Port::Any => {}
    //     }
    // }
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
            self.orient.turn_left();
        } else if dir > 0 {
            self.orient.turn_right();
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
            Orient::Right => -PI / 2.,
            Orient::Bottom => PI,
            Orient::Left => PI / 2.,
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

    pub fn from_dir(dir: IVec2) -> Self {
        if dir.x == 0 {
            assert!(dir.y != 0);
            if dir.y > 0 {
                Orient::Top
            } else {
                Orient::Bottom
            }
        } else {
            assert!(dir.y == 0);
            if dir.x > 0 {
                Orient::Right
            } else {
                Orient::Left
            }
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

enum SplitResult {
    None,
    DeleteSelf(Option<Entity>),
    ShortenSelf(Option<Entity>),
}

#[derive(Component, Debug)]
struct Beam {
    pub start: IVec2,
    pub end: IVec2,
    pub output_entity: Entity,        // beam source from output port
    pub input_entity: Option<Entity>, // beam target, None if not connected to input port
    pub pattern: BitPattern,
}

impl Beam {
    /// Rebuild the mesh for the current beam
    pub fn rebuild_mesh(&self, mesh: &mut Mesh, cell_size: Vec2) {
        // Find the size of the beam
        let (vertices, uvs) = if self.start.x == self.end.x {
            // vertical
            assert!(self.end.y != self.start.y);
            let x = 2.;
            let y = (self.end.y - self.start.y).abs() as f32 * cell_size.y;
            let vertices = vec![[0., 0., 0.], [x, 0., 0.], [0., y, 0.], [x, y, 0.]];
            let uvs = vec![[0., 0.], [0., 1.], [x, 0.], [x, 1.]];
            (vertices, uvs)
        } else {
            // horizontal
            assert!(self.end.y == self.start.y);
            let x = (self.end.x - self.start.x).abs() as f32 * cell_size.x;
            let y = 2.;
            let vertices = vec![[0., 0., 0.], [x, 0., 0.], [0., y, 0.], [x, y, 0.]];
            let uvs = vec![[0., 0.], [x, 0.], [0., 1.], [x, 1.]];
            (vertices, uvs)
        };

        // Build the mesh
        let normals = vec![[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]];
        let indices = vec![0, 1, 2, 2, 1, 3];
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.set_indices(Some(Indices::U32(indices)));
    }

    pub fn width(&self) -> i32 {
        if self.start.x == self.end.x {
            assert!(self.end.y != self.start.y);
            (self.end.y - self.start.y).abs()
        } else {
            assert!(self.end.y == self.start.y);
            (self.end.x - self.start.x).abs()
        }
    }

    /// Try to split the current beam at the given position.
    pub fn try_split_at(&mut self, ipos: IVec2, entity: Entity) -> SplitResult {
        if self.start.x == self.end.x && self.start.x == ipos.x {
            // vertical
            assert!(self.start.y != self.end.y);
            if self.start.y < self.end.y {
                // going up
                if ipos.y >= self.start.y && ipos.y <= self.end.y {
                    // intersect
                    let prev_end = self.input_entity;
                    if ipos.y == self.start.y {
                        // empty self
                        return SplitResult::DeleteSelf(prev_end);
                    }
                    self.end.y = ipos.y;
                    self.input_entity = Some(entity);
                    return SplitResult::ShortenSelf(prev_end);
                } else {
                    // disjoint
                    return SplitResult::None;
                }
            } else {
                // going down
                if ipos.y >= self.end.y && ipos.y <= self.start.y {
                    // intersect
                    let prev_end = self.input_entity;
                    if ipos.y == self.start.y {
                        // empty self
                        return SplitResult::DeleteSelf(prev_end);
                    }
                    self.end.y = ipos.y;
                    self.input_entity = Some(entity);
                    return SplitResult::ShortenSelf(prev_end);
                } else {
                    // disjoint
                    return SplitResult::None;
                }
            }
        } else if self.start.y == self.end.y && self.start.y == ipos.y {
            // horizontal
            assert!(self.start.x != self.end.x);
            if self.start.x < self.end.x {
                // going right
                if ipos.x >= self.start.x && ipos.x <= self.end.x {
                    // intersect
                    let prev_end = self.input_entity;
                    if ipos.x == self.start.x {
                        // empty self
                        return SplitResult::DeleteSelf(prev_end);
                    }
                    self.end.x = ipos.x;
                    self.input_entity = Some(entity);
                    return SplitResult::ShortenSelf(prev_end);
                } else {
                    // disjoint
                    return SplitResult::None;
                }
            } else {
                // going left
                if ipos.x >= self.end.x && ipos.x <= self.start.x {
                    // intersect
                    let prev_end = self.input_entity;
                    if ipos.x == self.start.x {
                        // empty self
                        return SplitResult::DeleteSelf(prev_end);
                    }
                    self.end.x = ipos.x;
                    self.input_entity = Some(entity);
                    return SplitResult::ShortenSelf(prev_end);
                } else {
                    // disjoint
                    return SplitResult::None;
                }
            }
        }
        SplitResult::None
    }
}

#[derive(Debug, Default, Clone)]
struct Link {
    /// Connected entity, if any.
    pub entity: Option<Entity>,
    /// Active bit pattern at the port.
    pub pattern: BitPattern,
    /// Global-space orientation of link.
    pub global_orient: Orient,
}

/// Placed item on board.
#[derive(Debug, Clone)]
struct Tile {
    item_id: ItemId,
    ipos: IVec2,
    orient: Orient,
    entity: Entity,
    inputs: Vec<(InputPort, Link)>,
    outputs: Vec<(OutputPort, Link)>,
}

impl Tile {
    /// Clear all tile's inputs and outputs of any connection.
    pub fn clear_inputs_outputs(&mut self) {
        self.inputs.iter_mut().for_each(|(_, link)| {
            link.entity = None;
            link.pattern.pattern = 0;
        });
        self.outputs.iter_mut().for_each(|(_, link)| {
            link.entity = None;
            link.pattern.pattern = 0;
        });
    }

    /// Connect to an output entity emitting a beam from a given orientation.
    pub fn connect_input_from(
        &mut self,
        global_orient: Orient, // incoming orient
        output_entity: Entity,
        pattern: BitPattern,
    ) -> bool {
        trace!(
            "connect_input_from(global_orient={:?}, output_entity={:?}, pattern={:?})",
            global_orient,
            output_entity,
            pattern,
        );
        let local_orient = global_orient - self.orient;
        trace!(
            "=> self.orient={:?} local_orient={:?} self.inputs={}",
            self.orient,
            local_orient,
            self.inputs.len()
        );
        for (port, link) in &mut self.inputs {
            trace!("+ input: {:?} (link={:?})", port, link);
            if port.port.can_connect_from(local_orient) {
                trace!("  => CONNECT!");
                *link = Link {
                    entity: Some(output_entity),
                    pattern,
                    global_orient: local_orient + self.orient,
                };
                trace!("  => Link = {:?}", link);
                return true;
            }
        }
        false
    }
}

enum RaycastResult {
    /// Reached the board side at the given position without finding a tile.
    NotFound(IVec2),
    /// Found a tile.
    Found(usize),
}

#[derive(Component, Debug)]
struct Board {
    size: IVec2,
    cell_size: Vec2,
    /// Entity for drawning the cell.
    cells: Vec<Entity>,
    /// Entity for an item, if any.
    tiles: Vec<Option<Tile>>,
}

impl Board {
    pub fn new(size: IVec2) -> Self {
        let area = size.x as usize * size.y as usize;
        Self {
            size,
            cell_size: Vec2::splat(32.),
            cells: vec![], // TEMP; until set_cells() called
            tiles: vec![None; area],
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
        (r.0.y..=r.1.y)
            .flat_map(move |j| (r.0.x..=r.1.x).map(move |i| (i, j)))
            .map(|(i, j)| IVec2::new(i, j))
    }

    #[inline]
    pub fn size(&self) -> IVec2 {
        self.size
    }

    #[inline]
    pub fn rect(&self) -> (IVec2, IVec2) {
        // FIXME - works only on odd size...
        let half_size = (self.size - 1) / 2;
        (-half_size, half_size)
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
        item: &Item,
    ) -> &mut Tile {
        // Add item to board
        let index = self.index(ipos);
        self.tiles[index] = Some(Tile {
            item_id,
            ipos,
            orient,
            entity,
            inputs: item.inputs.iter().map(|i| (*i, Link::default())).collect(),
            outputs: item.outputs.iter().map(|o| (*o, Link::default())).collect(),
        });
        self.tiles[index].as_mut().unwrap()
    }

    pub fn try_get_tile(&self, entity: Entity) -> Option<&Tile> {
        for tile in &self.tiles {
            if let Some(tile) = tile {
                if tile.entity == entity {
                    return Some(tile);
                }
            }
        }
        None
    }

    pub fn try_get_tile_mut(&mut self, entity: Entity) -> Option<&mut Tile> {
        for tile in &mut self.tiles {
            if let Some(tile) = tile {
                if tile.entity == entity {
                    return Some(tile);
                }
            }
        }
        None
    }

    pub fn try_get(&self, ipos: IVec2) -> GetBoard {
        let r = self.rect();
        if ipos.x >= r.0.x && ipos.x <= r.1.x && ipos.y >= r.0.y && ipos.y <= r.1.y {
            let index = self.index(ipos);
            trace!(
                "try_get: r={:?} ipos={:?} index={} len={}",
                r,
                ipos,
                index,
                self.tiles.len()
            );
            if self.tiles[index].is_some() {
                GetBoard::Tile(index)
            } else {
                GetBoard::Empty
            }
        } else {
            GetBoard::Outside
        }
    }

    /// Find a tile search from an existing starting tile alongside the given orientation.
    pub fn find(&self, start: IVec2, orient: Orient) -> RaycastResult {
        let r = self.rect();
        let index = self.index(start);
        let tile = self.tiles[index].as_ref().unwrap();

        // Get the world-space orientation of the search from its local orientation
        // and the orientation of the start tile itself.
        let global_orient = tile.orient + orient;
        let dir = orient.to_dir();

        // Check all tiles in the given direction until a connection is made or
        // the border of the board is reached without finding one.
        let mut prev_ipos = start;
        let mut ipos = start + dir;
        loop {
            match self.try_get(ipos) {
                // No tile at this position, continue checking next one in same direction
                GetBoard::Empty => {
                    prev_ipos = ipos;
                    ipos += dir;
                }
                // Reached the border of the board, found nothing. Return the last position
                // actually on the board.
                GetBoard::Outside => return RaycastResult::NotFound(prev_ipos),
                // Reached an item, return it.
                GetBoard::Tile(index) => return RaycastResult::Found(index),
            }
        }
    }

    pub fn rebuild_beams(&mut self, database: &ItemDatabase) -> Option<Vec<Beam>> {
        trace!("Board::rebuild_beams()");

        let mut beams = vec![];
        let mut queue = vec![];

        // Loop over all existing tiles placed on the board
        for index in 0..self.tiles.len() {
            if let Some(tile) = &mut self.tiles[index] {
                // Clear all existing connections in and out of that tile
                trace!("Clear tile at {:?}", tile.ipos);
                tile.clear_inputs_outputs();

                // Queue the entity for update
                queue.push(index);
            }
        }

        // Depth-first search
        trace!("Depth-first search");
        while let Some(index) = queue.pop() {
            let tile = self.tiles[index].as_ref().unwrap();
            trace!(
                "Processing tile #{} at {:?} (orient={:?}; Entity={:?})",
                index,
                tile.ipos,
                tile.orient,
                tile.entity
            );

            // Tick the item at the tile to propagate beam signals
            let item = database.get(tile.item_id);
            let inputs: Vec<InputBeam> = tile
                .inputs
                .iter()
                .map(|(port, link)| InputBeam {
                    port,
                    state: if let Some(entity) = link.entity {
                        Some(InputPortState {
                            pattern: link.pattern,
                            in_orient: link.global_orient,
                        })
                    } else {
                        None
                    },
                })
                .collect();
            let mut outputs: Vec<OutputBeam> = tile
                .outputs
                .iter()
                .map(|(port, link)| OutputBeam {
                    port,
                    state: if let Some(entity) = link.entity {
                        Some(OutputPortState {
                            pattern: link.pattern,
                            out_orient: link.global_orient,
                        })
                    } else {
                        None
                    },
                })
                .collect();
            item.tick(&inputs, &mut outputs);

            let out_tile_ipos = tile.ipos;
            let out_tile_orient = tile.orient;
            let out_tile_entity = tile.entity;

            // Disconnect outputs from tile by losing the port reference, to allow mutating the tile
            // FIXME - refactor with tick() taking the immutable port decription as & separately from the mutable state as &mut to avoid this!
            let mut outputs: Vec<_> = outputs.iter().map(|ob| ob.state).collect();

            // Check output ports after ticking
            trace!("Looping on {} outputs", outputs.len());
            for output_state in outputs {
                // Check if port was actived by the tick() call
                trace!("+ Output: active={}", output_state.is_some());
                if let Some(state) = &output_state {
                    // Output is active, create a beam and try to connect to an input
                    trace!("  State: {:?}", output_state);

                    // Raycast from the output port in its direction to find another tile
                    let global_orient = out_tile_orient + state.out_orient;
                    trace!(
                        "  Orient: local={:?} tile={:?} global={:?}",
                        state.out_orient,
                        out_tile_orient,
                        global_orient
                    );
                    match self.find(out_tile_ipos, global_orient) {
                        RaycastResult::Found(in_tile_index) => {
                            let in_tile = self.tiles[in_tile_index].as_mut().unwrap();
                            trace!(
                                "  => Found(#{}) @ {:?} orient={:?}",
                                in_tile_index,
                                in_tile.ipos,
                                in_tile.orient
                            );
                            // Try to connect the beam to an input port of the tile found
                            if in_tile.connect_input_from(
                                global_orient.reversed(),
                                out_tile_entity,
                                state.pattern,
                            ) {
                                // Connected; make beam between both entities
                                trace!("  => connected");
                                trace!(
                                    "  Beam: {:?}->{:?} (out_entity={:?}, in_entity={:?}, pattern={:?})",
                                    out_tile_ipos,
                                    in_tile.ipos,
                                    out_tile_entity,
                                    in_tile.entity,
                                    state.pattern
                                );
                                beams.push(Beam {
                                    start: out_tile_ipos,
                                    end: in_tile.ipos,
                                    output_entity: out_tile_entity,
                                    input_entity: Some(in_tile.entity),
                                    pattern: state.pattern,
                                });

                                // Since the new tile connected, in turns it must be processed
                                queue.push(in_tile_index);
                            } else {
                                // Not connected; simply make beam from current tile to position before blocker
                                trace!("  => not connected");
                                trace!(
                                    "  Beam: {:?}->{:?} (out_entity={:?}, pattern={:?})",
                                    out_tile_ipos,
                                    in_tile.ipos,
                                    out_tile_entity,
                                    state.pattern
                                );
                                beams.push(Beam {
                                    start: out_tile_ipos,
                                    end: in_tile.ipos,
                                    output_entity: out_tile_entity,
                                    input_entity: None,
                                    pattern: state.pattern,
                                });
                            }
                        }
                        RaycastResult::NotFound(last_ipos) => {
                            trace!("  => NotFound({:?})", last_ipos);
                            trace!(
                                "  Beam: {:?}->{:?} (out_entity={:?}, pattern={:?})",
                                out_tile_ipos,
                                last_ipos,
                                out_tile_entity,
                                state.pattern
                            );
                            // No endpoint tile; only create beam, until the board side
                            beams.push(Beam {
                                start: out_tile_ipos,
                                end: last_ipos,
                                output_entity: out_tile_entity,
                                input_entity: None,
                                pattern: state.pattern,
                            });
                        }
                    }
                }
            }
        }

        if beams.is_empty() {
            trace!("Rebuilt no beam.");
            None
        } else {
            trace!("Rebuilt {} beams.", beams.len());
            Some(beams)
        }
    }
}

#[derive(Debug)]
enum GetBoard {
    Empty,
    Outside,
    Tile(usize),
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
            Box::new(BitManipulator::new(0xF0F0, BitOp::And)) as Box<dyn Gate>,
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
    //input_map.insert(KeyCode::Return, PlayerAction::PlaceSelectedItem); // this conflicts with Start Game menu entry
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

enum TickResult {
    Idle,
    OutputChanged,
}

trait Gate: Send + Sync + 'static {
    fn tick(&self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) -> TickResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassThroughOrient {
    Horizontal,
    Vertical,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Port {
    /// Single-orientation port.
    Single(Orient),
    /// Pass-through port, can be connected from either of its opposite endpoints.
    PassThrough(PassThroughOrient),
    /// All-way pass-through, can be connected from all orientations.
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

/// Description on an input port of an item.
#[derive(Debug, Clone, Copy)]
struct InputPort {
    port: Port,
}

/// Description on an output port of an item.
#[derive(Debug, Clone, Copy)]
struct OutputPort {
    port: Port,
}

#[derive(Debug, Clone, Copy)]
struct InputPortState {
    pattern: BitPattern,
    in_orient: Orient,
}

/// Input port of a tile and its state.
#[derive(Debug, Clone, Copy)]
struct InputBeam<'a> {
    /// Port description.
    port: &'a InputPort,
    /// Current port state.
    state: Option<InputPortState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutputPortState {
    pattern: BitPattern,
    out_orient: Orient,
}

/// Output port of a tile and its state.
#[derive(Debug, Clone, Copy)]
struct OutputBeam<'a> {
    /// Port description.
    port: &'a OutputPort,
    /// Current port state.
    state: Option<OutputPortState>,
}

struct Item {
    name: String,
    image: Handle<Image>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    gate: Box<dyn Gate>,
}

impl Item {
    #[inline]
    pub fn tick(&self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) {
        self.gate.tick(inputs, outputs);
    }
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
    fn tick(&self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) -> TickResult {
        // Create continuous bit pattern
        let pattern = BitPattern::simple(self.bit.color(), 0xFFFF, self.bit.thickness());

        // Assign to (single) output
        let output = &mut outputs[0];
        match output.port.port {
            Port::Single(orient) => {
                output.state = Some(OutputPortState {
                    pattern,
                    out_orient: orient,
                });
            }
            _ => panic!("Emitter only supports Port::Single() output."),
        }

        TickResult::OutputChanged
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
    fn tick(&self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) -> TickResult {
        trace!(
            "BitManipulator::tick(inputs = {:?}, outputs = {:?})",
            inputs,
            outputs
        );

        // Only 1 input and 1 output by design
        let input = &inputs[0];
        let output = &mut outputs[0];

        // Check if the input port is active and get its state
        let input_state = if let Some(input_state) = &input.state {
            input_state
        } else if output.state.is_some() {
            // No input, but there was an output; clear it
            trace!("Inactive input, clearing output too.");
            output.state = None;
            return TickResult::OutputChanged;
        } else {
            // Neither input not output
            trace!("Inactive input and output, nothing to do.");
            return TickResult::Idle;
        };

        trace!("input_state = {:?}", input_state);

        // Calculate the new output state based on the input one
        let mut new_output_state = None;
        trace!("BitOp = {:?}", self.op);
        match self.op {
            BitOp::None => (),
            BitOp::Not => {
                if let Some(color) = input_state.pattern.monochrome() {
                    if let Some(thickness) = input_state.pattern.thickness() {
                        trace!("mono({:?}) + thickness({:?})", color, thickness);
                        let mut ret = input_state.pattern;
                        trace!("<= pattern = {:?}", ret.pattern);
                        ret.pattern = !ret.pattern;
                        trace!("=> pattern = {:?}", ret.pattern);
                        for i in 0..16 {
                            if ret.pattern & (1u16 << i) != 0 {
                                ret.thicknesses[i] = thickness;
                                ret.colors[i] = color;
                            }
                        }
                        trace!(
                            "ret = {:?} | out_orient = {:?}",
                            ret,
                            input_state.in_orient.reversed()
                        );
                        new_output_state = Some(OutputPortState {
                            pattern: ret,
                            out_orient: input_state.in_orient.reversed(),
                        });
                    }
                }
            }
            BitOp::And => {
                let mut ret = input_state.pattern;
                trace!("<= pattern = {:?}", ret.pattern);
                ret.pattern &= self.pattern;
                trace!("=> pattern = {:?}", ret.pattern);
                trace!(
                    "ret = {:?} | out_orient = {:?}",
                    ret,
                    input_state.in_orient.reversed()
                );
                new_output_state = Some(OutputPortState {
                    pattern: ret,
                    out_orient: input_state.in_orient.reversed(),
                });
            }
            BitOp::Or => {
                unimplemented!()
            }
            BitOp::Xor => {
                unimplemented!()
            }
        }

        // Check if state changed
        if let Some(prev_output_state) = &output.state {
            if let Some(new_output_state) = &new_output_state {
                // old = Some, new = Some
                trace!(
                    "output: old = {:?} | new = {:?} | changed = {}",
                    prev_output_state,
                    new_output_state,
                    prev_output_state == new_output_state
                );
                if prev_output_state == new_output_state {
                    trace!("Idle: old == new == {:?}", prev_output_state);
                    return TickResult::Idle;
                }
                output.state = Some(*new_output_state);
                trace!("OutputChanged: output => {:?}", output.state);
                return TickResult::OutputChanged;
            } else {
                // old = Some, new = None
                output.state = None;
                trace!("OutputChanged: output => None");
                return TickResult::OutputChanged;
            }
        } else if new_output_state.is_some() {
            // old = None, new = Some
            output.state = new_output_state;
            trace!("OutputChanged: output => {:?}", output.state);
            return TickResult::OutputChanged;
        } else {
            // old = None, new = None
            trace!("Idle: old == new == None");
            return TickResult::Idle;
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
    fn tick(&self, inputs: &[InputBeam], outputs: &mut [OutputBeam]) -> TickResult {
        TickResult::Idle // TODO
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
            state: None,
        }];
        let output = OutputPort {
            port: Orient::Bottom.into(),
        };
        let mut outputs = [OutputBeam {
            port: &output,
            state: None,
        }];

        // {
        //     let mut none = BitManipulator::new(0xFFFF, BitOp::None);
        //     let input_pattern = BitPattern::simple(BitColor::Red, 0xFFFF, 16);
        //     inputs[0].pattern = Some(input_pattern);
        //     none.tick(&inputs, &mut outputs);
        //     assert_eq!(1, outputs.len());
        //     let output = &outputs[0];
        //     assert!(output.pattern.is_none());
        // }

        // {
        //     let mut not = BitManipulator::new(0xFFFF, BitOp::Not);
        //     let input_pattern = BitPattern::simple(BitColor::Red, 0xBEEF, 1);
        //     inputs[0].pattern = Some(input_pattern);
        //     not.tick(&inputs, &mut outputs);
        //     assert_eq!(1, outputs.len());
        //     let output = &outputs[0];
        //     assert!(output.pattern.is_some());
        //     let pattern = output.pattern.unwrap();
        //     let mono = pattern.monochrome();
        //     assert!(mono.is_some());
        //     let color = mono.unwrap();
        //     assert_eq!(BitColor::Red, color);
        // }
    }
}
