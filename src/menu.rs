use bevy::{
    app::{AppExit, CoreStage},
    asset::AssetStage,
    input::gamepad::GamepadButtonType,
    prelude::*,
};
use bevy_kira_audio::{
    Audio as KiraAudio, AudioChannel as KiraAudioChannel, AudioPlugin as KiraAudioPlugin,
    AudioSource as KiraAudioSource, InstanceHandle,
};
use bevy_tweening::{lens::*, *};
use leafwing_input_manager::prelude::*;
use std::time::Duration;

pub struct MenuPlugin;

use crate::{AppState, SfxAudio};

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(InputManagerPlugin::<MenuAction>::default())
            .add_plugin(KiraAudioPlugin)
            .init_resource::<AudioManager>()
            .add_system_set(
                SystemSet::on_enter(AppState::Menu)
                    .with_system(menu_setup)
                    .with_system(start_background_audio),
            )
            .add_system_set(
                SystemSet::on_update(AppState::Menu)
                    .with_system(menu_run)
                    .with_system(bevy_tweening::component_animator_system::<UiColor>),
            )
            .add_system_set(SystemSet::on_exit(AppState::Menu).with_system(menu_cleanup));
    }
}

#[derive(Actionlike, PartialEq, Eq, Clone, Copy, Hash, Debug)]
enum MenuAction {
    SelectNext,
    SelectPrev,
    ClickButton,
}

#[derive(Component, Default)]
struct Menu {
    selected_index: i32,
    sound_click: Handle<KiraAudioSource>,
}

#[derive(Component, Default)]
struct Button(pub i32);

pub struct AudioManager {
    pub menu_bgm: Handle<KiraAudioSource>,
    pub menu_instance: Option<InstanceHandle>,
    pub game_bgm: Handle<KiraAudioSource>,
    pub game_instance: Option<InstanceHandle>,
}

impl Default for AudioManager {
    fn default() -> Self {
        AudioManager {
            menu_bgm: Handle::default(),
            menu_instance: None,
            game_bgm: Handle::default(),
            game_instance: None,
        }
    }
}

fn menu_run(
    mut q_menu: Query<(&mut Menu, &mut ActionState<MenuAction>)>,
    mut q_animators: Query<(&Button, &mut Animator<UiColor>), Without<Menu>>,
    q_buttons: Query<(&Button, &Node, &GlobalTransform)>,
    mut exit: EventWriter<AppExit>,
    audio: Res<KiraAudio>,
    sfx_audio: Res<KiraAudioChannel<SfxAudio>>,
    mut app_state: ResMut<State<AppState>>,
    mut cursor_moved_events: EventReader<CursorMoved>,
    mouse_button_input: Res<Input<MouseButton>>,
    mut seq_completed_reader: EventReader<TweenCompleted>,
    mut q_bg: Query<&mut Animator<UiColor>, With<Menu>>,
) {
    // Workaround for lack of loopable sequence
    if seq_completed_reader.iter().last().is_some() {
        let mut bg_animator = q_bg.single_mut();
        bg_animator.rewind();
    }

    let (mut menu, mut action_state) = q_menu.single_mut();
    let prev_sel = menu.selected_index;
    if action_state.just_pressed(MenuAction::SelectNext) {
        menu.selected_index = (menu.selected_index + 1).min(1);
    }
    if action_state.just_pressed(MenuAction::SelectPrev) {
        menu.selected_index = (menu.selected_index - 1).max(0);
    }
    for ev in cursor_moved_events.iter() {
        for (button, node, transform) in q_buttons.iter() {
            let origin = transform.translation().truncate();
            let half_size = node.size / 2.;
            if (origin.x - ev.position.x).abs() < half_size.x
                && (origin.y - ev.position.y).abs() < half_size.y
            {
                menu.selected_index = button.0;
            }
        }
    }

    if prev_sel != menu.selected_index {
        sfx_audio.play(menu.sound_click.clone());
        for (button, mut animator) in q_animators.iter_mut() {
            if button.0 == prev_sel {
                let tween_out = Tween::new(
                    EaseFunction::QuadraticInOut,
                    TweeningType::Once,
                    Duration::from_secs_f32(0.4),
                    UiColorLens {
                        start: MENU_COLORS[1],
                        end: Color::rgb_u8(48, 48, 48),
                    },
                );
                animator.set_tweenable(tween_out);
                animator.state = AnimatorState::Playing;
            } else if button.0 == menu.selected_index {
                let tween_in = Tween::new(
                    EaseFunction::QuadraticInOut,
                    TweeningType::Once,
                    Duration::from_secs_f32(0.4),
                    UiColorLens {
                        start: Color::rgb_u8(48, 48, 48),
                        end: MENU_COLORS[1],
                    },
                );
                animator.set_tweenable(tween_in);
                animator.state = AnimatorState::Playing;
            }
        }
    }

    if action_state.just_pressed(MenuAction::ClickButton) {
        match menu.selected_index {
            0 => {
                // BUGBUG -- https://bevy-cheatbook.github.io/programming/states.html
                action_state.release_all();

                app_state.set(AppState::InGame).unwrap()
            },
            1 => exit.send(AppExit),
            _ => unreachable!(),
        }
    }
}

#[derive(Component)]
struct MenuCamera;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct UiColorLens {
    /// Start color.
    pub start: Color,
    /// End color.
    pub end: Color,
}

impl Lens<UiColor> for UiColorLens {
    fn lerp(&mut self, target: &mut UiColor, ratio: f32) {
        // Note: Add<f32> for Color affects alpha, but not Mul<f32>. So use Vec4 for
        // consistency.
        let start: Vec4 = self.start.into();
        let end: Vec4 = self.end.into();
        let value = start.lerp(end, ratio);
        *target = UiColor(Color::rgb(value.x, value.y, value.z));
    }
}

const MENU_COLORS: &'static [Color] = &[
    Color::rgb(255. / 255., 255. / 255., 255. / 255.),
    Color::rgb(255. / 255., 0. / 255., 0. / 255.),
    Color::rgb(255. / 255., 216. / 255., 0. / 255.),
    Color::rgb(61. / 255., 206. / 255., 0. / 255.),
    Color::rgb(0. / 255., 148. / 255., 255. / 255.),
    Color::rgb(178. / 255., 0. / 255., 255. / 255.),
];

fn menu_setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    println!("menu_setup");
    commands
        .spawn_bundle(Camera2dBundle::default())
        .insert(MenuCamera)
        .insert(Name::new("menu_camera"));

    let font = asset_server.load("fonts/ShareTechMono-Regular.ttf");

    let title_image = asset_server.load("title.png");

    let mut menu = Menu::default();
    //menu.sound_click = asset_server.load("sounds/click4.ogg");

    let mut input_map = InputMap::default();
    input_map.insert(KeyCode::Down, MenuAction::SelectNext);
    input_map.insert(KeyCode::S, MenuAction::SelectNext);
    input_map.insert(GamepadButtonType::DPadDown, MenuAction::SelectNext);
    input_map.insert(KeyCode::Up, MenuAction::SelectPrev);
    input_map.insert(KeyCode::W, MenuAction::SelectPrev);
    input_map.insert(GamepadButtonType::DPadUp, MenuAction::SelectPrev);
    input_map.insert(KeyCode::Return, MenuAction::ClickButton);
    input_map.insert(KeyCode::Space, MenuAction::ClickButton);
    input_map.insert(GamepadButtonType::South, MenuAction::ClickButton);
    #[cfg(not(debug_assertions))] // only in release, otherwise annoying with egui inspector
    input_map.insert(MouseButton::Left, MenuAction::ClickButton);

    let menu_bg_color_seq = Sequence::new(MENU_COLORS.windows(2).map(|w| {
        Tween::new(
            EaseFunction::QuadraticInOut,
            TweeningType::Once,
            Duration::from_secs(3),
            UiColorLens {
                start: w[0],
                end: w[1],
            },
        )
    }))
    .then(
        Tween::new(
            EaseFunction::QuadraticInOut,
            TweeningType::Once,
            Duration::from_secs(3),
            UiColorLens {
                start: MENU_COLORS[5],
                end: MENU_COLORS[0],
            },
        )
        .with_completed_event(0),
    );

    commands
        .spawn_bundle(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                position: UiRect::all(Val::Px(0.)),
                margin: UiRect::all(Val::Px(16.)),
                padding: UiRect::all(Val::Px(16.)),
                flex_direction: FlexDirection::ColumnReverse,
                align_content: AlignContent::Center,
                align_items: AlignItems::Center,
                align_self: AlignSelf::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            color: UiColor(MENU_COLORS[0]),
            ..default()
        })
        .insert(Name::new("menu"))
        .insert(menu)
        .insert(Animator::new(menu_bg_color_seq))
        .insert_bundle(InputManagerBundle::<MenuAction> {
            action_state: ActionState::default(),
            input_map,
        })
        .with_children(|parent| {
            // Title
            let w = 230.;
            let h = 80.;
            parent
                .spawn_bundle(NodeBundle {
                    node: Node {
                        size: Vec2::new(w, h),
                    },
                    style: Style {
                        size: Size::new(Val::Px(w), Val::Px(h)),
                        min_size: Size::new(Val::Px(w), Val::Px(h)),
                        margin: UiRect::all(Val::Px(64.)),
                        padding: UiRect::all(Val::Px(0.)),
                        align_content: AlignContent::Center,
                        align_items: AlignItems::Center,
                        align_self: AlignSelf::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    image: UiImage(title_image),
                    ..default()
                })
                .insert(Name::new("title"));

            const DURATION_SEC: f32 = 1.2;
            const DELAY_MS: u64 = 200;

            let mut start_time_ms = 0;
            for (index, (text, color)) in [("New Game", Color::RED), ("Quit", Color::ORANGE)]
                .iter()
                .enumerate()
            {
                let delay = Delay::new(Duration::from_millis(start_time_ms));
                start_time_ms += DELAY_MS;
                let tween_scale = Tween::new(
                    EaseFunction::BounceOut,
                    TweeningType::Once,
                    Duration::from_secs_f32(DURATION_SEC),
                    UiColorLens {
                        start: Color::rgb_u8(48, 48, 48),
                        end: if index == 0 { MENU_COLORS[1] } else { Color::rgb_u8(48, 48, 48) },
                    },
                );
                let seq = delay.then(tween_scale.with_completed_event(0));
                let w = 200.;
                let h = 40.;
                parent
                    .spawn_bundle(NodeBundle {
                        node: Node {
                            size: Vec2::new(w, h),
                        },
                        style: Style {
                            min_size: Size::new(Val::Px(w), Val::Px(h)),
                            margin: UiRect::all(Val::Px(6.)),
                            padding: UiRect::all(Val::Px(6.)),
                            align_content: AlignContent::Center,
                            align_items: AlignItems::Center,
                            align_self: AlignSelf::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        color: UiColor(Color::rgb_u8(48, 48, 48)),
                        ..default()
                    })
                    .insert(Name::new(format!("button:{}", text)))
                    .insert(Button(index as i32))
                    .insert(Animator::new(seq))
                    .with_children(|parent| {
                        parent.spawn_bundle(TextBundle {
                            text: Text::from_section(
                                text.to_string(),
                                TextStyle {
                                    font: font.clone(),
                                    font_size: 32.0,
                                    color: Color::rgb_u8(192, 192, 192),
                                },
                            )
                            .with_alignment(TextAlignment {
                                vertical: VerticalAlign::Center,
                                horizontal: HorizontalAlign::Center,
                            }),
                            ..default()
                        });
                    });
            }
        });
}

fn menu_cleanup(
    mut commands: Commands,
    query: Query<Entity, With<Menu>>,
    query_camera: Query<Entity, With<MenuCamera>>,
) {
    commands.entity(query.single()).despawn_recursive();
    commands.entity(query_camera.single()).despawn_recursive();
}

fn start_background_audio(
    asset_server: Res<AssetServer>,
    audio: Res<KiraAudio>,
    mut audio_manager: ResMut<AudioManager>,
) {
    audio_manager.menu_bgm = asset_server.load("bgm/bgm.ogg");
    audio.set_volume(1.);
    audio_manager.menu_instance = Some(audio.play_looped(audio_manager.menu_bgm.clone()));
}
