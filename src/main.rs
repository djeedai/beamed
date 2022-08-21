#![allow(dead_code, unused_imports, unused_variables, unused_mut)]

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    render::{render_resource::AddressMode, texture::ImageSettings},
    window::PresentMode,
};
use bevy_kira_audio::{Audio, AudioApp, AudioPlugin};
use bevy_tweening::*;

#[cfg(debug_assertions)]
use bevy_inspector_egui::WorldInspectorPlugin;

mod debug;
mod game;
mod menu;

use debug::DebugPlugin;
use game::{GamePlugin, SfxAudio};
use menu::MenuPlugin;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppState {
    Boot,
    Menu,
    InGame,
}

fn main() {
    // RHHHHAAAAAAA!!!! HAMMER TIME!!!!
    let mut sampler = ImageSettings::default_nearest();
    sampler.default_sampler.address_mode_u = AddressMode::Repeat;
    sampler.default_sampler.address_mode_v = AddressMode::Repeat;
    sampler.default_sampler.address_mode_w = AddressMode::Repeat;

    let mut app = App::new();
    app.insert_resource(WindowDescriptor {
        title: "Combine".to_string(),
        // width: 1200.,
        // height: 600.,
        present_mode: PresentMode::Fifo, // vsync
        ..Default::default()
    })
    .insert_resource(bevy::log::LogSettings {
        level: bevy::log::Level::WARN,
        filter: "combine=trace".to_string(),
    })
    .insert_resource(sampler) // RHHHHAAAAAAA!!!!
    .insert_resource(ClearColor(Color::rgba(0., 0., 0., 0.)))
    .add_plugins(DefaultPlugins)
    //.add_plugin(LogDiagnosticsPlugin::default())
    .add_plugin(FrameTimeDiagnosticsPlugin::default());

    #[cfg(debug_assertions)]
    app.add_plugin(DebugPlugin)
        .add_plugin(WorldInspectorPlugin::new());

    app.add_plugin(TweeningPlugin)
        .add_plugin(AudioPlugin)
        .add_audio_channel::<SfxAudio>();

    let initial_state = AppState::Boot;
    app.add_state(initial_state)
        .add_state_to_stage(CoreStage::First, initial_state) // BUG #1671
        .add_state_to_stage(CoreStage::PreUpdate, initial_state) // BUG #1671
        .add_state_to_stage(CoreStage::PostUpdate, initial_state) // BUG #1671
        .add_state_to_stage(CoreStage::Last, initial_state); // BUG #1671

    app.add_plugin(MenuPlugin).add_plugin(GamePlugin);

    // Only enable MSAA on non-web platforms
    #[cfg(not(target_arch = "wasm32"))]
    app.insert_resource(Msaa { samples: 4 });

    app.add_system_set(SystemSet::on_update(AppState::Boot).with_system(boot));

    app.run();
}

fn boot(mut state: ResMut<State<AppState>>) {
    // workaround for on_enter() not working on initial state; use a dummy initial state instead
    state.set(AppState::Menu).unwrap();
}
