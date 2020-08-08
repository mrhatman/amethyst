//! Custom Render Pass example

mod custom_pass;

use crate::custom_pass::{RenderCustom};
use amethyst::{
    input::{
        is_close_requested, is_key_down, InputBundle, InputEvent, ScrollDirection, StringBindings,
    },
    prelude::*,
    renderer::{plugins::RenderToWindow, types::DefaultBackend, RenderingBundle},
    utils::application_root_dir,
    winit::VirtualKeyCode,
};
use amethyst_core::{Transform, TransformBundle};
use amethyst_rendy::{Camera, ActiveCamera, ImageFormat, SpriteSheet, Texture, SpriteSheetFormat, RenderFlat2D, SpriteRender};
use amethyst_rendy::sprite::SpriteSheetHandle;
use amethyst_assets::{Loader,AssetStorage};
use amethyst_rendy::bundle::Target;

pub struct CustomShaderState;

impl SimpleState for CustomShaderState {
    fn on_start(&mut self, data: StateData<'_, GameData<'_, '_>>) {
        initialise_camera(data.world);
        let sprite_sheet = load_sprite_sheet(data.world,"texture/logo.png","texture/logo.ron");

        println!("her");

        data.world.create_entity()
            .with(SpriteRender{sprite_sheet, sprite_number: 0})
            .with(Transform::default())
            .build();
    }

}

fn main() -> amethyst::Result<()> {
    amethyst::start_logger(Default::default());

    let app_root = application_root_dir()?;
    let display_config_path = app_root.join("examples/multipass/config/display.ron");
    let assets_dir = app_root.join("examples/multipass/assets/");

    let game_data = GameDataBuilder::default()
        // Add the transform bundle which handles tracking entity positions
        .with_bundle(TransformBundle::new())?
        .with_bundle(
            RenderingBundle::<DefaultBackend>::new()
                // The RenderToWindow plugin provides all the scaffolding for opening a window and
                // drawing on it
                .with_plugin(
                    RenderToWindow::from_config_path(display_config_path)?
                        .with_clear([0.0, 0.0, 0.0, 1.0]),
                )
                // RenderFlat2D plugin is used to render entities with `SpriteRender` component.

                .with_plugin(RenderFlat2D::default().with_target(Target::Custom("Logo")))
                .with_plugin(RenderCustom::default()),

        )?;

    let mut game = Application::new(assets_dir, CustomShaderState, game_data)?;
    game.run();
    Ok(())
}


fn load_sprite_sheet(world: &mut World, png_path: &str, ron_path: &str) -> SpriteSheetHandle {
    let texture_handle = {
        let loader = world.read_resource::<Loader>();
        let texture_storage = world.read_resource::<AssetStorage<Texture>>();
        loader.load(png_path, ImageFormat::default(), (), &texture_storage)
    };
    let loader = world.read_resource::<Loader>();
    let sprite_sheet_store = world.read_resource::<AssetStorage<SpriteSheet>>();
    loader.load(
        ron_path,
        SpriteSheetFormat(texture_handle),
        (),
        &sprite_sheet_store,
    )
}

fn initialise_camera(world: &mut World) {
    // Setup camera in a way that our screen covers whole arena and (0, 0) is in the bottom left.
    let mut transform = Transform::default();
    transform.set_translation_xyz(0.0,0.0, 1.0);

    let cam = world
        .create_entity()
        .with(Camera::standard_2d(
            500.0 ,
            500.0,
        ))
        .with(transform)
        .build();

    let mut act_cam = world.write_resource::<ActiveCamera>();
    (*act_cam).entity = Some(cam);
}
