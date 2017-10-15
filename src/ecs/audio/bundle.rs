//! ECS audio bundles

use core::bundle::{ECSBundle, Result};

use audio::Dj;
use audio::output::{default_output, Output};
use ecs::{World, DispatcherBuilder};
use ecs::audio::DjSystem;
use shred::ResourceId;

/// DJ bundle
///
/// Will only register the `Dj` and the `DjSystem` if an audio output is found.
/// `DjSystem` will be registered with name "dj_system".
///
/// ## Errors
///
/// No errors returned by this bundle
///
/// ## Panics
///
/// Panics during `DjSystem` registration if the bundle is applied twice.
///
pub struct DjBundle<'a> {
    dep: &'a [&'a str],
}

impl<'a> DjBundle<'a> {
    /// Create a new DJ bundle
    pub fn new() -> Self {
        Self { dep: &[] }
    }

    /// Set dependencies for the `DjSystem`
    pub fn with_dep(mut self, dep: &'a [&'a str]) -> Self {
        self.dep = dep;
        self
    }
}

impl<'a, 'b, 'c> ECSBundle<'a, 'b> for DjBundle<'c> {
    fn build(
        &self,
        world: &mut World,
        mut builder: DispatcherBuilder<'a, 'b>,
    ) -> Result<DispatcherBuilder<'a, 'b>> {
        // Remove option here when specs get support for optional fetch in
        // released version
        if !world
            .res
            .has_value(ResourceId::new::<Option<Output>>())
        {
            world.add_resource(default_output());
        }

        let dj = world
            .read_resource::<Option<Output>>()
            .as_ref()
            .map(|audio_output| Dj::new(audio_output));

        if let Some(dj) = dj {
            world.add_resource(dj);
            builder = builder.add(DjSystem, "dj_system", self.dep);
        }

        Ok(builder)
    }
}