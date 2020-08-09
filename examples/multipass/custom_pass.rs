use amethyst::{
    core::ecs::{
        Component, DenseVecStorage, DispatcherBuilder, Join, ReadStorage, SystemData, World,
    },
    prelude::*,
    renderer::{
        bundle::{RenderOrder, RenderPlan, RenderPlugin, Target},
        pipeline::{PipelineDescBuilder, PipelinesBuilder},
        rendy::{
            command::{QueueId, RenderPassEncoder},
            factory::Factory,
            graph::{
                render::{PrepareResult, RenderGroup, RenderGroupDesc},
                GraphContext, NodeBuffer, NodeImage,
            },
            hal::{self, device::Device, format::Format, pso, pso::ShaderStageFlags},
            mesh::{AsVertex, VertexFormat},
            shader::{Shader, SpirvShader},
        },
        submodules::{DynamicUniform, DynamicVertexBuffer},
        types::Backend,
        util, ChangeDetection,
    },
};

use amethyst_error::Error;
use derivative::Derivative;
use glsl_layout::*;
use amethyst_rendy::rendy::graph::ImageId;
use amethyst_rendy::bundle::{TargetImage, TargetPlanOutputs, OutputColor, ImageOptions};
use amethyst_rendy::rendy::hal::command::{ClearDepthStencil, ClearValue};
use amethyst_rendy::Kind;
use amethyst_rendy::submodules::TextureSub;
use amethyst_rendy::rendy::texture::image::{load_from_image, ImageTextureConfig, TextureKind, Repr};
use amethyst_rendy::rendy::resource::{SamplerInfo, Filter, WrapMode, Image, Handle as RendyHandle, ImageView, ImageViewInfo, SubresourceRange, ViewKind, DescriptorSetLayout, Escape, DescriptorSet};
use failure::_core::num::Wrapping;


lazy_static::lazy_static! {
    // These uses the precompiled shaders.
    // These can be obtained using glslc.exe in the vulkan sdk.
    static ref VERTEX: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("./assets/shaders/compiled/vertex/custom.vert.spv"),
        ShaderStageFlags::VERTEX,
        "main",
    ).unwrap();

    static ref FRAGMENT: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("./assets/shaders/compiled/fragment/custom.frag.spv"),
        ShaderStageFlags::FRAGMENT,
        "main",
    ).unwrap();
}

/// Example code of using a custom shader
///
/// Requires "shader-compiler" flag
///
/// ''' rust
/// use std::path::PathBuf;
/// use amethyst::renderer::rendy::shader::{PathBufShaderInfo, ShaderKind, SourceLanguage};
///
///  lazy_static::lazy_static! {
///     static ref VERTEX: SpirvShader = PathBufShaderInfo::new(
///         PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/shaders/src/vertex/custom.vert")),
///         ShaderKind::Vertex,
///         SourceLanguage::GLSL,
///        "main",
///     ).precompile().unwrap();
///
///     static ref FRAGMENT: SpirvShader = PathBufShaderInfo::new(
///         PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/shaders/src/fragment/custom.frag")),
///         ShaderKind::Fragment,
///         SourceLanguage::GLSL,
///         "main",
///     ).precompile().unwrap();
/// }
/// '''

/// Draw triangles.
#[derive(Clone, Debug, PartialEq)]

pub struct DrawCustomDesc{
    image_ID: ImageId
}

impl DrawCustomDesc {
    /// Create instance of `DrawCustomDesc` render group
    pub fn new(image_ID: ImageId) -> Self { Self { image_ID } }
}


impl<B: Backend> RenderGroupDesc<B, World> for DrawCustomDesc {
    fn build(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _world: &World,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: hal::pass::Subpass<'_, B>,
        _buffers: Vec<NodeBuffer>,
        _images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, World>>, failure::Error> {
        let vertex = DynamicVertexBuffer::new();

        // get view on offscreen image
        let image = ctx.get_image(self.image_ID).unwrap();



        let view = factory.create_image_view(image.clone(), ImageViewInfo {
            view_kind: ViewKind::D2,
            format:hal::format::Format::Rgba8Unorm,
            swizzle:hal::format::Swizzle::NO,
            range: SubresourceRange {
                aspects:hal::format::Aspects::COLOR,
                levels:0..1,
                layers:0..1,
            }
        }).unwrap();

        // setup the offscreen texture descriptor set
        let texture_layout:RendyHandle<DescriptorSetLayout<B>> = RendyHandle::from(
            factory
            .create_descriptor_set_layout(vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: pso::ShaderStageFlags::FRAGMENT,
                immutable_samplers: true,
            }])
            .unwrap()
        );

        let texture_set = factory.create_descriptor_set(texture_layout.clone()).unwrap();

        // write to the texture description set

        unsafe {
            factory.device().write_descriptor_sets(vec![
                hal::pso::DescriptorSetWrite {
                    set: texture_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        view.raw(),
                        hal::image::Layout::ColorAttachmentOptimal,
                    ))
                }
            ]);
        }



        let (pipeline, pipeline_layout) = build_custom_pipeline(
            factory,
            subpass,
            framebuffer_width,
            framebuffer_height,
            vec![texture_layout.raw()],
        )?;

        Ok(Box::new(DrawCustom::<B> {
            pipeline,
            pipeline_layout,
            vertex,
            vertex_count: 0,
            change: Default::default(),
            texture_set,
            view
        }))
    }
}

/// Draws triangles to the screen.
#[derive(Debug)]
pub struct DrawCustom<B: Backend> {
    pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,
    vertex: DynamicVertexBuffer<B, CustomArgs>,
    vertex_count: usize,
    change: ChangeDetection,
    texture_set: Escape<DescriptorSet<B>>,
    view: Escape<ImageView<B>>,
}

impl<B: Backend> RenderGroup<B, World> for DrawCustom<B> {
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        world: &World,
    ) -> PrepareResult {



        //Update vertex count and see if it has changed
        let old_vertex_count = self.vertex_count;
        self.vertex_count = 4;
        let changed = old_vertex_count != self.vertex_count;

        // Create an iterator over the Triangle vertices
        let vertex_data = [
           CustomArgs{ pos: vec2::from([-1.0,-1.0])},
           CustomArgs{ pos: vec2::from([-1.0,1.0])},
           CustomArgs{ pos: vec2::from([1.0,-1.0])},
           CustomArgs{ pos: vec2::from([ 1.0,1.0])}];

        // Write the vector to a Vertex buffer
        self.vertex.write(
            factory,
            index,
            self.vertex_count as u64,
            vec![ vertex_data]
        );


        // Return with we can reuse the draw buffers using the utility struct ChangeDetection
        self.change.prepare_result(index, changed)
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        _world: &World,
    ) {
        // Don't worry about drawing if there are no vertices. Like before the state adds them to the screen.
        if self.vertex_count == 0 {
            return;
        }

        // Bind the pipeline to the the encoder
        encoder.bind_graphics_pipeline(&self.pipeline);



        // Bind the vertex buffer to the encoder
        self.vertex.bind(index, 0, 0, &mut encoder);

        // Draw the vertices
        unsafe {
            encoder.bind_graphics_descriptor_sets(&self.pipeline_layout,0, Some(self.texture_set.raw()), std::iter::empty());
            encoder.draw(0..self.vertex_count as u32, 0..1);
        }
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, _world: &World) {
        unsafe {
            factory.device().destroy_graphics_pipeline(self.pipeline);
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
        }
    }
}

fn build_custom_pipeline<B: Backend>(
    factory: &Factory<B>,
    subpass: hal::pass::Subpass<'_, B>,
    framebuffer_width: u32,
    framebuffer_height: u32,
    layouts: Vec<&B::DescriptorSetLayout>,
) -> Result<(B::GraphicsPipeline, B::PipelineLayout), failure::Error> {
    let pipeline_layout = unsafe {
        factory
            .device()
            .create_pipeline_layout(layouts, None as Option<(_, _)>)
    }?;

    // Load the shaders
    let shader_vertex = unsafe { VERTEX.module(factory).unwrap() };
    let shader_fragment = unsafe { FRAGMENT.module(factory).unwrap() };

    // Build the pipeline
    let pipes = PipelinesBuilder::new()
        .with_pipeline(
            PipelineDescBuilder::new()
                // This Pipeline uses our custom vertex description and does not use instancing
                .with_vertex_desc(&[(CustomArgs::vertex(), pso::VertexInputRate::Vertex)])
                .with_input_assembler(pso::InputAssemblerDesc::new(hal::Primitive::TriangleStrip))
                // Add the shaders
                .with_shaders(util::simple_shader_set(
                    &shader_vertex,
                    Some(&shader_fragment),
                ))
                .with_layout(&pipeline_layout)
                .with_subpass(subpass)
                .with_framebuffer_size(framebuffer_width, framebuffer_height)
                // We are using alpha blending
                .with_blend_targets(vec![pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                }]),
        )
        .build(factory, None);

    // Destoy the shaders once loaded
    unsafe {
        factory.destroy_shader_module(shader_vertex);
        factory.destroy_shader_module(shader_fragment);
    }

    // Handle the Errors
    match pipes {
        Err(e) => {
            unsafe {
                factory.device().destroy_pipeline_layout(pipeline_layout);
            }
            Err(e)
        }
        Ok(mut pipes) => Ok((pipes.remove(0), pipeline_layout)),
    }
}

/// A [RenderPlugin] for our custom plugin
#[derive(Default, Debug)]
pub struct RenderCustom {}

impl<B: Backend> RenderPlugin<B> for RenderCustom {
    fn on_build<'a, 'b>(
        &mut self,
        world: &mut World,
        _builder: &mut DispatcherBuilder<'a, 'b>,
    ) -> Result<(), Error> {
        // Add the required components to the world ECS
        Ok(())
    }

    fn on_plan(
        &mut self,
        plan: &mut RenderPlan<B>,
        _factory: &mut Factory<B>,
        _world: &World,
    ) -> Result<(), Error> {

        let kind = Kind::D2(1 as u32, 1 as u32, 1, 1);
        let depth_options = ImageOptions {
            kind: kind,
            levels: 1,
            format: Format::D32Sfloat,
            clear: Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
        };
        plan.add_root(Target::Custom("Logo"));
        plan.define_pass(
            Target::Custom("Logo"),
            TargetPlanOutputs{
                colors: vec![OutputColor::Image(ImageOptions {
                    kind:kind,
                    levels: 1,
                    format: Format::Rgba8Unorm,
                    clear: Some(ClearValue::Color([0.0, 0.0, 0.0, 1.0].into())),
                })],
                depth: Some(depth_options)
            }
        )?;



        plan.extend_target(Target::Main, |ctx| {
            // Add our Description
            let image_ID = ctx.get_image(TargetImage::Color(Target::Custom("Logo") , 0)).unwrap();
            ctx.add(RenderOrder::Transparent, DrawCustomDesc::new(image_ID).builder())?;
            Ok(())
        });
        Ok(())
    }
}

/// Vertex Arguments to pass into shader.
/// VertexData in shader:
/// layout(location = 0) out VertexData {
///    vec2 pos;
///    vec4 color;
/// } vertex;
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, AsStd140)]
#[repr(C, align(4))]
pub struct CustomArgs {
    /// vec2 pos;
    pub pos: vec2,
}

/// Required to send data into the shader.
/// These names must match the shader.
impl AsVertex for CustomArgs {
    fn vertex() -> VertexFormat {
        VertexFormat::new((
            // vec2 pos;
            (Format::Rg32Sfloat, "pos"),
        ))
    }
}
