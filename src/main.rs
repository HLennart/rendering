use std::fmt::Debug;

use anyhow::Context;
use camera::CameraUniform;
use cgmath::prelude::*;
use egui::FontDefinitions;
use egui_winit_platform::PlatformDescriptor;
use epi::App;
use instance::Instance;
use legion::{system, Query};
use light::LightUniform;
use log::error;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
};
// use winit::event::Event::*;
use winit::event::Event::WindowEvent;
mod ui;
mod camera;
mod model;
mod texture;
mod state;
mod instance;
mod light;

use model::{DrawLight, DrawModel};

use crate::state::State;










#[derive(Debug)]
struct InstanceInformation {
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
}

type CameraRenderInformation = (CameraUniform, CameraBuffer, CameraBindGroup);




#[derive(Debug)]
pub struct LightInformation {
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
}

struct CameraBuffer(wgpu::Buffer);
struct CameraBindGroup(wgpu::BindGroup);





struct ScaleFactor(f64);



struct DeltaTime(std::time::Duration);
struct InstantApplicationStart(std::time::Instant);

#[tokio::main]
async fn main() {
    env_logger::init();
    let event_loop = EventLoop::with_user_event();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .with_decorations(true)
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await; // NEW!

    let platform = egui_winit_platform::Platform::new(PlatformDescriptor {
        physical_width: state.size.width as u32,
        physical_height: state.size.height as u32,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    let egui_rpass = {
        let device = state.resources.get::<wgpu::Device>().unwrap();
        let adapter = state.resources.get::<wgpu::Adapter>().unwrap();
        let surface_format = state
            .resources
            .get::<wgpu::Surface>()
            .unwrap()
            .get_preferred_format(&adapter)
            .unwrap();
        egui_wgpu_backend::RenderPass::new(&device, surface_format, 1)
    };

    let repaint_signal = std::sync::Arc::new(RenderRepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));

    let app = ui::UiApp::default();

    state.resources.insert(EguiResources {
        egui_rpass,
        egui_platform: platform,
        start_time: InstantApplicationStart(std::time::Instant::now()),
        repaint_signal,
        size: state.size,
        app,
    });

    let mut last_render_time = std::time::Instant::now();

    let mut schedule = legion::Schedule::builder()
        .add_system(update_system())
        .flush()
        .add_system(render_system())
        .build();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        state.resources.get_mut::<EguiResources>().unwrap().egui_platform.handle_event(&event);
        match event {
            winit::event::Event::RedrawRequested(..) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                state.resources.insert(DeltaTime(dt));
                last_render_time = now;

                schedule.execute(&mut state.world, &mut state.resources);
            }
            winit::event::Event::MainEventsCleared | winit::event::Event::UserEvent(Event::RequestRedraw) => {
                window.request_redraw();
            }
            winit::event::Event::DeviceEvent {
                ref event,
                .. // We're not using device_id currently
            } => {
                state.input(event);
            }
            WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    winit::event::WindowEvent::CloseRequested
                    | winit::event::WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed                                ,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    winit::event::WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            // Event::RedrawRequested(_) => {
                
            // }
            _ => {}
        }
    });
}

#[allow(clippy::too_many_arguments)]
#[system]
fn update(
    world: &mut legion::world::SubWorld,
    query: &mut Query<&mut LightInformation>,
    #[resource] DeltaTime(dt): &DeltaTime,
    #[resource] queue: &wgpu::Queue,
    #[resource] camera: &mut camera::Camera,
    #[resource] camera_controller: &mut camera::CameraController,
    #[resource] (camera_uniform, camera_buffer, _): &mut CameraRenderInformation,
    #[resource] projection: &camera::Projection,
    #[resource] EguiResources { app, .. }: &mut EguiResources,
) {
    camera_controller.update_camera(camera, *dt);
    camera_uniform.update_view_proj(camera, projection);

    queue.write_buffer(
        &camera_buffer.0,
        0,
        bytemuck::cast_slice(&[*camera_uniform]),
    );

    app.player_pos = camera.position;

    query.iter_mut(world).for_each(|light_info| {
        //Update the light
        let old_position: cgmath::Vector3<_> = light_info.light_uniform.position.into();
        light_info.light_uniform.position =
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
                * old_position)
                .into();
        queue.write_buffer(
            &light_info.light_buffer,
            0,
            bytemuck::cast_slice(&[light_info.light_uniform]),
        );
    })
}

enum Event {
    RequestRedraw,
}

struct RenderRepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<Event>>);

impl epi::RepaintSignal for RenderRepaintSignal {
    fn request_repaint(&self) {
        self.0.lock().unwrap().send_event(Event::RequestRedraw).ok();
    }
}

struct EguiResources {
    egui_platform: egui_winit_platform::Platform,
    egui_rpass: egui_wgpu_backend::RenderPass,
    start_time: InstantApplicationStart,
    repaint_signal: std::sync::Arc<RenderRepaintSignal>,
    size: winit::dpi::PhysicalSize<u32>,
    app: ui::UiApp,
}

#[allow(clippy::too_many_arguments)]
#[system]
fn render(
    world: &mut legion::world::SubWorld,
    light_query: &mut Query<&LightInformation>,
    instance_query: &mut Query<(&InstanceInformation, &legion::Entity)>,
    model_query: &mut Query<&model::Model>,
    #[resource] surface: &wgpu::Surface,
    #[resource] device: &wgpu::Device,
    #[resource] depth_texture: &texture::Texture,
    #[resource] (_, _, camera_bind_group): &CameraRenderInformation,
    #[resource] render_pipeline: &wgpu::RenderPipeline,
    #[resource] queue: &wgpu::Queue,
    #[resource] EguiResources {
        egui_platform,
        egui_rpass,
        start_time,
        repaint_signal,
        size,
        app,
    }: &mut EguiResources,
    #[resource] DeltaTime(dt): &DeltaTime,
) {
    let result: anyhow::Result<()> = (|| {
        let output = surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            let light_info = light_query
                .iter(world)
                .next()
                .ok_or_else(|| anyhow::anyhow!("No light information"))?;

            for (instance_info, obj_model_entity) in instance_query.iter(world) {
                let obj_model = {
                    model_query
                        .get(world, *obj_model_entity)
                        .with_context(|| anyhow::anyhow!("No model information"))?
                };
                render_pass.set_vertex_buffer(1, instance_info.instance_buffer.slice(..));
                render_pass.set_pipeline(&light_info.light_render_pipeline);
                render_pass.draw_light_model(
                    obj_model,
                    &camera_bind_group.0,
                    &light_info.light_bind_group,
                );

                render_pass.set_pipeline(render_pipeline);
                render_pass.draw_model_instanced(
                    obj_model,
                    0..instance_info.instances.len() as u32,
                    &camera_bind_group.0,
                    &light_info.light_bind_group,
                );
            }
        }

        {
            egui_platform.begin_frame();

            let mut app_output = epi::backend::AppOutput::default();

            let mut frame = epi::backend::FrameBuilder {
                info: epi::IntegrationInfo {
                    web_info: None,
                    cpu_usage: Some(dt.as_secs_f32()),
                    name: "Test",
                    native_pixels_per_point: Some(start_time.0.elapsed().as_secs_f32()),
                    prefer_dark_mode: Some(true),
                },
                tex_allocator: egui_rpass,
                output: &mut app_output,
                repaint_signal: repaint_signal.clone(),
            }
            .build();

            app.update(&egui_platform.context(), &mut frame);

            let (_output, paint_commands) = egui_platform.end_frame(None); // NOTE: Some(window) should be correct here
            let paint_jobs = egui_platform.context().tessellate(paint_commands);

            let screen_descriptor = egui_wgpu_backend::ScreenDescriptor {
                physical_width: size.width,
                physical_height: size.height,
                scale_factor: 1.0, // NOTE: THIS WORKS, BUT IS TECHNICALLY NOT CORRECT
            };

            egui_rpass.update_texture(device, queue, &egui_platform.context().texture());
            egui_rpass.update_user_textures(device, queue);
            egui_rpass.update_buffers(device, queue, &paint_jobs, &screen_descriptor);

            egui_rpass
                .execute(&mut encoder, &view, &paint_jobs, &screen_descriptor, None)
                .unwrap();
        }
        queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    })();
    if let Err(e) = result {
        error!("{}", e);
    }
}
