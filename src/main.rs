use anyhow::Context;
use cgmath::prelude::*;
use egui::FontDefinitions;
use egui_winit_platform::PlatformDescriptor;
use epi::App;
use legion::{system, IntoQuery, Query, Write};
use log::error;
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
// use winit::event::Event::*;
use winit::event::Event::WindowEvent;
mod camera;
mod model;
mod texture;

use model::{DrawLight, DrawModel, Vertex};

const NUM_INSTANCES_PER_ROW: u32 = 10;

#[derive(Default)]
struct EguiApp {}

trait WarnLabel {
    fn warn_label(&mut self, label: impl ToString) -> egui::Response;
}

impl WarnLabel for egui::Ui {
    #[inline(always)]
    fn warn_label(&mut self, text: impl ToString) -> egui::Response {
        use egui::Widget;

        egui::Label::new(text)
            .text_color(egui::Color32::RED)
            .text_style(egui::TextStyle::Monospace)
            .ui(self)
    }
}

// #[inline(always)]
// pub fn red_label(&mut self, text: impl ToString) -> Response {
//     Label::new(text).ui(self)
// }

impl EguiApp {
    fn bar_contents(&mut self, ui: &mut egui::Ui, frame: &mut epi::Frame<'_>) {
        ui.horizontal_wrapped(|ui| {
            ui.with_layout(egui::Layout::left_to_right(), |ui| {
                ui.warn_label("frame time: ");
                ui.warn_label(format!("{:.8}", frame.info().cpu_usage.unwrap() * 1000.0));
                ui.warn_label("ms");
            });
        });
    }
}

impl epi::App for EguiApp {
    fn update(&mut self, ctx: &egui::CtxRef, frame: &mut epi::Frame<'_>) {
        egui::TopBottomPanel::top("egui_app_top_bar")
            .frame(egui::Frame {
                fill: egui::Color32::TRANSPARENT,
                ..Default::default()
            })
            .show(ctx, |ui| {
                egui::trace!(ui);
                self.bar_contents(ui, frame);
            });
    }

    fn setup(
        &mut self,
        _ctx: &egui::CtxRef,
        _frame: &mut epi::Frame<'_>,
        _storage: Option<&dyn epi::Storage>,
    ) {
        
    }

    fn clear_color(&self) -> egui::Rgba {
        egui::Rgba::TRANSPARENT
    }

    fn name(&self) -> &str {
        "test egui"
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

#[derive(Debug)]
struct LightInformation {
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
}

#[derive(Debug)]
struct InstanceInformation {
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
}

type CameraRenderInformation = (CameraUniform, CameraBuffer, CameraBindGroup);

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();

        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into()
    }
}

#[derive(Debug)]
struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl model::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We don't have to do this in code though.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
}

struct CameraBuffer(wgpu::Buffer);
struct CameraBindGroup(wgpu::BindGroup);

struct State {
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    #[allow(dead_code)]
    debug_material: model::Material,
    mouse_pressed: bool,
    world: legion::World,
    resources: legion::Resources,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{:?}", shader)),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLAMPING
            clamp_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    })
}

struct ScaleFactor(f64);

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                    // normal map
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let camera = camera::Camera::new([0.0, 5.0, 10.0], cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection = camera::Projection::new(config.width, config.height, 45.0, 0.1, 100.0);
        let camera_controller = camera::CameraController::new(4.0, 0.4);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .into_par_iter() // NEW!
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).into_par_iter().map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                    let position = cgmath::Vector3 { x, y: 0.0, z };

                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                    };

                    Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        let obj_model = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            res_dir.join("cube.obj"),
        )
        .unwrap();

        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
        };

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        let debug_material = {
            let diffuse_bytes = include_bytes!("../res/cobble-diffuse.png");
            let normal_bytes = include_bytes!("../res/cobble-normal.png");

            let diffuse_texture = texture::Texture::from_bytes(
                &device,
                &queue,
                diffuse_bytes,
                "res/alt-diffuse.png",
                false,
            )
            .unwrap();
            let normal_texture = texture::Texture::from_bytes(
                &device,
                &queue,
                normal_bytes,
                "res/alt-normal.png",
                true,
            )
            .unwrap();

            model::Material::new(
                &device,
                "alt-material",
                diffuse_texture,
                normal_texture,
                &texture_bind_group_layout,
            )
        };

        let mut world = legion::World::default();
        let mut resources = legion::Resources::default();

        resources.insert(camera);
        resources.insert(camera_controller);
        resources.insert(projection);
        resources.insert((
            camera_uniform,
            CameraBuffer(camera_buffer),
            CameraBindGroup(camera_bind_group),
        ));
        resources.insert(queue);
        resources.insert(surface);
        resources.insert(device);
        resources.insert(adapter);
        resources.insert(depth_texture);
        resources.insert(render_pipeline);
        resources.insert(ScaleFactor(window.scale_factor()));

        let obj_model_id = world.push((obj_model,));

        world.push((LightInformation {
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
        },));

        world.push((
            InstanceInformation {
                instances,
                instance_buffer,
            },
            obj_model_id,
        ));

        Self {
            config,
            size,
            #[allow(dead_code)]
            debug_material,
            mouse_pressed: false,
            world,
            resources,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            let mut query = Write::<camera::Projection>::query();
            query.iter_mut(&mut self.world).for_each(|projection| {
                projection.resize(new_size.width, new_size.height);
            });
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            let device = match self.resources.get::<wgpu::Device>() {
                Some(device) => device,
                None => {
                    error!("could not find device in resources");
                    return;
                }
            };

            if let Some(surface) = self.resources.get::<wgpu::Surface>() {
                surface.configure(&device, &self.config)
            };

            if let Some(mut texture) = self.resources.get_mut::<texture::Texture>() {
                *texture =
                    texture::Texture::create_depth_texture(&device, &self.config, "depth_texture");
            }

            if let Some(mut res) = self.resources.get_mut::<EguiResources>() {
                res.size = new_size;
            }
        }
    }

    fn input(&mut self, event: &DeviceEvent) -> bool {
        let mut camera_controller = self
            .resources
            .get_mut::<camera::CameraController>()
            .unwrap();

        match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(key),
                state,
                ..
            }) => camera_controller.process_keyboard(*key, *state),
            DeviceEvent::MouseWheel { delta, .. } => {
                camera_controller.process_scroll(delta);
                true
            }
            DeviceEvent::Button {
                button: 1, // Left Mouse Button
                state,
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            DeviceEvent::MouseMotion { delta } => {
                if self.mouse_pressed {
                    camera_controller.process_mouse(delta.0, delta.1);
                }
                true
            }
            _ => false,
        }
    }
}

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

    let app = EguiApp::default();

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
) {
    camera_controller.update_camera(camera, *dt);
    camera_uniform.update_view_proj(camera, projection);

    queue.write_buffer(
        &camera_buffer.0,
        0,
        bytemuck::cast_slice(&[*camera_uniform]),
    );

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
    app: EguiApp,
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
