mod warn_label;
use warn_label::WarnLabel;

pub struct UiApp {
    pub player_pos: nalgebra::Point3<f32>,
}

impl Default for UiApp {
    fn default() -> Self {
        Self {
            player_pos: [0.0f32; 3].into(),
        }
    }
}

impl UiApp {
    fn bar_contents(&mut self, ui: &mut egui::Ui, frame: &mut epi::Frame<'_>) {
        ui.horizontal_wrapped(|ui| {
            ui.with_layout(egui::Layout::left_to_right(), |ui| {
                ui.warn_label("frame time: ");
                ui.warn_label(format!("{:.8}", frame.info().cpu_usage.unwrap() * 1000.0));
                ui.warn_label("ms");
            });
        });
    }

    fn player_bar_contents(&mut self, ui: &mut egui::Ui, _frame: &mut epi::Frame<'_>) {
        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.warn_label(format!("x: {:.4}", self.player_pos.x));
            ui.warn_label(format!("y: {:.4}", self.player_pos.y));
            ui.warn_label(format!("z: {:.4}", self.player_pos.z));
        });
    }
}

impl epi::App for UiApp {
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
        egui::SidePanel::left("egui_players")
            .frame(egui::Frame {
                fill: egui::Color32::TRANSPARENT,
                ..Default::default()
            })
            .show(ctx, |ui| {
                egui::trace!(ui);
                self.player_bar_contents(ui, frame);
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
