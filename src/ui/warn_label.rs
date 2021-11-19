pub trait WarnLabel {
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
