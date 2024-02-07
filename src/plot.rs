use crate::MinimizationResult;

pub(crate) mod history;
pub(crate) mod plot_points;

impl MinimizationResult {
    /// Plot the minimization history points
    pub fn plot_points(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        if self.history.is_empty() {
            Err("No history to plot. Maybe you did not set settings.save_history?")?;
        }
        // TODO: Add choice for which dimensions to plot
        if self.x.len() != 2 {
            Err("The minimization result must have 2 dimensions to be plotted")?;
        }
        plot_points::plot_points(&self, path)
    }
}
