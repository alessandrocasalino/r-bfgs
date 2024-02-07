use crate::MinimizationResult;

pub(crate) mod history;
pub(crate) mod plot_points;

impl MinimizationResult {
    /// Plot the minimization history points
    /// The minimization history is plotted in a 2D graph, where the x and y axis are the first and second
    /// dimensions of the minimization result, respectively. The points are plotted in blue, and the number
    /// of the iteration is plotted in the same position of the point.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the plot
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    ///
    /// use bfgs::settings::MinimizationAlgorithm;
    ///
    /// // Global minimum: [1., 3.]
    /// fn booth(r: &[f64], _g: &[f64], f: &mut f64, _d: i32) {
    ///     let x = r[0];
    ///     let y = r[1];
    ///     *f = (x + 2. * y - 7.) * (x + 2. * y - 7.) + (2. * x + y - 5.) * (2. * x + y - 5.);
    /// }
    ///
    /// fn main() {
    ///     let mut settings: bfgs::settings::Settings = Default::default();
    ///     settings.minimization = MinimizationAlgorithm::Bfgs;
    ///     settings.save_history = true;
    ///
    ///     let x = vec![2., 3.];
    ///     let result = bfgs::get_minimum(&booth, &x, &settings).expect("Result not found");
    ///     let cmp = vec![1., 3.];
    ///     float_eq::assert_float_eq!(result.x, cmp, rmax_all <= 0.01);
    ///
    ///     let path = Path::new("test.png");
    ///     result.plot_points(path).expect("Plot error");
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// If the minimization history is empty, an error is returned.
    ///
    /// If the minimization result does not have 2 dimensions, an error is returned.
    ///
    /// If the plot cannot be saved, an error is returned.
    ///
    /// # Panics
    ///
    /// If the minimization algorithm is not recognized, a panic will occur.
    ///
    /// If the plot cannot be saved, a panic will occur.
    ///
    /// # Warnings
    ///
    /// The minimization history is saved in the minimization result only if the settings.save_history is set to true.
    /// Enable this option to have the history available for plotting.
    ///
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
