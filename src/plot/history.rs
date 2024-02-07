#[derive(Debug, Clone)]
pub struct MinimizationHistoryPoint {
    /// Iteration number
    pub k: usize,
    /// Energy
    pub f: f64,
    /// Gradient
    pub x: Vec<f64>,
    /// Function evaluations
    pub eval: usize,
}