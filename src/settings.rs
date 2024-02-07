/// Enumerator for minimization algorithm
pub enum MinimizationAlgorithm {
    /// Simple gradient descent algorithm (slow)
    GradientDescent,
    /// BFGS algorithm
    Bfgs,
    /// L-BFGS algorithm (memory efficient implementation of BFGS)
    Lbfgs,
    /// Use L-BFGS algorithm if BFGS fails
    BfgsBackup,
}

/// Enumerator for line search algorithm
pub enum LineSearchAlgorithm {
    /// A line search according to Wolfe's conditions: Armijo and curvature conditions (slower)
    Simple,
    /// A line search using More-Thuente and backtracking algorithm (faster, but might fail in some cases)
    Backtracking,
}

/// Settings struct for the optimization algorithm.
///
/// # Examples
///
/// ```
/// // Create settings with default parameters
/// let mut settings: bfgs::settings::Settings = Default::default();
/// // Choose the minimization algorithm
/// settings.minimization = bfgs::settings::MinimizationAlgorithm::Bfgs;
/// settings.verbose = false;
/// ```
///
pub struct Settings {
    /// Minimization algorithm (Bfgs, Lbfgs, BfgsBackup)
    pub minimization: MinimizationAlgorithm,
    /// Line search algorithm
    pub line_search: LineSearchAlgorithm,

    /// Exit condition
    pub ftol: f64,
    /// Exit condition
    pub gtol: f64,
    /// Exit condition
    pub gmax: f64,

    /// Maximum number of iterations before stopping with "no convergence" error
    pub iter_max: usize,

    /// Sufficient decrease constant (Armijo rule)
    pub mu: f64,
    /// Curvature condition constant
    pub eta: f64,

    // L-BFGS specific options (not used for standard BFGS)
    /// L-BFGS: number of history points
    /// This option is not considered for BFGS
    pub history_depth: usize,
    /// L-BFGS: m1qn3 diagonal matrix compute
    /// This option is not considered for BFGS
    pub m1qn3: bool,

    /// Try to estimate the value of a
    pub estimate_a: bool,
    
    /// Save the history
    pub save_history: bool,

    /// Verbosity
    pub verbose: bool,

    /// Layout with Row major
    pub(crate) layout: cblas::Layout,
    /// Use only the upper matrix part
    pub(crate) part: cblas::Part,
}

// Default trait for Settings
impl Default for Settings {
    fn default() -> Settings {
        Settings {
            minimization: MinimizationAlgorithm::Bfgs,
            line_search: LineSearchAlgorithm::Simple,
            ftol: 1e-6,
            gtol: 1e-14,
            gmax: 1e-14,
            iter_max: 10000,
            mu: 1e-4,
            eta: 0.9,
            history_depth: 10,
            m1qn3: true,
            estimate_a: true,
            save_history: false,
            verbose: false,
            layout: cblas::Layout::RowMajor,
            part: cblas::Part::Upper,
        }
    }
}