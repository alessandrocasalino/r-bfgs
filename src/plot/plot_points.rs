use plotters::prelude::*;
use crate::MinimizationResult;
use crate::settings::MinimizationAlgorithm;


pub(super) fn plot_points(result: &&MinimizationResult, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let border_factor = 0.1;

    let data: Vec<(f64, f64, usize)> = result.history.iter().map(|x| (x.x[0], x.x[1], x.k)).collect();

    let max_x: f64 = (border_factor + 1.) * data.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap().0;
    let min_x: f64 = (1. - border_factor) * data.iter().max_by(|a, b| b.0.total_cmp(&a.0)).unwrap().0;
    let max_y: f64 = (border_factor + 1.) * data.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().1;
    let min_y: f64 = (1. - border_factor) * data.iter().max_by(|a, b| b.1.total_cmp(&a.1)).unwrap().1;

    let root_drawing_area = BitMapBackend::new(path, (1000, 600))
        .into_drawing_area();

    root_drawing_area.fill(&WHITE)?;

    // Find the name of the plot accordingly to the minimization algorithm
    let mut caption = String::from("Minimization History");
    match result.minimization_algorithm {
        MinimizationAlgorithm::Bfgs => {
            caption.push_str(" - BFGS");
        },
        MinimizationAlgorithm::Lbfgs => {
            caption.push_str(" - LBFGS");
        },
        MinimizationAlgorithm::GradientDescent => {
            caption.push_str(" - Gradient Descent");
        },
        MinimizationAlgorithm::BfgsBackup => {
            caption.push_str(" - BFGS Backup");
        },
    }

    let mut ctx = ChartBuilder::on(&root_drawing_area)
        .caption(caption, ("Arial", 30))
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    ctx.configure_mesh().draw()?;

    // Plot the connection between the points
    ctx.draw_series(
        LineSeries::new(data.iter().map(|point| (point.0, point.1)), &RED)
    )?;

    // Plot the points
    ctx.draw_series(PointSeries::of_element(
        data.iter().map(|point| (point.0, point.1, point.2)),
        5,
        &BLUE,
        &|c, s, st| {
            return EmptyElement::at((c.0,c.1))    // We want to construct a composed element on-the-fly
                + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
                + Text::new(format!("{:?}", c.2), (10, 0), ("sans-serif", 15).into_font());
        },
    ))?;

    Ok(())
}
