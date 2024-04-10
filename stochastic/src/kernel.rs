

pub enum Kernel {
    Radial(f64),
    Periodic((f64, f64)),
    Linear((f64, f64))
}

/// 
pub fn apply(xs: &(f64, f64), kernel: Kernel) -> f64 {
    match kernel {
        Kernel::Radial(length) => rbf(xs, &length),
        Kernel::Periodic((length, period)) => periodic(xs, &length, &period),
        Kernel::Linear((offset, sigma)) => linear(xs, &offset, &sigma),
    }
}

/// Radial Basis Function
/// k(x, x') = (-0.5 * (x1 - x2).abs().powi(2) / self.length.powi(2)).exp()
fn rbf(xs: &(f64, f64), length: &f64) -> f64 {
    let x1 = xs.0;
    let x2 = xs.1;

    (-0.5 * (x1 - x2).abs().powi(2) / length.powi(2)).exp()
}

/// Periodic Function
/// k(x, x') = (-1.0 * (std::f64::consts::PI * (x1 - x2).abs() / self.period).sin().powi(2) / self.length.powi(2)).exp()
fn periodic(xs: &(f64, f64), length: &f64, period: &f64) -> f64 {
    let x1 = xs.0;
    let x2 = xs.1;

    (-1.0 * (std::f64::consts::PI * (x1 - x2).abs() / period).sin().powi(2) / length.powi(2)).exp()
}

/// Linear Function
/// self.sigma + (x1 - self.offset) * (x2 - self.offset)
fn linear(xs: &(f64, f64), offset: &f64, sigma: &f64) -> f64 {
    let x1 = xs.0;
    let x2 = xs.1;

    sigma + (x1 - offset) * (x2 - offset)
}

