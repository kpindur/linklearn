
/// Enumeration representing different types of kernel functions
pub enum Kernel {
    /// Radial basis function (RBF) kernel.
    /// It takes a single parameter representing the width or scale of the kernel
    Radial(f64),
    /// Periodic kernel.
    /// It takes a typle of two paramters representing the length scale and the periodicity of the kernel
    Periodic((f64, f64)),
    /// Linear kernel.
    /// It takes a tuple of parameters representing the slop and intercept of the linear function
    Linear((f64, f64))
}

/// Applies the specified kernel function to the input data point.
///
/// # Args:
/// + 'xs' - A tuple representing the input data point
/// + 'kernel' - The kernel function to be applied
///
/// # Return:
/// + The result of applying the kernel function to the input data point
/// 
/// # Examples:
/// ```
/// let xs = (0.5, 0.8);
/// let radial_kernel = Kernel::Radial(0.5);
/// let result = apply(&xs, radial_kernel);
/// println!("Result: {}", result);
/// ```
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

