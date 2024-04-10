use nalgebra::DMatrix;

#[derive(Debug)]
struct GaussianProcess<K>
where
    K: Kernel
{
    x_train:    Vec<f64>,
    y_train:    Vec<f64>,
    kernel:     K,
    cov_matrix: Option<DMatrix<f64>>
}

pub trait Kernel {
    fn apply(&self, x1: &f64, x2: &f64) -> f64;
}

#[derive(Debug)]
struct RBF {
    length: f64
}

impl Kernel for RBF {
    fn apply(&self, x1: &f64, x2: &f64) -> f64 {
         (-0.5 * (x1 - x2).abs().powi(2) / self.length.powi(2)).exp()
    }
}

impl RBF {
    pub fn new(length: f64) -> Self {
        RBF { length }
    }
}

#[derive(Debug)]
struct Periodic {
    length: f64,
    period: f64,
}

impl Kernel for Periodic {
    fn apply(&self, x1: &f64, x2: &f64) -> f64 {
        (-1.0 * (std::f64::consts::PI * (x1 - x2).abs() / self.period).sin().powi(2) / self.length.powi(2)).exp()
    }
}

#[derive(Debug)]
struct Linear {
    offset: f64,
    sigma:  f64
}

impl Kernel for Linear {
    fn apply(&self, x1: &f64, x2: &f64) -> f64 {
        self.sigma + (x1 - self.offset) * (x2 - self.offset)
    }
}

impl<K> GaussianProcess<K>
where
    K: Kernel
{
    pub fn new(x_train: Vec<f64>, y_train: Vec<f64>, kernel: K) -> Self {
        let mut gp = GaussianProcess {
            x_train, y_train, kernel, cov_matrix: None
        };
        gp.covariance();

        gp
    }

    fn covariance(&mut self) {
        let mut cov_matrix = DMatrix::from_element(self.x_train.len(), self.x_train.len(), 0.0f64);
        for (i, xn) in self.x_train.iter().enumerate() {
            for (j, xm) in self.x_train[(i+1)..].iter().enumerate() {
                cov_matrix[(i,j)] = self.kernel.apply(xn, xm);
            }
        }
        self.cov_matrix = Some(cov_matrix.clone());
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_gaussian_works() {
        let x_train: Vec<f64> = vec![0.5; 5];
        let y_train: Vec<f64> = vec![0.1; 5];
        let kernel: RBF = RBF::new(1.0);

        let process: GaussianProcess<RBF> = GaussianProcess::new(x_train, y_train, kernel);

        println!("{:?}", process);

        assert_eq!(0, 1);
    }
}
