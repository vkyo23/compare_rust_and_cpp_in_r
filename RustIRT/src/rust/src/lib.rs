use extendr_api::prelude::*;
use ndarray::{s, Array1, Array2, ShapeBuilder};

struct RustIRTModel {
    I: usize,
    J: usize,
    alpha: Array1<f64>,
    beta: Array1<f64>,
    theta: Array1<f64>,
    a0: f64,
    A0: f64,
    b0: f64,
    B0: f64,
    theta_constraint: usize,
    theta_strict_identification: bool,
    maxit: usize,
    verbose: usize,
    tol: f64,
    S: Array2<f64>,
    Omega: Array2<f64>,
    alpha_new: Array1<f64>,
    beta_new: Array1<f64>,
    theta_new: Array1<f64>,
    update_histories: Array2<f64>,
    converged: bool,
    convergence_metric: f64,
    iter: usize,
}

impl RustIRTModel {
    pub fn new(
        Y: Array2<f64>,
        alpha: Array1<f64>,
        beta: Array1<f64>,
        theta: Array1<f64>,
        a0: f64,
        A0: f64,
        b0: f64,
        B0: f64,
        theta_constraint: usize,
        theta_strict_identification: bool,
        maxit: usize,
        verbose: usize,
        tol: f64,
    ) -> Self {
        let I: usize = Y.nrows();
        let J: usize = Y.ncols();
        let S: Array2<f64> = &Y - 0.5;
        let Omega: Array2<f64> = Array2::<f64>::zeros((I, J));
        let alpha_new: Array1<f64> = Array1::<f64>::zeros(J);
        let beta_new: Array1<f64> = Array1::<f64>::zeros(J);
        let theta_new: Array1<f64> = Array1::<f64>::zeros(I);
        let update_histories: Array2<f64> = Array2::<f64>::zeros((maxit, 3));
        RustIRTModel {
            I,
            J,
            alpha,
            beta,
            theta,
            a0,
            A0,
            b0,
            B0,
            theta_constraint,
            theta_strict_identification,
            maxit,
            verbose,
            tol,
            S,
            Omega,
            alpha_new,
            beta_new,
            theta_new,
            update_histories,
            converged: false,
            convergence_metric: 1.0,
            iter: 0,
        }
    }

    pub fn fit(&mut self) {
        while self.iter < self.maxit {
            self.update_all_parameters();
            self.save_update_history();
            self.convergence_check();
            if self.converged {
                return;
            }
            if (self.iter + 1) % self.verbose == 0 {
                rprintln!(
                    "  - Iteration {}: eval = {}",
                    self.iter + 1,
                    format!("{:.5e}", self.convergence_metric)
                );
            }
            self.iter += 1;
        }
    }

    fn output(&self) -> List {
        let out_update_histories: Array2<f64> = self.update_histories.slice(s![0..self.iter, ..]).to_owned();
        let update_histories_robj: RMatrix<f64> = array2_to_rmatrix(out_update_histories);

        let modelout: List = list!(
            alpha = r!(self.alpha.to_vec()),
            beta = r!(self.beta.to_vec()),
            theta = r!(self.theta.to_vec()),
            converged = r!(self.converged),
            iteration = r!(self.iter),
            update_histories = r!(update_histories_robj)
        );

        return modelout;
    }

    fn update_all_parameters(&mut self) {
        self.calc_EOmega();
        self.update_alpha();
        self.update_beta();
        self.update_theta();
    }

    fn calc_EOmega(&mut self) {
        for i in 0..self.I {
            for j in 0..self.J {
                let psi: f64 = self.alpha[j] + self.beta[j] * self.theta[i];
                self.Omega[(i, j)] = (psi / 2.0).tanh() / (2.0 * psi);
            }
        }
    }

    fn update_alpha(&mut self) {
        for j in 0..self.J {
            let mut mu_part: f64 = self.a0 / self.A0;
            let mut sig_part: f64 = 1.0 / self.A0;
            for i in 0..self.I {
                mu_part += self.S[(i, j)] - self.Omega[(i, j)] * self.beta[j] * self.theta[i];
                sig_part += self.Omega[(i, j)];
            }
            self.alpha_new[j] = mu_part / sig_part;
        }
    }

    fn update_beta(&mut self) {
        for j in 0..self.J {
            let mut mu_part: f64 = self.b0 / self.B0;
            let mut sig_part: f64 = 1.0 / self.B0;
            for i in 0..self.I {
                mu_part += self.theta[i] * (self.S[(i, j)] - self.Omega[(i, j)] * self.alpha[j]);
                sig_part += self.Omega[(i, j)] * self.theta[i].powf(2.0);
            }
            self.beta_new[j] = mu_part / sig_part;
        }
    }

    fn update_theta(&mut self) {
        for i in 0..self.I {
            let mut mu_part: f64 = 0.0;
            let mut sig_part: f64 = 1.0;
            for j in 0..self.J {
                mu_part += self.beta[j] * (self.S[(i, j)] - self.Omega[(i, j)] * self.alpha[j]);
                sig_part += self.Omega[(i, j)] * self.beta[j].powf(2.0);
            }
            self.theta_new[i] = mu_part / sig_part;
        }

        if self.theta_new[self.theta_constraint] < 0.0 {
            self.theta_new.mapv_inplace(|x: f64| -x);
        }
        if self.theta_strict_identification {
            let mean_theta: f64 = self.theta_new.mean().unwrap();
            let std_theta: f64 = self.theta_new.std(0.0);
            self.theta_new.mapv_inplace(|x: f64| (x - mean_theta) / std_theta);
        }
    }

    fn save_update_history(&mut self) {
        self.update_histories[(self.iter, 0)] = 1.0 - calc_corcoeff(&self.alpha_new, &self.alpha);
        self.update_histories[(self.iter, 1)] = 1.0 - calc_corcoeff(&self.beta_new, &self.beta);
        self.update_histories[(self.iter, 2)] = 1.0 - calc_corcoeff(&self.theta_new, &self.theta);

        self.alpha = self.alpha_new.clone();
        self.beta = self.beta_new.clone();
        self.theta = self.theta_new.clone();
    }

    fn convergence_check(&mut self) {
        self.convergence_metric = self.update_histories
            .row(self.iter)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if self.convergence_metric < self.tol {
            self.converged = true;
        }
    }
}

// Exported function to R
#[extendr]
fn fit_RustIRT(
    Y: RMatrix<i32>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    theta: Vec<f64>,
    a0: f64,
    A0: f64,
    b0: f64,
    B0: f64,
    theta_constraint: usize,
    theta_strict_identification: bool,
    maxit: usize,
    verbose: usize,
    tol: f64,
) -> List {
    let alpha: Array1<f64> = Array1::from_vec(alpha);
    let beta: Array1<f64> = Array1::from_vec(beta);
    let theta: Array1<f64> = Array1::from_vec(theta);
    let Y: Array2<f64> = rmatrix_i32_to_array2_f64(Y);


    let mut model: RustIRTModel = RustIRTModel::new(
        Y,
        alpha,
        beta,
        theta,
        a0,
        A0,
        b0,
        B0,
        theta_constraint,
        theta_strict_identification,
        maxit,
        verbose,
        tol,
    );
    model.fit();
    model.output()
}

fn calc_corcoeff(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n: f64 = x.len() as f64;

    let x_bar: f64 = x.mean().unwrap();
    let y_bar: f64 = y.mean().unwrap();

    let centered_x: Array1<f64> = x - x_bar;
    let centered_y: Array1<f64> = y - y_bar;

    let numerator: f64 = centered_x.dot(&centered_y);

    let x_sigma: f64 = (centered_x.mapv(|v| v * v).sum() / (n - 1.0)).sqrt();
    let y_sigma: f64 = (centered_y.mapv(|v| v * v).sum() / (n - 1.0)).sqrt();

    return numerator / ((n - 1.0) * x_sigma * y_sigma);
}

fn rmatrix_i32_to_array2_f64(mat: RMatrix<i32>) -> Array2<f64> {
    let dim: &[usize; 2] = mat.dim();
    let data: Vec<f64> = mat.data().iter().map(|&x| x as f64).collect();
    return Array2::from_shape_vec((dim[0], dim[1]).f(), data).expect("failed to convert RMatrix to Array2");
}

fn array2_to_rmatrix(arr: Array2<f64>) -> RMatrix<f64> {
    let dim: (usize, usize) = arr.dim();
    return RMatrix::new_matrix(dim.0, dim.1, |i, j| arr[(i, j)])
}
extendr_module! {
    mod RustIRT;
    fn fit_RustIRT;
}
