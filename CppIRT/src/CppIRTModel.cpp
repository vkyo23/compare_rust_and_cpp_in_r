//[[Rcpp::depends(RcppArmadillo)]]
#include "CppIRTModel.h"

CppIRTModel::CppIRTModel(const arma::mat &Y, arma::vec alpha, arma::vec beta,
                         arma::vec theta, const double &a0, const double &A0,
                         const double &b0, const double &B0,
                         const int &theta_constraint,
                         const bool &theta_strict_identification,
                         const int &maxit, const int &verbose,
                         const double &tol)
    : I(Y.n_rows),
      J(Y.n_cols),
      alpha(alpha),
      beta(beta),
      theta(theta),
      a0(a0),
      A0(A0),
      b0(b0),
      B0(B0),
      theta_constraint(theta_constraint),
      maxit(maxit),
      verbose(verbose),
      theta_strict_identification(theta_strict_identification),
      tol(tol) {
  S = Y - 0.5;
  Omega = arma::mat(I, J);
  alpha_new = arma::vec(J);
  beta_new = arma::vec(J);
  theta_new = arma::vec(I);
  update_histories = arma::mat(maxit, 3);
  converged = false;
  convergence_metric = 1.0;
  iter = 0;
}

CppIRTModel::~CppIRTModel() {}

//[[Rcpp::export]]
Rcpp::List fit_CppIRT(const arma::mat &Y, arma::vec alpha, arma::vec beta,
                      arma::vec theta, const double &a0, const double &A0,
                      const double &b0, const double &B0,
                      const int &theta_constraint,
                      const bool &theta_strict_identification, const int &maxit,
                      const int &verbose, const double &tol) {
  CppIRTModel model =
      CppIRTModel(Y, alpha, beta, theta, a0, A0, b0, B0, theta_constraint,
                  theta_strict_identification, maxit, verbose, tol);

  model.fit();
  Rcpp::List output = model.output();

  return output;
}

Rcpp::List CppIRTModel::output() {
  Rcpp::List modeloutput = Rcpp::List::create(
      Rcpp::Named("alpha") = alpha, Rcpp::Named("beta") = beta,
      Rcpp::Named("theta") = theta, Rcpp::Named("converged") = converged,
      Rcpp::Named("iteration") = iter,
      Rcpp::Named("update_histories") = update_histories.rows(0, iter));

  return modeloutput;
}

void CppIRTModel::fit() {
  for (; iter < maxit; iter++) {
    Rcpp::checkUserInterrupt();

    update_all_parameters();

    save_update_history(iter);
    convergence_check(iter);

    if (converged) return;
    if ((iter + 1) % verbose == 0) {
      Rcpp::Rcout << "  - Iteration " << iter + 1
                  << ": eval = " << convergence_metric << "\n";
    }
  }
}

void CppIRTModel::update_all_parameters() {
  calc_EOmega();
  update_alpha();
  update_beta();
  update_theta();
}

void CppIRTModel::calc_EOmega() {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      double psi = alpha[j] + beta[j] * theta[i];
      Omega(i, j) = std::tanh(psi / 2.0) / (2.0 * psi);
    }
  }
}

void CppIRTModel::update_alpha() {
  for (int j = 0; j < J; j++) {
    double mu_part = a0 / A0;
    double sig_part = 1.0 / A0;
    for (int i = 0; i < I; i++) {
      mu_part += S(i, j) - Omega(i, j) * beta[j] * theta[i];
      sig_part += Omega(i, j);
    }
    alpha_new[j] = mu_part / sig_part;
  }
}

void CppIRTModel::update_beta() {
  for (int j = 0; j < J; j++) {
    double mu_part = b0 / B0;
    double sig_part = 1.0 / B0;
    for (int i = 0; i < I; i++) {
      mu_part += theta[i] * (S(i, j) - Omega(i, j) * alpha[j]);
      sig_part += Omega(i, j) * std::pow(theta[i], 2.0);
    }
    beta_new[j] = mu_part / sig_part;
  }
}

void CppIRTModel::update_theta() {
  for (int i = 0; i < I; i++) {
    double mu_part = 0.0;
    double sig_part = 1.0;
    for (int j = 0; j < J; j++) {
      mu_part += beta[j] * (S(i, j) - Omega(i, j) * alpha[j]);
      sig_part += Omega(i, j) * std::pow(beta[j], 2.0);
    }
    theta_new[i] = mu_part / sig_part;
  }

  if (theta_new[theta_constraint] < 0) theta_new = -theta_new;
  if (theta_strict_identification)
    theta_new = (theta_new - arma::mean(theta_new)) / arma::stddev(theta_new);
}

void CppIRTModel::save_update_history(int iter) {
  update_histories(iter, 0) = 1.0 - arma::cor(alpha_new, alpha).min();
  update_histories(iter, 1) = 1.0 - arma::cor(beta_new, beta).min();
  update_histories(iter, 2) = 1.0 - arma::cor(theta_new, theta).min();

  alpha = alpha_new;
  beta = beta_new;
  theta = theta_new;
}

void CppIRTModel::convergence_check(int iter) {
  convergence_metric = update_histories.row(iter).max();
  if (convergence_metric < tol) converged = true;
}