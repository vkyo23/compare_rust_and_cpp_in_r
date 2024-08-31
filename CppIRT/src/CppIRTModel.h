#ifndef __CPPIRTMODEL__INCLUDED__
#define __CPPIRTMODEL__INCLUDED__

#include <RcppArmadillo.h>

class CppIRTModel {
 public:
  CppIRTModel(const arma::mat &Y, arma::vec alpha, arma::vec beta,
              arma::vec theta, const double &a0, const double &A0,
              const double &b0, const double &B0, const int &theta_constraint,
              const bool &theta_strict_identification, const int &maxit,
              const int &verbose, const double &tol);
  ~CppIRTModel();

  void fit();
  Rcpp::List output();

 private:
  void update_all_parameters();
  void calc_EOmega();
  void update_alpha();
  void update_beta();
  void update_theta();
  void save_update_history(int iter);
  void convergence_check(int iter);

  arma::mat S;
  const int I, J;

  arma::vec alpha, beta, theta;
  arma::mat Omega;
  arma::vec alpha_new, beta_new, theta_new;

  const double &a0, &A0, &b0, &B0;

  const int &theta_constraint, &maxit, &verbose;
  const bool &theta_strict_identification;
  const double &tol;

  arma::mat update_histories;
  bool converged;
  double convergence_metric;
  int iter;
};

#endif