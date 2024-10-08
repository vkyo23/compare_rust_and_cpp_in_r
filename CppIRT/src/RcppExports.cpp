// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// fit_CppIRT
Rcpp::List fit_CppIRT(const arma::mat& Y, arma::vec alpha, arma::vec beta, arma::vec theta, const double& a0, const double& A0, const double& b0, const double& B0, const int& theta_constraint, const bool& theta_strict_identification, const int& maxit, const int& verbose, const double& tol);
RcppExport SEXP _CppIRT_fit_CppIRT(SEXP YSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP thetaSEXP, SEXP a0SEXP, SEXP A0SEXP, SEXP b0SEXP, SEXP B0SEXP, SEXP theta_constraintSEXP, SEXP theta_strict_identificationSEXP, SEXP maxitSEXP, SEXP verboseSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const double& >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double& >::type A0(A0SEXP);
    Rcpp::traits::input_parameter< const double& >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const double& >::type B0(B0SEXP);
    Rcpp::traits::input_parameter< const int& >::type theta_constraint(theta_constraintSEXP);
    Rcpp::traits::input_parameter< const bool& >::type theta_strict_identification(theta_strict_identificationSEXP);
    Rcpp::traits::input_parameter< const int& >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const int& >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_CppIRT(Y, alpha, beta, theta, a0, A0, b0, B0, theta_constraint, theta_strict_identification, maxit, verbose, tol));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_CppIRT_fit_CppIRT", (DL_FUNC) &_CppIRT_fit_CppIRT, 13},
    {NULL, NULL, 0}
};

RcppExport void R_init_CppIRT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
