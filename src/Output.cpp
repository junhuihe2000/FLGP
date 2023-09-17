// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "Fit.h"


// [[Rcpp::export(fit_lae_logit_gp_output)]]
SEXP fit_lae_logit_gp_output(Rcpp::NumericMatrix X_train, Rcpp::NumericVector Y_train, Rcpp::NumericMatrix X_test,
                                  int s, int r, int K=-1, SEXP N_train=R_NilValue,
                                  double sigma=1e-3, std::string approach="posterior",
                                  SEXP models=R_NilValue,
                                  bool output_cov=false) {

  Rcpp::NumericVector _N_train;
  if(TYPEOF(N_train)==NILSXP) {
    _N_train = Rcpp::rep(1, Y_train.size());
  } else if(TYPEOF(N_train)==REALSXP) {
    _N_train = Rcpp::as<Rcpp::NumericVector>(N_train);
  } else {
    Rcpp::stop("Invalid SEXPTYPE of N_train!\n");
  }


  Rcpp::List _models;
  if(TYPEOF(models)==NILSXP) {
    _models = Rcpp::List::create(Rcpp::Named("subsample")="kmeans",
                                 Rcpp::Named("kernel")="lae",
                                 Rcpp::Named("gl")="rw",
                                 Rcpp::Named("root")=false);
  } else if(TYPEOF(models)==VECSXP) {
    _models = Rcpp::as<Rcpp::List>(models);
  } else {
    Rcpp::stop("Invalid SEXPTYPE of models!\n");
  }


  Rcpp::List res = fit_lae_logit_gp_cpp(X_train, Y_train, X_test, s, r, K, _N_train, sigma, approach, _models, output_cov);

  if(output_cov) {
    return res;
  } else{
    return Rcpp::wrap(res["Y_pred"]);
  }

}
