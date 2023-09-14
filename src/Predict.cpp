// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "PGLogitModel.h"
#include "Utils.h"


/*
using namespace Rcpp;
using namespace Eigen;
*/


Rcpp::List test_pgbinary_cpp(const Eigen::MatrixXd & C,
                             const Eigen::VectorXd & Y,
                             const Eigen::MatrixXd & Cnv,
                             int N_sample,
                             bool output_pi) {
  PGLogitModel pglogit(C, Y);
  pglogit.resample_model(N_sample);
  Eigen::VectorXd pi_pred = pglogit.predict(Cnv);
  Eigen::VectorXd Y_pred = pi_to_Y(pi_pred);
  if(output_pi) {
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred, Rcpp::Named("pi_pred")=pi_pred);
  }
  else {
    return Rcpp::List::create(Rcpp::Named("Y_pred")=Y_pred);
  }
}

