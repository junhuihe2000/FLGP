// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "PGLogitModel.h"

using namespace Rcpp;
using namespace Eigen;

//' @importFrom Rcpp sourceCpp


//' Predict labels on new samples with Polya-Gamma
//'
//' @param C A numeric matrix with dim(m,m), the self covariance matrix
//' in the training samples.
//' @param Y A numeric vector with length(m), count of the positive class.
//' @param Cnv A numeric matrix with dim(m_new,m), cross covariance matrix
//' between new sample and training sample.
//' @param N_sample An integer, the length of the Gibbs sampler chain.
//' @param output_pi Bool, whether or not to output pi_new, defaulting value is `FALSE`.
//'
//' @return `list(Y_pred)` if `output_pi=FALSE`, otherwise `list(pi_pred,Y_pred)`.
//' @export
//'
//' @examples
//' Z <- matrix(rnorm(3*3),3,3)
//' C <- Z%*%t(Z)
//' Y <- sample(c(0,1),3, replace=TRUE)
//' Cnv <- matrix(rnorm(5*3),5,3)
//' test_pgbinary_cpp(C, Y, Cnv)
//[[Rcpp::export(test_pgbinary_cpp)]]
Rcpp::List test_pgbinary_cpp(const Eigen::MatrixXd & C,
                             const Eigen::VectorXi & Y,
                             const Eigen::MatrixXd & Cnv,
                             int N_sample = 100,
                             bool output_pi = false) {
  PGLogitModel pglogit(C, Y);
  pglogit.resample_model(N_sample);
  Eigen::VectorXd pi_pred = pglogit.predict(Cnv);
  Eigen::VectorXi Y_pred = pi_to_Y(pi_pred);
  if(output_pi) {
    return Rcpp::List::create(Named("pi_pred")=pi_pred, Named("Y_pred")=Y_pred);
  }
  else {
    return Rcpp::List::create(Named("Y_pred")=Y_pred);
  }
}

