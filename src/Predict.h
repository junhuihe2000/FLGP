#ifndef PREDICT_H
#define PREDICT_H


// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

/*
using namespace Rcpp;
using namespace Eigen;
*/

/*-----------------------------------------------------------------*/


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
//' @return `list(Y_pred)` if `output_pi=FALSE`, otherwise `list(Y_pred,pi_pred)`.
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
                            const Eigen::VectorXd & Y,
                            const Eigen::MatrixXd & Cnv,
                            int N_sample = 100,
                            bool output_pi = false);

#endif
