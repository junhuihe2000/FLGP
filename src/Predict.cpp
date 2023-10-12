// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "PGLogitModel.h"
#include "Utils.h"
#include "train.h"


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


Eigen::VectorXd test_regression_cpp(const Eigen::MatrixXd & C,
                                    const Eigen::VectorXd & Y,
                                    const Eigen::MatrixXd & Cnv) {
  // Algorithm 2.1 in GPML
  Eigen::LLT<Eigen::MatrixXd> chol_C(C);
  Eigen::VectorXd alpha = chol_C.solve(Y);
  Eigen::VectorXd Y_pred = Cnv*alpha;
  return Y_pred;
}


Eigen::MatrixXd predict_regression_cpp(const EigenPair & eigenpair, const Eigen::MatrixXd & Y,
                                    const Eigen::VectorXi & idx0, const Eigen::VectorXi & idx1,
                                    int K, double t, double noise, double sigma) {
  int m = Y.rows();
  if(m<=K) {
    Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, t, idx0, idx0);
    Eigen::MatrixXd C_noisy = Cvv;
    C_noisy.diagonal().array() += sigma;
    C_noisy.diagonal().array() += noise;
    Eigen::MatrixXd Cnv = HK_from_spectrum_cpp(eigenpair, K, t, idx1, idx0);

    // Algorithm 2.1 in GPML
    Eigen::LLT<Eigen::MatrixXd> chol_C(C_noisy);
    Eigen::MatrixXd alpha = chol_C.solve(Y);
    Eigen::MatrixXd Y_pred = Cnv*alpha;
    return Y_pred;
  } else {
    Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(K).array();
    const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
    Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(K,0,K-1);

    Eigen::MatrixXd V = mat_indexing(eigenvectors, idx0, cols);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Lambda_sqrt = (Eigen::exp(-0.5*t*eigenvalues.array())+0.0).matrix().asDiagonal();
    Eigen::MatrixXd Q = Lambda_sqrt*V.transpose()*V*Lambda_sqrt;
    Q.diagonal().array() += noise + sigma;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::MatrixXd alpha = 1.0/(noise+sigma)*(Y - V*Lambda_sqrt*chol_Q.solve(Lambda_sqrt*(V.transpose()*Y)));

    Eigen::MatrixXd Vnv = mat_indexing(eigenvectors, idx1, cols);
    Eigen::MatrixXd Y_pred = Vnv*(Eigen::exp(-t*eigenvalues.array()+0.0).matrix().asDiagonal()*(V.transpose()*alpha));
    return Y_pred;
  }
}
