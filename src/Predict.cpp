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
                                        int K, const std::vector<double> & pars, double sigma,
                                        std::string noisepar) {
  int m = Y.rows();
  if(noisepar=="same") {
    double t = pars[0];
    double noise = pars[1];
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
  } else if(noisepar=="different") {
    double t = pars[0];
    if(m<=K) {
      Eigen::MatrixXd Cvv = HK_from_spectrum_cpp(eigenpair, K, t, idx0, idx0);
      Eigen::MatrixXd & C_noisy = Cvv;
      C_noisy.diagonal().array() += sigma;
      for(int i=1;i<=m;i++) {
        C_noisy.diagonal()[i-1] += pars[i];
      }
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
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> Z_inv(m);
      for(int i=1;i<=m;i++) {
        Z_inv.diagonal()[i-1] = 1.0/(pars[i]+sigma);
      }
      Eigen::MatrixXd VtZV = V.transpose()*Z_inv*V;
      Eigen::MatrixXd Q = Lambda_sqrt*VtZV*Lambda_sqrt;
      Q.diagonal().array() += 1.0;
      Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
      Eigen::MatrixXd alpha = Z_inv*Y - Z_inv*V*Lambda_sqrt*chol_Q.solve(Lambda_sqrt*(V.transpose()*(Z_inv*Y)));

      Eigen::MatrixXd Vnv = mat_indexing(eigenvectors, idx1, cols);
      Eigen::MatrixXd Y_pred = Vnv*(Eigen::exp(-t*eigenvalues.array()+0.0).matrix().asDiagonal()*(V.transpose()*alpha));
      return Y_pred;
    }
  } else {
    Rcpp::stop("The noise setting is illegal!");
  }

  }


Eigen::MatrixXd predict_rbf_regression_cpp(const Eigen::MatrixXd & Y,
                                           const Eigen::MatrixXd & dist_UU, const Eigen::MatrixXd & dist_XU,
                                           const Eigen::MatrixXd & dist_XnewU,
                                           int s, const std::vector<double> & pars, double sigma,
                                           std::string noisepar) {
  int m = Y.rows();
  if(noisepar=="same") {
    double t = pars[0];
    double noise = pars[1];

    Eigen::MatrixXd C_ss = Eigen::exp(-dist_UU.array()/(2*t));
    Eigen::MatrixXd C_ms = Eigen::exp(-dist_XU.array()/(2*t));

    Eigen::MatrixXd Q = (noise+sigma)*C_ss + C_ms.transpose()*C_ms;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::MatrixXd alpha = chol_Q.solve(C_ms.transpose()*Y);

    Eigen::MatrixXd C_ns = Eigen::exp(-dist_XnewU.array()/(2*t));

    Eigen::MatrixXd Y_pred = C_ns*alpha;
    return Y_pred;
  } else if(noisepar=="different") {
    double t = pars[0];
    Eigen::MatrixXd C_ss = Eigen::exp(-dist_UU.array()/(2*t));
    Eigen::MatrixXd C_ms = Eigen::exp(-dist_XU.array()/(2*t));

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Z(m);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Z_inv(m);
    for(int i=1;i<=m;i++) {
      Z.diagonal()[i-1] = pars[i] + sigma;
      Z_inv.diagonal()[i-1] = 1.0/(pars[i]+sigma);
    }

    Eigen::MatrixXd Q = C_ss + C_ms.transpose()*Z_inv*C_ms;
    Eigen::LLT<Eigen::MatrixXd> chol_Q(Q);
    Eigen::MatrixXd alpha = chol_Q.solve(C_ms.transpose()*Z_inv*Y);

    Eigen::MatrixXd C_ns = Eigen::exp(-dist_XnewU.array()/(2*t));

    Eigen::MatrixXd Y_pred = C_ns*alpha;
    return Y_pred;
  } else {
    Rcpp::stop("The noise setting is illegal!");
  }
}
