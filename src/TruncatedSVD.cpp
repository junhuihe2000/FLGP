// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>


#include "Utils.h"
#include "Spectrum.h"


EigenPair truncated_SVD_cpp(const Eigen::SparseMatrix<double,Eigen::RowMajor> & Z,
                            int K) {
  if(K<0) {
    K = Z.cols();
  }

  Eigen::MatrixXd vectors;
  Eigen::VectorXd values;
  if(K==Z.cols()) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(Z), Eigen::ComputeThinU|Eigen::ComputeThinV);
    vectors = svd.matrixU();
    values = svd.singularValues().array().square();
  } else {

    Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
    Rcpp::Function svds = RSpectra["svds"];
    Rcpp::List pairs = svds(Rcpp::Named("A")=Rcpp::wrap(Eigen::SparseMatrix<double>(Z)),
                            Rcpp::Named("k")=K,
                            Rcpp::Named("nu")=K,
                            Rcpp::Named("nv")=0);
    values = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(pairs["d"]).array().square();
    vectors = Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(pairs["u"]);
  }

  return EigenPair(values, vectors);
}
