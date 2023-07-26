// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// #include <Spectra/SymEigsSolver.h>
// #include <Spectra/MatOp/SparseSymMatProd.h>

#include "Utils.h"
#include "lae.h"
#include "Spectrum.h"

using namespace Rcpp;
using namespace Eigen;



Eigen::SparseMatrix<double,Eigen::RowMajor> cross_similarity_lae_cpp(
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & U,
    int r,
    Rcpp::String gl) {
  int d = X.cols();
  Eigen::SparseMatrix<double,Eigen::RowMajor> Z = LAE_cpp(X, U.leftCols(d), r);

  Eigen::VectorXd num_class;
  if(gl=="cluster-normalized") {
    num_class = U.col(d);
  }

  graphLaplacian_cpp(Z, gl, num_class);

  return Z;
}



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
    /*
    // Construct matrix operation object using the wrapper class SparseSymMatProd
    Spectra::SparseSymMatProd<double> W_op((Z.transpose()*Z).pruned());
    // Construct eigen solver object, requesting the largest three eigenvalues
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(W_op, K, 2*K);
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestAlge);
    if(eigs.info() == Spectra::CompInfo::Successful) {
      values = eigs.eigenvalues();
      vectors = Z * (eigs.eigenvectors() * (1.0/values.array().sqrt()).matrix().asDiagonal());
    } else {
      Rcpp::stop("Truncated SVD fails!");
    }
    */
    Rcpp::Environment RSpectra = Rcpp::Environment::namespace_env("RSpectra");
    Rcpp::Function eigs_sym = RSpectra["eigs_sym"];
    Rcpp::List pairs = eigs_sym(Rcpp::Named("A")=Rcpp::wrap(Z.transpose()*Z),
                                Rcpp::Named("k")=K);
    values = Rcpp::as<Eigen::VectorXd>(pairs["values"]);
    vectors = Rcpp::as<Eigen::MatrixXd>(pairs["vectors"]);
    vectors = Z * (vectors * (1.0/values.array().sqrt()).matrix().asDiagonal());
  }

  return EigenPair(values, vectors);
}




EigenPair spectrum_from_Z_cpp(const Eigen::SparseMatrix<double,Eigen::RowMajor> & Z,
                               int K,
                               bool root) {
  Eigen::VectorXd Z_colsum = Eigen::RowVectorXd::Ones(Z.rows()) * Z;
  Eigen::SparseMatrix<double,Eigen::RowMajor> A = Z*(1.0/Z_colsum.array().sqrt()).matrix().asDiagonal();
  EigenPair pairs = truncated_SVD_cpp(A, K);

  if(root) {
    pairs.values = pairs.values.array().sqrt().matrix();
  }

  double n = Z.rows();
  pairs.vectors *= std::sqrt(n);

  return pairs;
}
