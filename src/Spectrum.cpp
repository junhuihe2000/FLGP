// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// #include <Spectra/SymEigsSolver.h>
// #include <Spectra/MatOp/SparseSymMatProd.h>

#include "Utils.h"
#include "lae.h"
#include "Spectrum.h"

using namespace Rcpp;
using namespace Eigen;


/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/


EigenPair heat_kernel_spectrum_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & X_new,
                                   int s, int r, int K, const Rcpp::List & models) {
  int m = X.rows(); int m_new = X_new.rows();
  Eigen::MatrixXd X_all(m+m_new, X.cols());
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  Eigen::MatrixXd U = subsample_cpp(X_all, s, models["subsample"]);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Z = cross_similarity_lae_cpp(X_all, U, r, models["gl"]);

  if(K<0) {
    K = s;
  }

  EigenPair eigenpair = spectrum_from_Z_cpp(Z, K, models["root"]);

  return eigenpair;
}


Eigen::MatrixXd HK_from_spectrum_cpp(const EigenPair & eigenpair, int K, double t,
                                     const Eigen::VectorXi & idx0,
                                     const Eigen::VectorXi & idx1) {
  Eigen::VectorXd eigenvalues = 1 - eigenpair.values.head(K).array();
  const Eigen::MatrixXd & eigenvectors = eigenpair.vectors;
  Eigen::VectorXi cols = Eigen::VectorXi::LinSpaced(K,0,K-1);

  Eigen::MatrixXd H = mat_indexing(eigenvectors, idx0, cols)*Eigen::exp(-t*eigenvalues.array()).matrix().asDiagonal()\
    *mat_indexing(eigenvectors, idx1, cols).transpose();

  return H;
}






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
