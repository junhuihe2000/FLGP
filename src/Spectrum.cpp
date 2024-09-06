// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>


#include "Utils.h"
#include "lae.h"
#include "Spectrum.h"

#include <iostream>


/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/


Rcpp::List lae_eigenmap(const Eigen::MatrixXd & X,
                        int s, int r, int ndim, std::string subsample, std::string norm, int nstart) {
  Eigen::MatrixXd U = subsample_cpp(X, s, subsample, nstart);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Z = cross_similarity_lae_cpp(X, U, r, norm);
  EigenPair eigenpair = spectrum_from_Z_cpp(Z, ndim, true);

  return Rcpp::List::create(Rcpp::Named("eigenvalues") = 1-eigenpair.values.array(),
                            Rcpp::Named("eigenvectors") = eigenpair.vectors);
}


Eigen::MatrixXd heat_kernel_covariance_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & X_new,
                                           int s, int r, double t, int K,
                                           Rcpp::List models, int nstart, double epsilon) {
  if (K<0) {
    K = s;
  }
  EigenPair eigenpair = heat_kernel_spectrum_cpp(X, X_new, s, r, K, models, nstart, epsilon);

  int m = X.rows(); int m_new = X_new.rows();
  int n = m + m_new;
  Eigen::VectorXi idx0 = Eigen::VectorXi::LinSpaced(n, 0, n-1);
  Eigen::VectorXi idx1 = Eigen::VectorXi::LinSpaced(m, 0, m-1);
  Eigen::MatrixXd H = HK_from_spectrum_cpp(eigenpair, K, t, idx0, idx1);

  return H;
}




EigenPair heat_kernel_spectrum_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & X_new,
                                   int s, int r, int K, const Rcpp::List & models, int nstart, double epsilon) {
  int m = X.rows(); int m_new = X_new.rows();
  Eigen::MatrixXd X_all(m+m_new, X.cols());
  X_all.topRows(m) = X;
  X_all.bottomRows(m_new) = X_new;

  Eigen::MatrixXd U = subsample_cpp(X_all, s, Rcpp::as<std::string>(models["subsample"]), nstart);
  // Eigen::SparseMatrix<double, Eigen::RowMajor> Z = cross_similarity_lae_cpp(X_all, U, r, Rcpp::as<std::string>(models["gl"]));


  Eigen::SparseMatrix<double, Eigen::RowMajor> Z;
  std::string kernel_type = Rcpp::as<std::string>(models["kernel"]);
  if(kernel_type=="lae") {
    Z = cross_similarity_lae_cpp(X_all, U, r, Rcpp::as<std::string>(models["gl"]));
  } else if(kernel_type=="se") {
    Z = cross_similarity_se_cpp(X_all, U, r, Rcpp::as<std::string>(models["gl"]), epsilon);
  } else {
    Rcpp::Rcout << "The kernel type is not supported!\n";
  }

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


Eigen::SparseMatrix<double,Eigen::RowMajor> cross_similarity_se_cpp(
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & U,
    int r,
    Rcpp::String gl, double epsilon) {
  int d = X.cols();
  Rcpp::List res_knn = KNN_cpp(X, U.leftCols(d), r, "Euclidean", true);
  const Eigen::MatrixXi& ind_knn = res_knn["ind_knn"];
  const Eigen::SparseMatrix<double, Eigen::RowMajor> & distances_sp = res_knn["distances_sp"];

  Eigen::SparseMatrix<double, Eigen::RowMajor> Z = distances_sp;

  Z.coeffs() = Eigen::exp(-distances_sp.coeffs()/(4*epsilon*epsilon));

  Eigen::VectorXd num_class;
  if(gl=="cluster-normalized") {
    num_class = U.col(d);
  }

  graphLaplacian_cpp(Z, gl, num_class);

  return Z;
}



EigenPair spectrum_from_Z_cpp(const Eigen::SparseMatrix<double,Eigen::RowMajor> & Z,
                               int K,
                               bool root) {
  Eigen::VectorXd Z_colsum = Eigen::RowVectorXd::Ones(Z.rows()) * Z;
  Eigen::SparseMatrix<double,Eigen::RowMajor> A = Z*(1.0/(Z_colsum.array().abs()+1e-9).sqrt()).matrix().asDiagonal();
  EigenPair pairs = truncated_SVD_cpp(A, K);

  if(root) {
    pairs.values = pairs.values.array().sqrt().matrix();
  }

  double n = Z.rows();
  pairs.vectors *= std::sqrt(n);

  return pairs;
}


