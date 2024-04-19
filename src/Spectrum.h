#ifndef SPECTRUM_H
#define SPECTRUM_H



// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>


/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/



struct EigenPair;


//' Compute the Laplacian eigenmap by local anchor embedding
//'
//' @param X A numeric matrix with dim (n,d),
//' each row indicates one point in R^d.
//' @param s An integer, the number of the induced points.
//' @param r An integer, the number of the nearest neighbor points, defaulting `3`.
//' @param ndim An integer, the dimension of the manifold projection, defaulting `2`.
//' @param subsample A character vector in c("kmeans", "random"), indicates the method
//' of subsampling, defaulting "kmeans".
//' @param norm A character vector in c("rw", "normalized", "cluster-normalized"),
//' indicates how to construct the stochastic transition matrix. "rw" means random walk,
//' "normalized" means normalized random walk, "cluster-normalized" means
//' normalized random walk with cluster membership re-balance. The defaulting gl
//' is "cluster-normalized".
//' @param nstart An integer, the number of random sets chosen in kmeans, defaulting `1`.
//'
//' @returns List of two component,
//' \describe{
//' \item{eigenvalues}{A numeric vector with length(ndim), the eigenvalues of the graph Laplacian.}
//' \item{eigenvectors}{A numeric matrix with dim(n,ndim), the eigenvector of the graph Laplacian.
//' Each row indicates the embedding coordinate representation of one point.}
//' }
//' @export
// [[Rcpp::export(lae_eigenmap)]]
Rcpp::List lae_eigenmap(const Eigen::MatrixXd & X,
                        int s, int r = 3, int ndim = 2, std::string subsample = "kmeans", std::string norm = "cluster-normalized", int nstart = 1);


// [[Rcpp::export(heat_kernel_covariance_cpp)]]
Eigen::MatrixXd heat_kernel_covariance_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & X_new,
                                           int s, int r, double t, int K,
                                           Rcpp::List models, int nstart);

// Compute spectrum of graph Laplacian by FLGP
EigenPair heat_kernel_spectrum_cpp(const Eigen::MatrixXd & X, const Eigen::MatrixXd & X_new,
                                   int s, int r, int K = -1, const Rcpp::List & models = \
                                     Rcpp::List::create(Rcpp::Named("subsample")="kmeans",
                                                        Rcpp::Named("kernel")="lae",
                                                        Rcpp::Named("gl")="rw",
                                                        Rcpp::Named("root")=false),
                                   int nstart = 1);


// Construct heat kernel covariance matrix from the spectrum of W
Eigen::MatrixXd HK_from_spectrum_cpp(const EigenPair & eigenpair, int K, double t,
                                     const Eigen::VectorXi & idx0,
                                     const Eigen::VectorXi & idx1);





//' Compute cross similarity matrix Z between X and U
//'
//' @param X A numeric matrix with dim (n,d), original sample,
//' each row indicates one original point in R^d.
//' @param U A numeric matrix with dim (s,d) or (s,d+1), sub-sample,
//' each row indicates one representative point in R^d,
//' where the d+1 column indicates the number of points in each cluster if it exists.
//' @param r An integer, the number of the nearest neighbor points.
//' @param gl A character vector in c("rw", "normalized", "cluster-normalized"),
//' indicates how to construct the stochastic transition matrix. "rw" means random walk,
//' "normalized" means normalized random walk, "cluster-normalized" means
//' normalized random walk with cluster membership re-balance. The defaulting gl
//' is "rw".
//'
//' @returns `Z` A numeric sparse dgr matrix with dim (n,s),
//' the stochastic transition matrix from X to U.
//' @export
// [[Rcpp::export(cross_similarity_lae_cpp)]]
Eigen::SparseMatrix<double,Eigen::RowMajor> cross_similarity_lae_cpp(
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & U,
    int r = 3,
    Rcpp::String gl = "rw");


// Truncated SVD, compute the non-trivial spectrums of A%\*%t(A) by calculating the spectrums
// of t(A)%*%A firstly.
EigenPair truncated_SVD_cpp(const Eigen::SparseMatrix<double,Eigen::RowMajor> & Z,
                             int K = -1);


// The spectrum of the similarity matrix W from Z
EigenPair spectrum_from_Z_cpp(const Eigen::SparseMatrix<double,Eigen::RowMajor> & Z,
                               int K,
                               bool root = false);


struct EigenPair {
  Eigen::VectorXd values;
  Eigen::MatrixXd vectors;

  EigenPair(const Eigen::VectorXd & values, const Eigen::MatrixXd & vectors)
    : values(values), vectors(vectors) {}
  EigenPair() {}
};

#endif

