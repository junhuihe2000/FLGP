#ifndef SPECTRUM_H
#define SPECTRUM_H


// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

struct EigenPair;



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
//' @return `Z` A numeric sparse dgr matrix with dim (n,s),
//' the stochastic transition matrix from X to U.
//' @export
//'
//' @examples
//' X <- matrix(rnorm(5*2), 5, 2)
//' U <- subsample(X, 2, "random")
//' r <- 2
//' cross_similarity_lae_cpp(X, U, r)
// [[Rcpp::export(cross_similarity_lae_cpp)]]
Eigen::SparseMatrix<double,Eigen::RowMajor> cross_similarity_lae_cpp(
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & U,
    int r = 3,
    Rcpp::String gl = "rw");


/*
//' Truncated SVD, compute the non-trivial spectrums of A%\*%t(A) by calculating the spectrums
//' of t(A)%*%A firstly.
//'
//' @param A A (n,s) numeric matrix, sparseMatrix is supported as well.
//' The eigenvalues and eigenvectors of A%*%t(A) will be computed.
//' @param K An integer, the number of eigenvalues requested.
//'
//' @return A list of converged eigenvalues and eigenvectors of AA^T. The return results
//' should be the same as svd(A, nu=K, nv=0). When ncol(A) << nrow(A), truncated_svd
//' will be much faster than svd:
//' \describe{
//' \item{values}{a vector containing the K eigenvalues of AA^T, sorted in decreasing order.}
//' \item{vectors}{a n \* K matrix whose columns contain the eigenvectors of AA^T.
//' The vectors are normalized to unit length.}
//' }
//' @export
//'
//' @examples
//' A <- abs(Matrix::sparseMatrix(i=c(1:5),j=sample.int(5),x=rnorm(5),repr = "R"))
//' truncated_SVD_cpp(A, 3)
// [[Rcpp::export(truncated_SVD_cpp)]]
*/
EigenPair truncated_SVD_cpp(const Eigen::SparseMatrix<double,Eigen::RowMajor> & Z,
                             int K = -1);


/*
//' The spectrum of the similarity matrix W from Z
//' @description The transition matrix W is the two step composition of
//' the cross similarity matrix Z by $W=ZLambda^{-1}Z^T$.
//'
//' @param Z A numeric matrix with dim (n,s), the cross similarity matrix
//' between the original sample and the sub-sample.
//' @param K An integer, the number of eigenpairs requested, the defaulting value
//' is NULL, indicating all non-trivial eigenpairs, that is, K=min(n,s).
//' @param root A logical value, indicating whether to square root eigenvalues of W,
//' the defaulting value is FALSE.
//'
//' @return A list of converged eigenvalues and eigenvectors of W.
//' \describe{
//' \item{values}{eigenvalues, descending order.}
//' \item{vectors}{eigenvectors, the vectors are normalized to sqrt(n) length.}
//' }
//' @export
//'
//' @examples
//' Z <- Matrix::sparseMatrix(i=c(1:5),j=sample.int(5),x=abs(rnorm(5)),repr = "R")
//' K <- 2
//' spectrum_from_Z_cpp(Z, K)
*/
EigenPair spectrum_from_Z_cpp(const Eigen::SparseMatrix<double,Eigen::RowMajor> & Z,
                               int K = -1,
                               bool root = false);


struct EigenPair {
  Eigen::VectorXd values;
  Eigen::MatrixXd vectors;

  EigenPair(const Eigen::VectorXd & values, const Eigen::MatrixXd & vectors)
    : values(values), vectors(vectors) {}
  EigenPair() {}
};

#endif

