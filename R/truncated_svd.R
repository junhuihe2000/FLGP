#' Truncated SVD, compute the non-trivial spectrums of A%*%t(A) by calculating the spectrums
#' of t(A)%*%A firstly.
#'
#' @param A A (n,s) numeric matrix, sparseMatrix is supported as well.
#' The eigenvalues and eigenvectors of A%*%t(A) will be computed.
#' @param K An integer, the number of eigenvalues requested.
#'
#' @return A list of converged eigenvalues and eigenvectors. The return results
#' should be the same as svd(A, nu=K, nv=0). When ncol(A) << nrow(A), truncated_svd
#' will be much faster than svd.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(100*10), 100, 10)
#' truncated_SVD(A, 3)
truncated_SVD <- function(A, K=NULL) {
  stopifnot(methods::is(A, "Matrix")||methods::is(A, "matrix"))
  if(ncol(A)>nrow(A)) {
    warning("Warning: the truncated_svd() might be quite slow, use svd() instead.")
  }
  if(is.null(K)) {K=ncol(A)}
  if(K==ncol(A)) {
    pairs = eigen(Matrix::t(A)%*%A)
  } else {
    pairs = RSpectra::eigs_sym(Matrix::t(A)%*%A, k=K)
  }
  pairs$vectors = Matrix::colScale(A%*%pairs$vectors, sqrt(pairs$values+1e-5)^{-1})

  return(pairs)
}
