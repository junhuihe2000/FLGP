#' Truncated SVD, compute the non-trivial spectrums of A%\*%t(A) by calculating the spectrums
#' of t(A)%*%A firstly.
#'
#' @param A A (n,s) numeric matrix, sparseMatrix is supported as well.
#' The eigenvalues and eigenvectors of A%*%t(A) will be computed.
#' @param K An integer, the number of eigenvalues requested.
#'
#' @return A list of converged eigenvalues and eigenvectors of AA^T. The return results
#' should be the same as svd(A, nu=K, nv=0). When ncol(A) << nrow(A), truncated_svd
#' will be much faster than svd:
#' \describe{
#' \item{values}{a vector containing the K eigenvalues of AA^T, sorted in decreasing order.}
#' \item{vectors}{a n \* K matrix whose columns contain the eigenvectors of AA^T.
#' The vectors are normalized to unit length.}
#' }
#' @export
#'
#' @examples
#' A <- matrix(rnorm(5*2), 5, 2)
#' truncated_SVD(A, 2)
truncated_SVD <- function(A, K=NULL) {
  stopifnot(methods::is(A, "Matrix")||methods::is(A, "matrix"))
  if(ncol(A)>nrow(A)) {
    warning("Warning: the truncated_svd() might be quite slow, use svd() instead.")
  }
  if(is.null(K)) {K=ncol(A)}
  if(K==ncol(A)) {
    # pairs = eigen(Matrix::t(A)%*%A)
    res = svd(A)
  } else {
    # pairs = RSpectra::eigs_sym(Matrix::t(A)%*%A, k=K)
    res = irlba::irlba(A, K, work = 2*K)

  }
  pairs = list(values=res$d^2, vectors=res$u)
  # pairs$vectors = as.matrix(Matrix::colScale(A%*%pairs$vectors, sqrt(pairs$values+1e-5)^{-1}))

  return(pairs)
}
