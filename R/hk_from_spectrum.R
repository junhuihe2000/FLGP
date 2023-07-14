#' Construct heat kernel covariance matrix from the spectrum of W
#' @description The graph Laplacian L = I - W.
#'
#' @param eigenpairs A list of converged eigenvalues and eigenvectors of W,
#' including `values` and `vectors`.
#' @param K An integer, the number of used eigenpairs.
#' @param t A non-negative number, the heat diffusion time.
#' @param idx0, An integer vector with length(m0), index vector.
#' @param idx1, An integer vector with length(m1), index vector.
#'
#' @return A numeric matrix with dim (m0,m1), the covariance matrix of heat kernel
#' Gaussian processes.
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' X <- Z%*%t(Z)
#' eigens <- eigen(X)
#' HK_from_spectrum(eigens, 2, 0.1, c(1:2), c(1:3))
HK_from_spectrum <- function(eigenpairs, K, t, idx0=NULL, idx1=NULL) {
  if(is.null(K)) {
    K = ncol(eigenpairs$vectors)
  }

  n = nrow(eigenpairs$vectors)
  if(is.null(idx0)) {
    idx0 = c(1:n)
  }
  if(is.null(idx1)) {
    idx1 = c(1:n)
  }

  stopifnot(K<=length(eigenpairs$values),
            abs(K-round(K))<.Machine$double.eps^0.5,
            0<=t)

  eigenvalues = 1 - eigenpairs$values[1:K]
  H_t = Matrix::colScale(eigenpairs$vectors[idx0,1:K], exp(-eigenvalues*t))%*%
    Matrix::t(eigenpairs$vectors[idx1,1:K])
  return(H_t)
}
