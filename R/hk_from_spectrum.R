#' Construct heat kernel covariance matrix from the spectrum of W
#' @description The graph Laplacian L = I - W.
#'
#' @param eigenpairs A list of converged eigenvalues and eigenvectors of W,
#' including `values` and `vectors`.
#' @param m An integer, the number of labeled samples.
#' @param K An integer, the number of used eigenpairs.
#' @param t A non-negative number, the heat diffusion time.
#'
#' @return A numeric matrix with dim (n,m), the covariance matrix of heat kernel
#' Gaussian processes.
#' @export
#'
#' @examples
#' Z <- matrix(rnorm(3*3),3,3)
#' X <- Z%*%t(Z)
#' eigens <- eigen(X)
#' HK_from_spectrum(eigens, 2, 2, 0.1)
HK_from_spectrum <- function(eigenpairs, m, K, t) {
  stopifnot(K<=length(eigenpairs$values),
            m<=nrow(eigenpairs$vectors),
            abs(m-round(m))<.Machine$double.eps^0.5,
            abs(K-round(K))<.Machine$double.eps^0.5,
            0<=t)
  n = nrow(eigenpairs$vectors)
  eigenvalues = 1 - eigenpairs$values[1:K]
  H_t = n*Matrix::colScale(eigenpairs$vectors[,1:K], exp(-eigenvalues*t))%*%
    Matrix::t(eigenpairs$vectors[1:m,1:K])
  return(H_t)
}
