#' Robustly predict response on new samples for Gaussian regression
#'
#' @param eigenpair A list includes values and vectors.
#' @param Y A numeric vector with length(m), response vector.
#' @param theta A numeric vector, hyper-parameters.
#' @param K An integer, the number of used eigenpairs.
#'
#' @return `Y_pred` A numeric vector with length(m_new), predictive response vector.
#' @export
#'
#' @examples
#' X <- matrix(rnorm(3*3), 3, 3)
#' Y <- rowSums(X)
#' X_new <- matrix(rnorm(10*3),10,3)
#' s <- 5; r <- 3
#' m <- 3; K <- 3
#' eigenpair <- heat_kernel_spectrum(X, X_new, s, r, K=K)
#' theta <- c(1,0.1)
#' K <- 3
#' test_reg_robust(eigenpair, Y, theta, K)
test_reg_robust <- function(eigenpair, Y, theta, K) {
  # using equation 8.14 in GPML

  m = length(Y)
  vectors = eigenpair$vectors[,1:K]
  values = eigenpair$values
  n = nrow(vectors)

  if(FALSE) {
  Q = Matrix::crossprod(vectors[1:m,]) + (theta[2]^2)*diag(exp((1-values)*theta[1])) + 1e-9*diag(1,K)
  Y_pred = vectors[(m+1):n,]%*%Matrix::solve(Q, Matrix::t(vectors[1:m,])%*%cbind(Y))
  }

  Umk = Matrix::colScale(vectors[1:m,], exp(-0.5*theta[1]*(1-values)))
  Q = Matrix::crossprod(Umk) + (theta[2]^2+1e-9)*diag(1,K)
  Unk = Matrix::colScale(vectors[(m+1):n,], exp(-0.5*theta[1]*(1-values)))
  a = Matrix::t(Umk)%*%Y
  Y_pred = Unk%*%Matrix::solve(Q, a)

  return(as.vector(Y_pred))
}
